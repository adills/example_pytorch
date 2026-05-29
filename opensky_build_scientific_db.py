"""
Build and query a filtered PostgreSQL database from OpenSky scientific datasets.

Purpose
-------
This module creates a reduced local PostgreSQL database from two public OpenSky
scientific datasets:

- The COVID-19 flight dataset hosted on Zenodo
- The weekly state-vector dataset hosted on OpenSky's public S3 bucket

The builder is intentionally selective to keep the local database size bounded:

- Only flights departing from a configured set of origin airports are kept
- Only long-haul flights above a configured duration threshold are kept
- Only state-vector rows that match those filtered flights are stored

The resulting database can then be queried locally to provide a stable data
source for ``opensky_example.py`` or other downstream scripts.

Default scope
-------------
The default airport set is:

- EGLL (London Heathrow)
- WSSS (Singapore Changi)

This provides one large European hub and one geographically distant Asia-Pacific
hub, while still keeping the stored data smaller than a full global ingest.
These origins are fully configurable at build time.

Schema
------
Two tables are created:

- ``scientific_flights``
  Filtered long-haul flight metadata from the COVID dataset.

- ``scientific_state_vectors``
  Only the state-vector rows that can be matched to those flights.

An index-backed ``flight_key`` ties trajectory rows to the corresponding flight.

CLI
---
Build the filtered database:

    python opensky_build_scientific_db.py build \
        --download-dir /Volumes/external/opensky

Query the local database with the default join:

    python opensky_build_scientific_db.py query \
        --origin-airport EGLL \
        --minimum-duration-hours 6 \
        --sample-trajectories 25

The ``build`` command will not re-download files that already exist in the
specified download directory.

Requirements
------------
- Python packages:
  - pandas
  - httpx
  - sqlalchemy
- A PostgreSQL driver compatible with SQLAlchemy, e.g. ``psycopg``
- A writable local download directory, such as an external drive

Local PostgreSQL setup
----------------------
This script assumes a local PostgreSQL database named
``opensky_scientific`` running on ``localhost`` unless you override the
database URL.

Recommended macOS setup with Homebrew:

Automated setup option:

Run the helper script in this repository:

    ./opensky_create_postgresql_db.sh

That script will:

- install ``postgresql@16`` with Homebrew if needed
- start the local PostgreSQL service
- add the PostgreSQL client ``bin`` directory to ``PATH`` for the script run,
  but only if that directory exists
- create the local ``opensky_scientific`` database if needed
- print the default SQLAlchemy URL used by this module

Manual setup option:

1. Install PostgreSQL:

       brew install postgresql@16

2. Start the local server:

       brew services start postgresql@16

3. Add the PostgreSQL client tools to ``PATH``:

       export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

4. Install the Python driver in the project environment:

       pipenv install psycopg[binary]

5. Create the local project database:

       createdb opensky_scientific

After that, the script's default database URL is usually sufficient:

    postgresql+psycopg://[your-macos-username]@localhost/opensky_scientific

The PostgreSQL server decides where the database files live on disk. With the
Homebrew setup above, the data directory is typically managed under
``/opt/homebrew/var/postgresql@16``. The large scientific source downloads are
separate and should be directed to an external drive with ``--download-dir``.

Testing
-------
Unit tests for the filtering and query-builder logic live in:

    tests/test_opensky_build_scientific_db.py

Run them with:

    /Users/anthonydills/.local/share/virtualenvs/example_pytorch-XJ8KPHaX/bin/python -m unittest discover -s tests -v

Notes
-----
- The COVID dataset provides flight-level grouping metadata such as origin,
  destination, and first/last seen timestamps.
- The weekly state-vector dataset does not provide origin/destination directly,
  so state rows are matched to filtered COVID flights by ``icao24``,
  normalized ``callsign``, and time-window overlap.
- The builder uses remote metadata endpoints and public bucket listing where
  possible, falling back to HTML scraping only if needed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import getpass
import gzip
import io
import json
from pathlib import Path
import re
import tarfile
import time
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import httpx
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional at runtime
    tqdm = None


COVID_RECORD_ID = "7923702"
COVID_RECORD_URL = f"https://zenodo.org/records/{COVID_RECORD_ID}"
COVID_RECORD_API_URL = f"https://zenodo.org/api/records/{COVID_RECORD_ID}"
STATES_BUCKET_LIST_URL = "https://s3.opensky-network.org/"
STATES_BUCKET_PREFIX = "data-samples/states/"
DEFAULT_ORIGIN_AIRPORTS = ("EGLL", "WSSS")
DEFAULT_MINIMUM_DURATION_HOURS = 6.0
DEFAULT_STATE_ARCHIVE_FORMAT = "csv"
DEFAULT_HTTP_TIMEOUT_SECONDS = 60.0
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
COVID_CHUNK_ROWS = 100_000
STATE_VECTOR_INSERT_BATCH_SIZE = 5_000
DEFAULT_DATABASE_NAME = "opensky_scientific"


def build_default_database_url() -> str:
    return (
        f"postgresql+psycopg://{getpass.getuser()}@localhost/"
        f"{DEFAULT_DATABASE_NAME}"
    )


DEFAULT_DATABASE_URL = build_default_database_url()


@dataclass(frozen=True)
class BuildConfig:
    database_url: str
    download_dir: Path
    origin_airports: tuple[str, ...] = DEFAULT_ORIGIN_AIRPORTS
    minimum_duration_hours: float = DEFAULT_MINIMUM_DURATION_HOURS
    state_archive_format: str = DEFAULT_STATE_ARCHIVE_FORMAT
    max_covid_files: int | None = None
    max_state_archives: int | None = None


def normalize_callsign(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().upper()
    return normalized or None


def build_flight_key(
    icao24: str,
    callsign: str | None,
    firstseen: pd.Timestamp,
) -> str:
    callsign_clean = normalize_callsign(callsign) or "UNKNOWN"
    firstseen_ts = pd.Timestamp(firstseen).tz_convert("UTC")
    return f"{icao24.lower()}|{callsign_clean}|{firstseen_ts.isoformat()}"


def progress(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit)


def get_engine(database_url: str) -> Engine:
    try:
        return create_engine(database_url, future=True)
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Failed to create the SQLAlchemy engine. Ensure the database URL is "
            "valid and that a PostgreSQL driver such as 'psycopg' is installed."
        ) from exc


def create_schema(engine: Engine) -> None:
    schema_sql = [
        """
        CREATE TABLE IF NOT EXISTS scientific_flights (
            flight_key TEXT PRIMARY KEY,
            icao24 TEXT NOT NULL,
            callsign TEXT,
            origin TEXT NOT NULL,
            destination TEXT,
            firstseen TIMESTAMP WITH TIME ZONE NOT NULL,
            lastseen TIMESTAMP WITH TIME ZONE NOT NULL,
            duration_hours DOUBLE PRECISION NOT NULL,
            flight_number TEXT,
            registration TEXT,
            typecode TEXT,
            source_dataset TEXT NOT NULL,
            source_file TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scientific_state_vectors (
            id BIGSERIAL PRIMARY KEY,
            flight_key TEXT NOT NULL REFERENCES scientific_flights(flight_key)
                ON DELETE CASCADE,
            time TIMESTAMP WITH TIME ZONE NOT NULL,
            icao24 TEXT NOT NULL,
            callsign TEXT,
            lat DOUBLE PRECISION,
            lon DOUBLE PRECISION,
            velocity DOUBLE PRECISION,
            heading DOUBLE PRECISION,
            vertrate DOUBLE PRECISION,
            onground BOOLEAN,
            alert BOOLEAN,
            spi BOOLEAN,
            squawk TEXT,
            baroaltitude DOUBLE PRECISION,
            geoaltitude DOUBLE PRECISION,
            lastposupdate TIMESTAMP WITH TIME ZONE,
            lastcontact TIMESTAMP WITH TIME ZONE,
            serials TEXT,
            hour TIMESTAMP WITH TIME ZONE,
            source_dataset TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source_member TEXT NOT NULL,
            UNIQUE (flight_key, time, icao24, callsign)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scientific_flights_origin_firstseen
        ON scientific_flights (origin, firstseen)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_scientific_flights_icao24_callsign
        ON scientific_flights (icao24, callsign)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_state_vectors_flight_time
        ON scientific_state_vectors (flight_key, time)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_state_vectors_icao24_time
        ON scientific_state_vectors (icao24, time)
        """,
    ]

    with engine.begin() as connection:
        for statement in schema_sql:
            connection.execute(text(statement))


def http_get_text(url: str) -> str:
    with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def fetch_covid_file_metadata() -> list[dict[str, str]]:
    try:
        with httpx.Client(
            timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            response = client.get(COVID_RECORD_API_URL)
            response.raise_for_status()
            payload = response.json()
        files = []
        for entry in payload.get("files", []):
            key = entry.get("key")
            if key is None or not key.endswith(".csv.gz"):
                continue
            files.append(
                {
                    "name": key,
                    "url": entry["links"]["self"],
                }
            )
        if files:
            return files
    except Exception:
        pass

    html = http_get_text(COVID_RECORD_URL)
    pattern = re.compile(
        r'href="(?P<url>[^"]*flightlist_[^"]+\.csv\.gz[^"]*)".*?>(?P<name>flightlist_[^<]+\.csv\.gz)<',
        re.IGNORECASE | re.DOTALL,
    )
    matches: dict[str, str] = {}
    for match in pattern.finditer(html):
        name = match.group("name")
        url = urljoin(COVID_RECORD_URL, match.group("url"))
        matches[name] = url
    return [{"name": name, "url": url} for name, url in sorted(matches.items())]


def list_s3_objects(prefix: str) -> list[str]:
    keys: list[str] = []
    continuation_token: str | None = None

    with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if continuation_token is not None:
                params["continuation-token"] = continuation_token
            response = client.get(STATES_BUCKET_LIST_URL, params=params)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            namespace = {"s3": root.tag.partition("}")[0].strip("{")}
            keys.extend(
                element.text or ""
                for element in root.findall("s3:Contents/s3:Key", namespace)
            )
            is_truncated = root.findtext("s3:IsTruncated", "false", namespace)
            if is_truncated != "true":
                break
            continuation_token = root.findtext(
                "s3:NextContinuationToken",
                None,
                namespace,
            )
            if continuation_token is None:
                break
    return keys


def extract_date_from_name(name: str) -> date | None:
    for pattern in (
        r"(?P<date>\d{4}-\d{2}-\d{2})",
        r"(?P<date>\d{8})",
    ):
        match = re.search(pattern, name)
        if match is None:
            continue
        token = match.group("date")
        if "-" in token:
            return date.fromisoformat(token)
        return date.fromisoformat(f"{token[0:4]}-{token[4:6]}-{token[6:8]}")
    return None


def fetch_state_archive_metadata(
    archive_format: str = DEFAULT_STATE_ARCHIVE_FORMAT,
) -> list[dict[str, str]]:
    suffixes = (f".{archive_format}.tar", f".{archive_format}.tar.gz")
    keys = list_s3_objects(STATES_BUCKET_PREFIX)
    files = []
    for key in keys:
        if not key.endswith(suffixes):
            continue
        files.append(
            {
                "name": Path(key).name,
                "url": urljoin(STATES_BUCKET_LIST_URL, key),
            }
        )
    return sorted(files, key=lambda item: item["name"])


def download_file_if_missing(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing download: {destination}")
        return destination

    print(f"Downloading {url} -> {destination}")
    with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)
    return destination


def ensure_downloads(
    file_metadata: Sequence[dict[str, str]],
    download_dir: Path,
    *,
    subdir: str,
    limit: int | None = None,
) -> list[Path]:
    selected = list(file_metadata[:limit] if limit is not None else file_metadata)
    downloaded_paths = []
    for entry in progress(selected, desc=f"Download {subdir}", total=len(selected), unit="file"):
        destination = download_dir / subdir / entry["name"]
        downloaded_paths.append(download_file_if_missing(entry["url"], destination))
    return downloaded_paths


def parse_timestamp_column(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(numeric_series, unit="s", utc=True)


def filter_covid_chunk(
    chunk: pd.DataFrame,
    origin_airports: Sequence[str],
    minimum_duration_hours: float,
    source_file: str,
) -> pd.DataFrame:
    normalized_chunk = chunk.copy()
    normalized_chunk["origin"] = normalized_chunk["origin"].astype("string")
    normalized_chunk["destination"] = normalized_chunk["destination"].astype("string")
    normalized_chunk["callsign"] = (
        normalized_chunk["callsign"].astype("string").str.strip().str.upper()
    )
    normalized_chunk["icao24"] = (
        normalized_chunk["icao24"].astype("string").str.strip().str.lower()
    )
    normalized_chunk["firstseen"] = parse_timestamp_column(normalized_chunk["firstseen"])
    normalized_chunk["lastseen"] = parse_timestamp_column(normalized_chunk["lastseen"])
    normalized_chunk["duration_hours"] = (
        normalized_chunk["lastseen"] - normalized_chunk["firstseen"]
    ).dt.total_seconds() / 3600.0

    filtered = normalized_chunk[
        normalized_chunk["origin"].isin(origin_airports)
        & normalized_chunk["duration_hours"].ge(minimum_duration_hours)
        & normalized_chunk["origin"].notna()
        & normalized_chunk["firstseen"].notna()
        & normalized_chunk["lastseen"].notna()
    ].copy()
    if filtered.empty:
        return filtered

    filtered["flight_key"] = filtered.apply(
        lambda row: build_flight_key(
            row["icao24"],
            row["callsign"],
            row["firstseen"],
        ),
        axis=1,
    )
    filtered["source_dataset"] = "covid_flight_dataset"
    filtered["source_file"] = source_file
    return filtered[
        [
            "flight_key",
            "icao24",
            "callsign",
            "origin",
            "destination",
            "firstseen",
            "lastseen",
            "duration_hours",
            "number",
            "registration",
            "typecode",
            "source_dataset",
            "source_file",
        ]
    ].rename(columns={"number": "flight_number"})


def insert_records(
    engine: Engine,
    table_name: str,
    records: list[dict[str, Any]],
    conflict_columns: Sequence[str],
) -> int:
    if not records:
        return 0

    columns = list(records[0].keys())
    placeholders = ", ".join(f":{column}" for column in columns)
    column_sql = ", ".join(columns)
    conflict_sql = ", ".join(conflict_columns)
    statement = text(
        f"""
        INSERT INTO {table_name} ({column_sql})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_sql}) DO NOTHING
        """
    )
    with engine.begin() as connection:
        connection.execute(statement, records)
    return len(records)


def ingest_filtered_covid_flights(
    engine: Engine,
    covid_paths: Sequence[Path],
    *,
    origin_airports: Sequence[str],
    minimum_duration_hours: float,
) -> None:
    for covid_path in progress(covid_paths, desc="Ingest COVID flights", total=len(covid_paths), unit="file"):
        print(f"Processing flight manifest: {covid_path}")
        for chunk in pd.read_csv(covid_path, compression="gzip", chunksize=COVID_CHUNK_ROWS):
            filtered = filter_covid_chunk(
                chunk,
                origin_airports=origin_airports,
                minimum_duration_hours=minimum_duration_hours,
                source_file=covid_path.name,
            )
            if filtered.empty:
                continue
            records = filtered.to_dict(orient="records")
            insert_records(
                engine,
                "scientific_flights",
                records,
                conflict_columns=["flight_key"],
            )


def load_candidate_days_from_db(
    engine: Engine,
    origin_airports: Sequence[str],
    minimum_duration_hours: float,
) -> set[date]:
    flights_df = pd.read_sql_query(
        text(
            """
            SELECT firstseen, lastseen, origin, duration_hours
            FROM scientific_flights
            """
        ),
        engine,
    )
    flights_df = flights_df[
        flights_df["origin"].isin(origin_airports)
        & flights_df["duration_hours"].ge(minimum_duration_hours)
    ].copy()

    candidate_days: set[date] = set()
    for row in flights_df.itertuples(index=False):
        first_seen = pd.Timestamp(row.firstseen).date()
        last_seen = pd.Timestamp(row.lastseen).date()
        current_day = first_seen
        while current_day <= last_seen:
            candidate_days.add(current_day)
            current_day += timedelta(days=1)
    return candidate_days


def load_flights_for_day(engine: Engine, day: date) -> list[dict[str, Any]]:
    day_start = pd.Timestamp(day, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    query = text(
        """
        SELECT flight_key, icao24, callsign, origin, destination,
               firstseen, lastseen, duration_hours
        FROM scientific_flights
        WHERE firstseen < :day_end
          AND lastseen >= :day_start
        ORDER BY firstseen
        """
    )
    flights_df = pd.read_sql_query(
        query,
        engine,
        params={"day_start": day_start, "day_end": day_end},
    )
    flights = flights_df.to_dict(orient="records")
    for flight in flights:
        flight["callsign_normalized"] = normalize_callsign(flight["callsign"])
    return flights


def build_flight_lookup(flights: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for flight in flights:
        lookup.setdefault(str(flight["icao24"]).lower(), []).append(flight)
    return lookup


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "t", "yes"}:
        return True
    if normalized in {"false", "0", "f", "no"}:
        return False
    return None


def parse_epoch_timestamp(value: str | None) -> pd.Timestamp | None:
    numeric_value = parse_optional_float(value)
    if numeric_value is None:
        return None
    return pd.Timestamp(numeric_value, unit="s", tz="UTC")


def match_state_row_to_flight(
    row: dict[str, str],
    flights_by_icao24: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    icao24 = (row.get("icao24") or "").strip().lower()
    if not icao24:
        return None

    state_time = parse_epoch_timestamp(row.get("time"))
    if state_time is None:
        return None

    candidates = [
        flight
        for flight in flights_by_icao24.get(icao24, [])
        if pd.Timestamp(flight["firstseen"]) <= state_time <= pd.Timestamp(flight["lastseen"])
    ]
    if not candidates:
        return None

    row_callsign = normalize_callsign(row.get("callsign"))
    if row_callsign is not None:
        exact = [
            flight
            for flight in candidates
            if flight["callsign_normalized"] == row_callsign
        ]
        if exact:
            candidates = exact

    return min(
        candidates,
        key=lambda flight: abs(
            (state_time - pd.Timestamp(flight["firstseen"])).total_seconds()
        ),
    )


def state_row_to_record(
    row: dict[str, str],
    matched_flight: dict[str, Any],
    *,
    source_file: str,
    source_member: str,
) -> dict[str, Any]:
    time_ts = parse_epoch_timestamp(row.get("time"))
    assert time_ts is not None
    return {
        "flight_key": matched_flight["flight_key"],
        "time": time_ts,
        "icao24": (row.get("icao24") or "").strip().lower(),
        "callsign": normalize_callsign(row.get("callsign")),
        "lat": parse_optional_float(row.get("lat")),
        "lon": parse_optional_float(row.get("lon")),
        "velocity": parse_optional_float(row.get("velocity")),
        "heading": parse_optional_float(row.get("heading")),
        "vertrate": parse_optional_float(row.get("vertrate")),
        "onground": parse_optional_bool(row.get("onground")),
        "alert": parse_optional_bool(row.get("alert")),
        "spi": parse_optional_bool(row.get("spi")),
        "squawk": (row.get("squawk") or "").strip() or None,
        "baroaltitude": parse_optional_float(row.get("baroaltitude")),
        "geoaltitude": parse_optional_float(row.get("geoaltitude")),
        "lastposupdate": parse_epoch_timestamp(row.get("lastposupdate")),
        "lastcontact": parse_epoch_timestamp(row.get("lastcontact")),
        "serials": None,
        "hour": time_ts.floor("h"),
        "source_dataset": "weekly_state_vectors",
        "source_file": source_file,
        "source_member": source_member,
    }


def iter_csv_members_from_tar(archive_path: Path) -> Iterator[tuple[str, csv.DictReader]]:
    mode = "r:*"
    with tarfile.open(archive_path, mode) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            text_stream = io.TextIOWrapper(extracted, encoding="utf-8")
            reader = csv.DictReader(text_stream)
            yield member.name, reader


def ingest_matching_state_vectors(
    engine: Engine,
    state_archives: Sequence[Path],
    candidate_days: set[date],
) -> None:
    for archive_path in progress(
        state_archives,
        desc="Ingest state vectors",
        total=len(state_archives),
        unit="archive",
    ):
        archive_day = extract_date_from_name(archive_path.name)
        if archive_day is not None and candidate_days and archive_day not in candidate_days:
            print(f"Skipping archive outside candidate day set: {archive_path.name}")
            continue

        print(f"Processing state archive: {archive_path}")
        flights = load_flights_for_day(engine, archive_day) if archive_day is not None else []
        if not flights:
            print(f"No eligible flights found for archive day: {archive_day}")
            continue

        flights_by_icao24 = build_flight_lookup(flights)
        buffered_records: list[dict[str, Any]] = []

        for member_name, reader in iter_csv_members_from_tar(archive_path):
            print(f"Reading member: {member_name}")
            row_counter = 0
            match_counter = 0
            for row in reader:
                row_counter += 1
                matched_flight = match_state_row_to_flight(row, flights_by_icao24)
                if matched_flight is None:
                    continue
                buffered_records.append(
                    state_row_to_record(
                        row,
                        matched_flight,
                        source_file=archive_path.name,
                        source_member=member_name,
                    )
                )
                match_counter += 1
                if len(buffered_records) >= STATE_VECTOR_INSERT_BATCH_SIZE:
                    insert_records(
                        engine,
                        "scientific_state_vectors",
                        buffered_records,
                        conflict_columns=["flight_key", "time", "icao24", "callsign"],
                    )
                    buffered_records.clear()
                if row_counter % 250_000 == 0:
                    print(
                        f"Processed {row_counter} rows from {member_name}; "
                        f"matched {match_counter} rows so far."
                    )

            print(
                f"Finished member {member_name}: processed {row_counter} rows, "
                f"matched {match_counter} rows."
            )

        if buffered_records:
            insert_records(
                engine,
                "scientific_state_vectors",
                buffered_records,
                conflict_columns=["flight_key", "time", "icao24", "callsign"],
            )


def build_default_trajectory_query(
    *,
    origin_airport: str = DEFAULT_ORIGIN_AIRPORTS[0],
    minimum_duration_hours: float = DEFAULT_MINIMUM_DURATION_HOURS,
    start_time: str | None = None,
    end_time: str | None = None,
    sample_trajectories: int | None = None,
) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {
        "origin_airport": origin_airport,
        "minimum_duration_hours": minimum_duration_hours,
        "start_time": start_time,
        "end_time": end_time,
    }

    sampled_cte = ""
    source_name = "filtered_flights"
    if sample_trajectories is not None:
        sampled_cte = """
        , sampled_flights AS (
            SELECT *
            FROM filtered_flights
            ORDER BY RANDOM()
            LIMIT :sample_trajectories
        )
        """
        params["sample_trajectories"] = sample_trajectories
        source_name = "sampled_flights"

    sql = f"""
    WITH filtered_flights AS (
        SELECT *
        FROM scientific_flights
        WHERE origin = :origin_airport
          AND duration_hours >= :minimum_duration_hours
          AND (:start_time IS NULL OR firstseen >= :start_time)
          AND (:end_time IS NULL OR lastseen <= :end_time)
    )
    {sampled_cte}
    SELECT
        sv.time,
        sv.icao24,
        sv.lat,
        sv.lon,
        sv.velocity,
        sv.heading,
        sv.vertrate,
        sv.callsign,
        sv.onground,
        sv.alert,
        sv.spi,
        sv.squawk,
        sv.baroaltitude,
        sv.geoaltitude,
        sv.lastposupdate,
        sv.lastcontact,
        sv.serials,
        sv.hour,
        sv.flight_key,
        f.origin,
        f.destination,
        f.firstseen AS first_seen,
        f.lastseen AS last_seen,
        f.duration_hours,
        'scientific_db' AS source_backend
    FROM scientific_state_vectors sv
    JOIN {source_name} f
      ON sv.flight_key = f.flight_key
    ORDER BY f.firstseen, sv.time
    """
    return sql.strip(), params


def query_scientific_db(
    database_url: str,
    *,
    sql_query: str | None = None,
    sql_params: dict[str, Any] | None = None,
    origin_airport: str = DEFAULT_ORIGIN_AIRPORTS[0],
    minimum_duration_hours: float = DEFAULT_MINIMUM_DURATION_HOURS,
    start_time: str | None = None,
    end_time: str | None = None,
    sample_trajectories: int | None = None,
) -> pd.DataFrame:
    if sql_query is None:
        sql_query, sql_params = build_default_trajectory_query(
            origin_airport=origin_airport,
            minimum_duration_hours=minimum_duration_hours,
            start_time=start_time,
            end_time=end_time,
            sample_trajectories=sample_trajectories,
        )
    elif sql_params is None:
        sql_params = {}

    engine = get_engine(database_url)
    return pd.read_sql_query(text(sql_query), engine, params=sql_params)


def build_scientific_db(config: BuildConfig) -> None:
    if config.state_archive_format != "csv":
        raise NotImplementedError(
            "This first implementation supports CSV state-vector archives only."
        )

    engine = get_engine(config.database_url)
    create_schema(engine)

    covid_files = fetch_covid_file_metadata()
    if not covid_files:
        raise RuntimeError("Could not discover COVID dataset files.")
    covid_paths = ensure_downloads(
        covid_files,
        config.download_dir,
        subdir="covid",
        limit=config.max_covid_files,
    )
    ingest_filtered_covid_flights(
        engine,
        covid_paths,
        origin_airports=config.origin_airports,
        minimum_duration_hours=config.minimum_duration_hours,
    )

    candidate_days = load_candidate_days_from_db(
        engine,
        origin_airports=config.origin_airports,
        minimum_duration_hours=config.minimum_duration_hours,
    )
    print(f"Identified {len(candidate_days)} candidate UTC days with eligible flights.")

    state_archives = fetch_state_archive_metadata(config.state_archive_format)
    if not state_archives:
        raise RuntimeError("Could not discover weekly state-vector archives.")

    filtered_archives = [
        archive
        for archive in state_archives
        if (
            extract_date_from_name(archive["name"]) in candidate_days
            if extract_date_from_name(archive["name"]) is not None
            else True
        )
    ]
    state_paths = ensure_downloads(
        filtered_archives,
        config.download_dir,
        subdir=f"states_{config.state_archive_format}",
        limit=config.max_state_archives,
    )
    ingest_matching_state_vectors(engine, state_paths, candidate_days)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and query a filtered PostgreSQL database from "
            "OpenSky scientific datasets."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Download filtered source files and build the PostgreSQL database.",
    )
    build_parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help=(
            "SQLAlchemy PostgreSQL URL. Default: "
            f"{DEFAULT_DATABASE_URL}"
        ),
    )
    build_parser.add_argument(
        "--download-dir",
        required=True,
        help="Directory for downloaded source files, e.g. an external drive path.",
    )
    build_parser.add_argument(
        "--origin-airports",
        default=",".join(DEFAULT_ORIGIN_AIRPORTS),
        help="Comma-separated origin airports to retain. Default: EGLL,WSSS",
    )
    build_parser.add_argument(
        "--minimum-duration-hours",
        type=float,
        default=DEFAULT_MINIMUM_DURATION_HOURS,
    )
    build_parser.add_argument(
        "--state-archive-format",
        choices=["csv", "json"],
        default=DEFAULT_STATE_ARCHIVE_FORMAT,
        help="Weekly state-vector archive format to download.",
    )
    build_parser.add_argument(
        "--max-covid-files",
        type=int,
        default=None,
        help="Optional limit for initial build/testing.",
    )
    build_parser.add_argument(
        "--max-state-archives",
        type=int,
        default=None,
        help="Optional limit for initial build/testing.",
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Run the default or a custom SQL query against the scientific DB.",
    )
    query_parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help=(
            "SQLAlchemy PostgreSQL URL. Default: "
            f"{DEFAULT_DATABASE_URL}"
        ),
    )
    query_parser.add_argument(
        "--origin-airport",
        default=DEFAULT_ORIGIN_AIRPORTS[0],
    )
    query_parser.add_argument(
        "--minimum-duration-hours",
        type=float,
        default=DEFAULT_MINIMUM_DURATION_HOURS,
    )
    query_parser.add_argument("--start-time", default=None)
    query_parser.add_argument("--end-time", default=None)
    query_parser.add_argument("--sample-trajectories", type=int, default=None)
    query_parser.add_argument(
        "--sql",
        default=None,
        help="Optional custom SQL query. If omitted, use the default flight join.",
    )
    query_parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save query results as CSV.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        config = BuildConfig(
            database_url=args.database_url,
            download_dir=Path(args.download_dir),
            origin_airports=tuple(
                airport.strip().upper()
                for airport in args.origin_airports.split(",")
                if airport.strip()
            ),
            minimum_duration_hours=args.minimum_duration_hours,
            state_archive_format=args.state_archive_format,
            max_covid_files=args.max_covid_files,
            max_state_archives=args.max_state_archives,
        )
        build_scientific_db(config)
        return

    query_df = query_scientific_db(
        args.database_url,
        sql_query=args.sql,
        origin_airport=args.origin_airport,
        minimum_duration_hours=args.minimum_duration_hours,
        start_time=args.start_time,
        end_time=args.end_time,
        sample_trajectories=args.sample_trajectories,
    )
    print(query_df.head())
    print(f"\nReturned {len(query_df)} rows.")
    if args.output_csv:
        output_path = Path(args.output_csv)
        query_df.to_csv(output_path, index=False)
        print(f"Saved query output to {output_path}")


if __name__ == "__main__":
    main()
