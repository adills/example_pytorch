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
- OMDB (Dubai)
- OTHH (Doha)
- EDDF (Frankfurt)

This provides a mix of large European and Gulf long-haul hubs with a strong
chance of producing many international long-duration flights in the historical
datasets, while still keeping the stored data smaller than a full global
ingest. These origins are fully configurable at build time.

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

Inspect a small unfiltered sample of COVID manifests before building:

    python opensky_build_scientific_db.py inspect-covid \
        --download-dir /Volumes/external/opensky \
        --max-covid-files 2 \
        --max-covid-chunks-per-file 1

The ``build`` command processes source files one at a time through a temporary
workspace, instead of downloading the full scientific datasets up front.

Requirements
------------
- Python packages:
  - pandas
  - httpx
  - sqlalchemy
- A PostgreSQL driver compatible with SQLAlchemy, e.g. ``psycopg``
- A writable temporary workspace directory, such as an external drive

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
``/opt/homebrew/var/postgresql@16``. The large scientific source files are
processed one at a time through the workspace directory you provide with
``--download-dir``.

Testing
-------
Unit tests for the filtering and query-builder logic live in:

    tests/test_opensky_build_scientific_db.py

Run them with:

    PYTHONPATH=. pipenv run pytest tests -v

    or directly with unittest:
    pipenv run python -m unittest discover -s tests -v

Notes
-----
- The COVID dataset provides flight-level grouping metadata such as origin,
  destination, and first/last seen timestamps.
- The weekly state-vector dataset does not provide origin/destination directly,
  so state rows are matched to filtered COVID flights by ``icao24``,
  normalized ``callsign``, and time-window overlap.
- The builder uses remote metadata endpoints and public bucket listing where
  possible, falling back to HTML scraping only if needed.
- To reduce intermediate storage, the build path downloads at most one source
  file at a time into the temporary workspace, ingests the filtered rows, and
  then deletes the temporary file.
- Build progress is resumable through the ``scientific_ingest_manifest`` table.
  This is usually helpful, but if you change the default origin airports,
  duration threshold, schema assumptions, or other ingest logic and want a
  completely fresh rebuild, reset the scientific DB tables first:

      python opensky_build_scientific_db.py reset-build --confirm-reset

  That clears ``scientific_flights``, ``scientific_state_vectors``, and
  ``scientific_ingest_manifest`` so the next ``build`` run starts from scratch.
- If a build inserts zero candidate flights, use ``inspect-covid`` first. It
  performs a dry-run examination of the COVID source files without inserting
  anything into PostgreSQL and reports the raw origin/destination frequencies
  and duration distribution so you can choose better filters.

  Use it like this for a first pass:

      python opensky_build_scientific_db.py inspect-covid \
      --download-dir /path/to/workspace \
      --max-covid-files 2 \
      --max-covid-chunks-per-file 1
  
  If you want a broader sample:

      python opensky_build_scientific_db.py inspect-covid \
      --download-dir /path/to/workspace \
      --max-covid-files 4 \
      --max-covid-chunks-per-file 2 \
      --top-n-airports 30

Querying the database
---------------------
After building the database, you can run the ``query`` command to execute a
predefined join query that retrieves a sample of flights and their trajectories
from the local database. The query parameters are configurable, and you can
also modify the query logic in the code if you want to explore different slices of
the data.

To see the size of the database, run:

    SELECT pg_size_pretty(pg_database_size('opensky_scientific')) AS db_size,
    pg_database_size('opensky_scientific') AS db_bytes;

To see how many long-haul flights are in scientific_flights regardless of matched state vectors, run:

    SELECT origin, COUNT(*) AS filtered_flights
    FROM scientific_flights
    GROUP BY origin
    ORDER BY origin;

To see what days actually exist in your currently ingested trajectory table, run:

    SELECT
        MIN(time) AS min_state_time,
        MAX(time) AS max_state_time,
        COUNT(*) AS state_rows
    FROM scientific_state_vectors;

To see the available flight wndow for EGLL directly:

    SELECT
        origin,
        MIN(firstseen) AS min_firstseen,
        MAX(lastseen) AS max_lastseen,
        COUNT(*) AS flights
    FROM scientific_flights
    WHERE origin = 'EGLL'
    GROUP BY origin;

To confirm which hourly archives were ingested, run:

    SELECT source_file, rows_inserted
    FROM scientific_ingest_manifest
    WHERE source_dataset = 'weekly_state_vectors'
    ORDER BY source_file;

To see that the max and min start flight times are for each origin that satisfy
the join conditions, run:

    SELECT
        f.origin,
        MIN(f.firstseen) AS min_start_time,
        MAX(f.firstseen) AS max_start_time,
        MAX(f.lastseen) AS max_end_time,
        COUNT(*) AS matched_flights
    FROM scientific_flights f
    JOIN (
        SELECT DISTINCT flight_key
        FROM scientific_state_vectors
    ) sv
        ON sv.flight_key = f.flight_key
    GROUP BY f.origin
    ORDER BY f.origin;

To see how many state vectors are matched to each origin, run:

    SELECT
        f.origin,
        COUNT(*) AS state_vector_rows,
        COUNT(DISTINCT f.flight_key) AS matched_flights
    FROM scientific_state_vectors sv
    JOIN scientific_flights f
        ON sv.flight_key = f.flight_key
    GROUP BY f.origin
    ORDER BY f.origin;

To see the average trajectory length in hours for matched flights from each 
origin, run:

    SELECT
        f.origin,
        COUNT(*) AS state_vector_rows,
        COUNT(DISTINCT f.flight_key) AS matched_flights,
        ROUND(COUNT(*)::numeric / COUNT(DISTINCT f.flight_key), 2) AS avg_rows_per_flight
    FROM scientific_state_vectors sv
    JOIN scientific_flights f
        ON sv.flight_key = f.flight_key
    GROUP BY f.origin
    ORDER BY f.origin;
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import getpass
import gzip
import io
from pathlib import Path
import re
import tarfile
import tempfile
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
STATES_BUCKET_LIST_URL = "https://s3.opensky-network.org/data-samples/"
STATES_BUCKET_PREFIX = "states/"
DEFAULT_ORIGIN_AIRPORTS = ("EGLL", "OMDB", "OTHH", "EDDF")
DEFAULT_MINIMUM_DURATION_HOURS = 6.0
DEFAULT_STATE_ARCHIVE_FORMAT = "csv"
DEFAULT_HTTP_TIMEOUT_SECONDS = 60.0
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_RETRY_ATTEMPTS = 4
DOWNLOAD_RETRY_SLEEP_SECONDS = 5
RETRIABLE_DOWNLOAD_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
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


@dataclass(frozen=True)
class InspectCovidConfig:
    download_dir: Path
    max_covid_files: int = 2
    max_covid_chunks_per_file: int = 1
    top_n_airports: int = 20


MANIFEST_STATUS_COMPLETED = "completed"
MANIFEST_STATUS_FAILED = "failed"
MANIFEST_STATUS_IN_PROGRESS = "in_progress"
REQUIRED_COVID_COLUMNS = (
    "callsign",
    "number",
    "icao24",
    "registration",
    "typecode",
    "origin",
    "destination",
    "firstseen",
    "lastseen",
)
REQUIRED_STATE_COLUMNS = (
    "time",
    "icao24",
    "callsign",
    "lat",
    "lon",
    "velocity",
    "heading",
    "vertrate",
    "onground",
    "alert",
    "spi",
    "squawk",
    "baroaltitude",
    "geoaltitude",
    "lastposupdate",
    "lastcontact",
)


def normalize_callsign(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().upper()
    return normalized or None


def normalize_airport_code(value: Any) -> str | None:
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
        CREATE TABLE IF NOT EXISTS scientific_ingest_manifest (
            source_dataset TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source_url TEXT,
            status TEXT NOT NULL,
            rows_inserted BIGINT NOT NULL DEFAULT 0,
            started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMP WITH TIME ZONE,
            error_message TEXT,
            PRIMARY KEY (source_dataset, source_file)
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
        """
        CREATE INDEX IF NOT EXISTS idx_ingest_manifest_status
        ON scientific_ingest_manifest (status, source_dataset)
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
        if "/." in key:
            continue
        archive_name = Path(key).name
        if archive_name.startswith("states_."):
            continue
        files.append(
            {
                "name": archive_name,
                "url": urljoin(STATES_BUCKET_LIST_URL, key),
            }
        )
    return sorted(files, key=lambda item: item["name"])


def ensure_workspace(download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


def is_retriable_download_exception(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RequestError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRIABLE_DOWNLOAD_STATUS_CODES
    return False


def download_to_temporary_file(url: str, workspace_dir: Path, filename: str) -> Path:
    ensure_workspace(workspace_dir)
    suffix = "".join(Path(filename).suffixes) or ".tmp"
    for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                prefix="opensky_",
                dir=workspace_dir,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                print(f"Streaming {url} -> temporary file {temp_path}")
                with httpx.Client(
                    timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
                    follow_redirects=True,
                ) as client:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                            if chunk:
                                handle.write(chunk)
            return temp_path
        except Exception as exc:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

            if not is_retriable_download_exception(exc) or attempt == DOWNLOAD_RETRY_ATTEMPTS:
                raise

            sleep_seconds = DOWNLOAD_RETRY_SLEEP_SECONDS * attempt
            if isinstance(exc, httpx.HTTPStatusError):
                error_detail = f"HTTP {exc.response.status_code}"
            else:
                error_detail = exc.__class__.__name__
            print(
                f"Download failed for {filename} with {error_detail}. "
                f"Sleeping {sleep_seconds}s before retry "
                f"{attempt + 1}/{DOWNLOAD_RETRY_ATTEMPTS}."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Download unexpectedly exhausted retries for {filename}")


def get_manifest_status(
    engine: Engine,
    source_dataset: str,
    source_file: str,
) -> str | None:
    query = text(
        """
        SELECT status
        FROM scientific_ingest_manifest
        WHERE source_dataset = :source_dataset
          AND source_file = :source_file
        """
    )
    with engine.begin() as connection:
        result = connection.execute(
            query,
            {
                "source_dataset": source_dataset,
                "source_file": source_file,
            },
        ).scalar_one_or_none()
    return str(result) if result is not None else None


def mark_manifest_started(
    engine: Engine,
    *,
    source_dataset: str,
    source_file: str,
    source_url: str,
) -> None:
    statement = text(
        """
        INSERT INTO scientific_ingest_manifest (
            source_dataset,
            source_file,
            source_url,
            status,
            rows_inserted,
            started_at,
            completed_at,
            error_message
        )
        VALUES (
            :source_dataset,
            :source_file,
            :source_url,
            :status,
            0,
            NOW(),
            NULL,
            NULL
        )
        ON CONFLICT (source_dataset, source_file)
        DO UPDATE SET
            source_url = EXCLUDED.source_url,
            status = EXCLUDED.status,
            rows_inserted = 0,
            started_at = NOW(),
            completed_at = NULL,
            error_message = NULL
        """
    )
    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                "source_dataset": source_dataset,
                "source_file": source_file,
                "source_url": source_url,
                "status": MANIFEST_STATUS_IN_PROGRESS,
            },
        )


def mark_manifest_completed(
    engine: Engine,
    *,
    source_dataset: str,
    source_file: str,
    source_url: str,
    rows_inserted: int,
    error_message: str | None = None,
) -> None:
    statement = text(
        """
        INSERT INTO scientific_ingest_manifest (
            source_dataset,
            source_file,
            source_url,
            status,
            rows_inserted,
            started_at,
            completed_at,
            error_message
        )
        VALUES (
            :source_dataset,
            :source_file,
            :source_url,
            :status,
            :rows_inserted,
            NOW(),
            NOW(),
            :error_message
        )
        ON CONFLICT (source_dataset, source_file)
        DO UPDATE SET
            source_url = EXCLUDED.source_url,
            status = EXCLUDED.status,
            rows_inserted = EXCLUDED.rows_inserted,
            completed_at = NOW(),
            error_message = EXCLUDED.error_message
        """
    )
    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                "source_dataset": source_dataset,
                "source_file": source_file,
                "source_url": source_url,
                "status": MANIFEST_STATUS_COMPLETED,
                "rows_inserted": rows_inserted,
                "error_message": error_message,
            },
        )


def mark_manifest_failed(
    engine: Engine,
    *,
    source_dataset: str,
    source_file: str,
    source_url: str,
    error_message: str,
) -> None:
    statement = text(
        """
        INSERT INTO scientific_ingest_manifest (
            source_dataset,
            source_file,
            source_url,
            status,
            rows_inserted,
            started_at,
            completed_at,
            error_message
        )
        VALUES (
            :source_dataset,
            :source_file,
            :source_url,
            :status,
            0,
            NOW(),
            NOW(),
            :error_message
        )
        ON CONFLICT (source_dataset, source_file)
        DO UPDATE SET
            source_url = EXCLUDED.source_url,
            status = EXCLUDED.status,
            completed_at = NOW(),
            error_message = EXCLUDED.error_message
        """
    )
    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                "source_dataset": source_dataset,
                "source_file": source_file,
                "source_url": source_url,
                "status": MANIFEST_STATUS_FAILED,
                "error_message": error_message,
            },
        )


def parse_timestamp_column(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    numeric_mask = numeric_series.notna()
    if numeric_mask.any():
        numeric_values = numeric_series[numeric_mask]
        millisecond_mask = numeric_values.abs().ge(1e11)
        second_values = numeric_values[~millisecond_mask]
        millisecond_values = numeric_values[millisecond_mask]
        if not second_values.empty:
            parsed.loc[second_values.index] = pd.to_datetime(
                second_values,
                unit="s",
                utc=True,
                errors="coerce",
            )
        if not millisecond_values.empty:
            parsed.loc[millisecond_values.index] = pd.to_datetime(
                millisecond_values,
                unit="ms",
                utc=True,
                errors="coerce",
            )

    non_numeric_mask = ~numeric_mask
    if non_numeric_mask.any():
        parsed.loc[non_numeric_mask] = pd.to_datetime(
            series[non_numeric_mask],
            utc=True,
            errors="coerce",
        )

    return parsed


def validate_required_columns(
    actual_columns: Sequence[str],
    required_columns: Sequence[str],
    *,
    source_label: str,
) -> None:
    actual_set = set(actual_columns)
    missing = [column for column in required_columns if column not in actual_set]
    if not missing:
        return
    raise RuntimeError(
        f"{source_label} is missing required columns: {missing}. "
        f"Available columns: {list(actual_columns)}"
    )


def summarize_covid_filter_counts(
    chunk: pd.DataFrame,
    *,
    origin_airports: Sequence[str],
    minimum_duration_hours: float,
) -> dict[str, int]:
    summary_df = chunk.copy()
    summary_df["origin"] = summary_df["origin"].astype("string")
    summary_df["firstseen"] = parse_timestamp_column(summary_df["firstseen"])
    summary_df["lastseen"] = parse_timestamp_column(summary_df["lastseen"])
    summary_df["duration_hours"] = (
        summary_df["lastseen"] - summary_df["firstseen"]
    ).dt.total_seconds() / 3600.0

    total_rows = len(summary_df)
    origin_match = int(summary_df["origin"].isin(origin_airports).sum())
    non_null_time_and_origin = int(
        (
            summary_df["origin"].notna()
            & summary_df["firstseen"].notna()
            & summary_df["lastseen"].notna()
        ).sum()
    )
    duration_match = int(summary_df["duration_hours"].ge(minimum_duration_hours).sum())
    final_match = int(
        (
            summary_df["origin"].isin(origin_airports)
            & summary_df["duration_hours"].ge(minimum_duration_hours)
            & summary_df["origin"].notna()
            & summary_df["firstseen"].notna()
            & summary_df["lastseen"].notna()
        ).sum()
    )
    return {
        "total_rows": total_rows,
        "origin_match_rows": origin_match,
        "non_null_time_and_origin_rows": non_null_time_and_origin,
        "duration_match_rows": duration_match,
        "final_match_rows": final_match,
    }


def summarize_unfiltered_covid_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    summary_df = chunk.copy()
    summary_df["origin"] = summary_df["origin"].map(normalize_airport_code).astype("string")
    summary_df["destination"] = summary_df["destination"].map(normalize_airport_code).astype("string")
    summary_df["callsign"] = (
        summary_df["callsign"].astype("string").str.strip().str.upper()
    )
    summary_df["icao24"] = (
        summary_df["icao24"].astype("string").str.strip().str.lower()
    )
    summary_df["firstseen"] = parse_timestamp_column(summary_df["firstseen"])
    summary_df["lastseen"] = parse_timestamp_column(summary_df["lastseen"])
    summary_df["duration_hours"] = (
        summary_df["lastseen"] - summary_df["firstseen"]
    ).dt.total_seconds() / 3600.0
    return summary_df


def format_duration_bucket_summary(summary_df: pd.DataFrame) -> dict[str, int]:
    duration_series = summary_df["duration_hours"]
    return {
        "ge_4h": int(duration_series.ge(4).sum()),
        "ge_6h": int(duration_series.ge(6).sum()),
        "ge_8h": int(duration_series.ge(8).sum()),
        "ge_10h": int(duration_series.ge(10).sum()),
        "ge_12h": int(duration_series.ge(12).sum()),
    }


def build_origin_duration_summary(
    summary_df: pd.DataFrame,
    *,
    top_n_airports: int,
) -> pd.DataFrame:
    valid = summary_df[summary_df["origin"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=["origin", "rows", "ge_4h", "ge_6h", "ge_8h", "ge_10h", "ge_12h"]
        )

    grouped = valid.groupby("origin", dropna=True)
    result = grouped["origin"].size().rename("rows").to_frame()
    for threshold in (4, 6, 8, 10, 12):
        result[f"ge_{threshold}h"] = grouped["duration_hours"].apply(
            lambda series, t=threshold: int(series.ge(t).sum())
        )
    result = result.sort_values(["ge_6h", "rows"], ascending=False).head(top_n_airports)
    return result.reset_index()


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


def process_covid_file_streaming(
    engine: Engine,
    *,
    file_metadata: dict[str, str],
    workspace_dir: Path,
    origin_airports: Sequence[str],
    minimum_duration_hours: float,
    validate_schema: bool = False,
) -> int:
    source_file = file_metadata["name"]
    source_url = file_metadata["url"]
    manifest_status = get_manifest_status(engine, "covid_flight_dataset", source_file)
    if manifest_status == MANIFEST_STATUS_COMPLETED:
        print(f"Skipping already ingested flight manifest: {source_file}")
        return 0

    mark_manifest_started(
        engine,
        source_dataset="covid_flight_dataset",
        source_file=source_file,
        source_url=source_url,
    )
    temp_path: Path | None = None
    inserted_rows = 0
    try:
        temp_path = download_to_temporary_file(source_url, workspace_dir, source_file)
        print(f"Processing flight manifest: {source_file}")
        first_chunk = True
        for chunk in pd.read_csv(temp_path, compression="gzip", chunksize=COVID_CHUNK_ROWS):
            if validate_schema and first_chunk:
                validate_required_columns(
                    list(chunk.columns),
                    REQUIRED_COVID_COLUMNS,
                    source_label=f"COVID manifest {source_file}",
                )
                print(
                    "COVID first-file schema check passed. Columns: "
                    f"{list(chunk.columns)}"
                )
                counts = summarize_covid_filter_counts(
                    chunk,
                    origin_airports=origin_airports,
                    minimum_duration_hours=minimum_duration_hours,
                )
                print(
                    "COVID first-file filter summary: "
                    f"{counts}"
                )
                first_chunk = False
            filtered = filter_covid_chunk(
                chunk,
                origin_airports=origin_airports,
                minimum_duration_hours=minimum_duration_hours,
                source_file=source_file,
            )
            if filtered.empty:
                continue
            records = filtered.to_dict(orient="records")
            inserted_rows += insert_records(
                engine,
                "scientific_flights",
                records,
                conflict_columns=["flight_key"],
            )
        mark_manifest_completed(
            engine,
            source_dataset="covid_flight_dataset",
            source_file=source_file,
            source_url=source_url,
            rows_inserted=inserted_rows,
        )
        return inserted_rows
    except Exception as exc:
        mark_manifest_failed(
            engine,
            source_dataset="covid_flight_dataset",
            source_file=source_file,
            source_url=source_url,
            error_message=str(exc),
        )
        raise
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def ingest_filtered_covid_flights_streaming(
    engine: Engine,
    covid_files: Sequence[dict[str, str]],
    *,
    workspace_dir: Path,
    origin_airports: Sequence[str],
    minimum_duration_hours: float,
) -> int:
    total_inserted = 0
    for index, file_metadata in enumerate(progress(
        covid_files,
        desc="Ingest COVID flights",
        total=len(covid_files),
        unit="file",
    )):
        total_inserted += process_covid_file_streaming(
            engine,
            file_metadata=file_metadata,
            workspace_dir=workspace_dir,
            origin_airports=origin_airports,
            minimum_duration_hours=minimum_duration_hours,
            validate_schema=index == 0,
        )
    return total_inserted


def inspect_covid_data(config: InspectCovidConfig) -> None:
    workspace_dir = ensure_workspace(config.download_dir)
    covid_files = fetch_covid_file_metadata()
    if not covid_files:
        raise RuntimeError("Could not discover COVID dataset files.")

    selected_files = list(covid_files[: config.max_covid_files])
    if not selected_files:
        raise RuntimeError("No COVID files were selected for inspection.")

    aggregated_frames: list[pd.DataFrame] = []
    inspected_file_names: list[str] = []
    first_columns: list[str] | None = None
    first_firstseen_samples: list[Any] | None = None
    first_lastseen_samples: list[Any] | None = None

    for file_metadata in progress(
        selected_files,
        desc="Inspect COVID manifests",
        total=len(selected_files),
        unit="file",
    ):
        source_file = file_metadata["name"]
        source_url = file_metadata["url"]
        temp_path: Path | None = None
        try:
            temp_path = download_to_temporary_file(source_url, workspace_dir, source_file)
            print(f"Inspecting flight manifest: {source_file}")
            chunk_count = 0
            for chunk in pd.read_csv(temp_path, compression="gzip", chunksize=COVID_CHUNK_ROWS):
                validate_required_columns(
                    list(chunk.columns),
                    REQUIRED_COVID_COLUMNS,
                    source_label=f"COVID manifest {source_file}",
                )
                if first_columns is None:
                    first_columns = list(chunk.columns)
                    first_firstseen_samples = chunk["firstseen"].head(5).tolist()
                    first_lastseen_samples = chunk["lastseen"].head(5).tolist()
                aggregated_frames.append(summarize_unfiltered_covid_chunk(chunk))
                chunk_count += 1
                if chunk_count >= config.max_covid_chunks_per_file:
                    break
            inspected_file_names.append(source_file)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    if not aggregated_frames:
        raise RuntimeError("No COVID rows were inspected.")

    combined = pd.concat(aggregated_frames, ignore_index=True)
    print("\n--- COVID Inspection Summary ---")
    print(f"Files inspected: {len(inspected_file_names)}")
    print(f"File names: {inspected_file_names}")
    print(f"Rows inspected: {len(combined)}")
    if first_columns is not None:
        print(f"Columns: {first_columns}")
    if first_firstseen_samples is not None and first_lastseen_samples is not None:
        print(f"Sample raw firstseen values: {first_firstseen_samples}")
        print(f"Sample raw lastseen values: {first_lastseen_samples}")
    print(
        "Parsed timestamp null counts: "
        f"firstseen={int(combined['firstseen'].isna().sum())}, "
        f"lastseen={int(combined['lastseen'].isna().sum())}"
    )

    print("\nTop origin airports by rows:")
    origin_counts = (
        combined["origin"]
        .value_counts(dropna=True)
        .head(config.top_n_airports)
        .rename_axis("origin")
        .reset_index(name="rows")
    )
    print(origin_counts.to_string(index=False))

    print("\nTop destination airports by rows:")
    destination_counts = (
        combined["destination"]
        .value_counts(dropna=True)
        .head(config.top_n_airports)
        .rename_axis("destination")
        .reset_index(name="rows")
    )
    print(destination_counts.to_string(index=False))

    valid_durations = combined["duration_hours"].dropna()
    if not valid_durations.empty:
        print("\nDuration quantiles (hours):")
        duration_quantiles = valid_durations.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).round(2)
        print(duration_quantiles.to_string())
        print("\nDuration bucket counts:")
        print(format_duration_bucket_summary(combined))
    else:
        print("\nNo valid duration values were parsed from the inspected rows.")

    print("\nTop origins by long-haul counts:")
    origin_duration_summary = build_origin_duration_summary(
        combined,
        top_n_airports=config.top_n_airports,
    )
    print(origin_duration_summary.to_string(index=False))


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


def load_candidate_archive_hours_from_db(
    engine: Engine,
    origin_airports: Sequence[str],
    minimum_duration_hours: float,
) -> set[pd.Timestamp]:
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

    candidate_hours: set[pd.Timestamp] = set()
    for row in flights_df.itertuples(index=False):
        current_hour = pd.Timestamp(row.firstseen).floor("h")
        last_hour = pd.Timestamp(row.lastseen).floor("h")
        while current_hour <= last_hour:
            candidate_hours.add(current_hour)
            current_hour += pd.Timedelta(hours=1)
    return candidate_hours


def summarize_filtered_flights_from_db(
    engine: Engine,
    origin_airports: Sequence[str],
    minimum_duration_hours: float,
) -> pd.DataFrame:
    flights_df = pd.read_sql_query(
        text(
            """
            SELECT origin, duration_hours
            FROM scientific_flights
            """
        ),
        engine,
    )
    flights_df = flights_df[
        flights_df["origin"].isin(origin_airports)
        & flights_df["duration_hours"].ge(minimum_duration_hours)
    ].copy()
    if flights_df.empty:
        return pd.DataFrame(columns=["origin", "rows", "mean_duration_hours", "max_duration_hours"])

    summary = (
        flights_df.groupby("origin", dropna=True)["duration_hours"]
        .agg(rows="size", mean_duration_hours="mean", max_duration_hours="max")
        .reset_index()
        .sort_values(["rows", "mean_duration_hours"], ascending=False)
    )
    summary["mean_duration_hours"] = summary["mean_duration_hours"].round(2)
    summary["max_duration_hours"] = summary["max_duration_hours"].round(2)
    return summary


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


def extract_archive_hour_from_name(name: str) -> pd.Timestamp | None:
    match = re.search(
        r"states_(?P<date>\d{4}-\d{2}-\d{2})[-_](?P<hour>\d{2})",
        name,
    )
    if match is None:
        return None
    return pd.Timestamp(
        f"{match.group('date')} {match.group('hour')}:00:00",
        tz="UTC",
    )


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
    if value is None or value == "":
        return None

    numeric_value = parse_optional_float(value)
    if numeric_value is not None:
        unit = "ms" if abs(numeric_value) >= 1e11 else "s"
        try:
            return pd.Timestamp(numeric_value, unit=unit, tz="UTC")
        except Exception:
            return None

    try:
        parsed = pd.Timestamp(value)
    except Exception:
        return None

    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


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
            if not member.isfile():
                continue
            is_plain_csv = member.name.endswith(".csv")
            is_gzipped_csv = member.name.endswith(".csv.gz")
            if not is_plain_csv and not is_gzipped_csv:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            if is_gzipped_csv:
                gzip_stream = gzip.GzipFile(fileobj=extracted)
                text_stream = io.TextIOWrapper(gzip_stream, encoding="utf-8")
            else:
                text_stream = io.TextIOWrapper(extracted, encoding="utf-8")
            reader = csv.DictReader(text_stream)
            yield member.name, reader


def process_state_archive_streaming(
    engine: Engine,
    *,
    archive_metadata: dict[str, str],
    workspace_dir: Path,
    candidate_days: set[date],
    validate_schema: bool = False,
) -> int:
    source_file = archive_metadata["name"]
    source_url = archive_metadata["url"]
    manifest_status = get_manifest_status(engine, "weekly_state_vectors", source_file)
    if manifest_status == MANIFEST_STATUS_COMPLETED:
        print(f"Skipping already ingested state archive: {source_file}")
        return 0

    archive_day = extract_date_from_name(source_file)
    if archive_day is not None and candidate_days and archive_day not in candidate_days:
        print(f"Skipping archive outside candidate day set: {source_file}")
        return 0

    mark_manifest_started(
        engine,
        source_dataset="weekly_state_vectors",
        source_file=source_file,
        source_url=source_url,
    )
    temp_path: Path | None = None
    total_inserted = 0
    diagnostic_summary_text: str | None = None
    try:
        temp_path = download_to_temporary_file(source_url, workspace_dir, source_file)
        print(f"Processing state archive: {source_file}")
        flights = load_flights_for_day(engine, archive_day) if archive_day is not None else []
        if not flights:
            print(f"No eligible flights found for archive day: {archive_day}")
            mark_manifest_completed(
                engine,
                source_dataset="weekly_state_vectors",
                source_file=source_file,
                source_url=source_url,
                rows_inserted=0,
                error_message=f"No eligible flights found for archive day: {archive_day}",
            )
            return 0

        flights_by_icao24 = build_flight_lookup(flights)
        buffered_records: list[dict[str, Any]] = []

        first_member = True
        for member_name, reader in iter_csv_members_from_tar(temp_path):
            print(f"Reading member: {member_name}")
            if validate_schema and first_member:
                fieldnames = list(reader.fieldnames or [])
                validate_required_columns(
                    fieldnames,
                    REQUIRED_STATE_COLUMNS,
                    source_label=f"State archive member {source_file}:{member_name}",
                )
                print(
                    "State first-file schema check passed. Columns: "
                    f"{fieldnames}"
                )
                diagnostic_sample_times: list[str | None] = []
                diagnostic_sample_icao24: list[str | None] = []
                diagnostic_sample_callsigns: list[str | None] = []
                diagnostic_rows_with_time = 0
                diagnostic_rows_with_candidate_icao24 = 0
                diagnostic_rows_with_time_overlap = 0
                diagnostic_rows_matched = 0
            row_counter = 0
            match_counter = 0
            for row in reader:
                row_counter += 1
                if validate_schema and first_member:
                    if len(diagnostic_sample_times) < 5:
                        diagnostic_sample_times.append(row.get("time"))
                        diagnostic_sample_icao24.append(row.get("icao24"))
                        diagnostic_sample_callsigns.append(row.get("callsign"))
                    row_icao24 = (row.get("icao24") or "").strip().lower()
                    row_time = parse_epoch_timestamp(row.get("time"))
                    if row_time is not None:
                        diagnostic_rows_with_time += 1
                    if row_icao24 and row_icao24 in flights_by_icao24:
                        diagnostic_rows_with_candidate_icao24 += 1
                        if row_time is not None:
                            overlap_candidates = [
                                flight
                                for flight in flights_by_icao24[row_icao24]
                                if pd.Timestamp(flight["firstseen"]) <= row_time <= pd.Timestamp(flight["lastseen"])
                            ]
                            if overlap_candidates:
                                diagnostic_rows_with_time_overlap += 1
                matched_flight = match_state_row_to_flight(row, flights_by_icao24)
                if matched_flight is None:
                    continue
                if validate_schema and first_member:
                    diagnostic_rows_matched += 1
                buffered_records.append(
                    state_row_to_record(
                        row,
                        matched_flight,
                        source_file=source_file,
                        source_member=member_name,
                    )
                )
                match_counter += 1
                if len(buffered_records) >= STATE_VECTOR_INSERT_BATCH_SIZE:
                    total_inserted += insert_records(
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
            if validate_schema and first_member:
                diagnostic_summary_text = (
                    "sample_times="
                    f"{diagnostic_sample_times}, "
                    "sample_icao24="
                    f"{diagnostic_sample_icao24}, "
                    "sample_callsigns="
                    f"{diagnostic_sample_callsigns}, "
                    "rows_with_parseable_time="
                    f"{diagnostic_rows_with_time}, "
                    "rows_with_candidate_icao24="
                    f"{diagnostic_rows_with_candidate_icao24}, "
                    "rows_with_time_overlap="
                    f"{diagnostic_rows_with_time_overlap}, "
                    "rows_matched="
                    f"{diagnostic_rows_matched}"
                )
                print("State first-member diagnostic summary:")
                print(diagnostic_summary_text)
                first_member = False

        if buffered_records:
            total_inserted += insert_records(
                engine,
                "scientific_state_vectors",
                buffered_records,
                conflict_columns=["flight_key", "time", "icao24", "callsign"],
            )

        if total_inserted == 0 and diagnostic_summary_text is not None:
            print(
                "No matching state-vector rows were inserted for "
                f"{source_file}. First-member diagnostics: {diagnostic_summary_text}"
            )

        mark_manifest_completed(
            engine,
            source_dataset="weekly_state_vectors",
            source_file=source_file,
            source_url=source_url,
            rows_inserted=total_inserted,
            error_message=(
                f"No matched state rows. Diagnostics: {diagnostic_summary_text}"
                if total_inserted == 0 and diagnostic_summary_text is not None
                else None
            ),
        )
        return total_inserted
    except Exception as exc:
        mark_manifest_failed(
            engine,
            source_dataset="weekly_state_vectors",
            source_file=source_file,
            source_url=source_url,
            error_message=str(exc),
        )
        raise
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def ingest_matching_state_vectors_streaming(
    engine: Engine,
    state_archives: Sequence[dict[str, str]],
    *,
    workspace_dir: Path,
    candidate_days: set[date],
) -> int:
    total_inserted = 0
    for index, archive_metadata in enumerate(progress(
        state_archives,
        desc="Ingest state vectors",
        total=len(state_archives),
        unit="archive",
    )):
        total_inserted += process_state_archive_streaming(
            engine,
            archive_metadata=archive_metadata,
            workspace_dir=workspace_dir,
            candidate_days=candidate_days,
            validate_schema=index == 0,
        )
    return total_inserted


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


def select_relevant_state_archives(
    archives: Sequence[dict[str, str]],
    candidate_days: set[date],
    candidate_hours: set[pd.Timestamp],
) -> list[dict[str, str]]:
    selected = []
    for archive in archives:
        archive_day = extract_date_from_name(archive["name"])
        if archive_day is not None and candidate_days and archive_day not in candidate_days:
            continue
        archive_hour = extract_archive_hour_from_name(archive["name"])
        if archive_hour is not None and candidate_hours and archive_hour not in candidate_hours:
            continue
        selected.append(archive)
    return selected


def build_scientific_db(config: BuildConfig) -> None:
    if config.state_archive_format != "csv":
        raise NotImplementedError(
            "This first implementation supports CSV state-vector archives only."
        )

    engine = get_engine(config.database_url)
    create_schema(engine)
    workspace_dir = ensure_workspace(config.download_dir)

    covid_files = fetch_covid_file_metadata()
    if not covid_files:
        raise RuntimeError("Could not discover COVID dataset files.")
    selected_covid_files = list(
        covid_files[: config.max_covid_files]
        if config.max_covid_files is not None
        else covid_files
    )
    inserted_flights = ingest_filtered_covid_flights_streaming(
        engine,
        selected_covid_files,
        workspace_dir=workspace_dir,
        origin_airports=config.origin_airports,
        minimum_duration_hours=config.minimum_duration_hours,
    )
    print(f"Inserted or refreshed {inserted_flights} filtered flight rows from streaming ingest.")
    filtered_flight_summary = summarize_filtered_flights_from_db(
        engine,
        origin_airports=config.origin_airports,
        minimum_duration_hours=config.minimum_duration_hours,
    )
    if filtered_flight_summary.empty:
        print("Filtered flight summary is empty after Step 1.")
    else:
        print("\n--- Filtered Flight Summary ---")
        print(filtered_flight_summary.to_string(index=False))

    candidate_days = load_candidate_days_from_db(
        engine,
        origin_airports=config.origin_airports,
        minimum_duration_hours=config.minimum_duration_hours,
    )
    candidate_hours = load_candidate_archive_hours_from_db(
        engine,
        origin_airports=config.origin_airports,
        minimum_duration_hours=config.minimum_duration_hours,
    )
    print(f"Identified {len(candidate_days)} candidate UTC days with eligible flights.")
    print(f"Identified {len(candidate_hours)} candidate UTC hourly archives with eligible flights.")
    if not candidate_days:
        raise RuntimeError(
            "No eligible long-haul flights were inserted into scientific_flights. "
            "This usually means the origin-airport or duration filters are too "
            "restrictive for the selected COVID files, or the source schema does not "
            "match the expected origin/destination columns. Diagnose with SQL such as: "
            "\"SELECT status, source_file, rows_inserted FROM scientific_ingest_manifest "
            "WHERE source_dataset = 'covid_flight_dataset' ORDER BY source_file;\" and "
            "\"SELECT COUNT(*) FROM scientific_flights;\""
        )

    state_archives = fetch_state_archive_metadata(config.state_archive_format)
    if not state_archives:
        raise RuntimeError("Could not discover weekly state-vector archives.")

    filtered_archives = select_relevant_state_archives(
        state_archives,
        candidate_days,
        candidate_hours,
    )
    selected_state_archives = list(
        filtered_archives[: config.max_state_archives]
        if config.max_state_archives is not None
        else filtered_archives
    )
    print(
        "Selected "
        f"{len(selected_state_archives)} state archives after hourly relevance filtering."
    )
    inserted_state_rows = ingest_matching_state_vectors_streaming(
        engine,
        selected_state_archives,
        workspace_dir=workspace_dir,
        candidate_days=candidate_days,
    )
    print(f"Inserted or refreshed {inserted_state_rows} matching state-vector rows.")


def reset_scientific_db(database_url: str, *, confirm: bool) -> None:
    if not confirm:
        raise RuntimeError(
            "Refusing to reset the scientific database without explicit confirmation. "
            "Re-run with: opensky_build_scientific_db.py reset-build --confirm-reset"
        )

    engine = get_engine(database_url)
    create_schema(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                TRUNCATE TABLE
                    scientific_state_vectors,
                    scientific_flights,
                    scientific_ingest_manifest
                RESTART IDENTITY
                """
            )
        )
    print("Reset scientific_state_vectors, scientific_flights, and scientific_ingest_manifest.")


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
        help=(
            "Temporary workspace directory for one-file-at-a-time source processing, "
            "e.g. an external drive path."
        ),
    )
    build_parser.add_argument(
        "--origin-airports",
        default=",".join(DEFAULT_ORIGIN_AIRPORTS),
        help="Comma-separated origin airports to retain. Default: EGLL,OMDB,OTHH,EDDF",
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

    reset_parser = subparsers.add_parser(
        "reset-build",
        help="Clear scientific DB tables and ingest history for a clean rebuild.",
    )
    reset_parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help=(
            "SQLAlchemy PostgreSQL URL. Default: "
            f"{DEFAULT_DATABASE_URL}"
        ),
    )
    reset_parser.add_argument(
        "--confirm-reset",
        action="store_true",
        help="Required safety flag to confirm truncating scientific DB tables.",
    )

    inspect_covid_parser = subparsers.add_parser(
        "inspect-covid",
        help="Dry-run inspection of unfiltered COVID manifests without DB inserts.",
    )
    inspect_covid_parser.add_argument(
        "--download-dir",
        required=True,
        help="Temporary workspace directory for one-file-at-a-time source inspection.",
    )
    inspect_covid_parser.add_argument(
        "--max-covid-files",
        type=int,
        default=2,
        help="Number of COVID manifest files to inspect. Default: 2",
    )
    inspect_covid_parser.add_argument(
        "--max-covid-chunks-per-file",
        type=int,
        default=1,
        help="Number of chunks to inspect per COVID file. Default: 1",
    )
    inspect_covid_parser.add_argument(
        "--top-n-airports",
        type=int,
        default=20,
        help="Number of top origins/destinations to print. Default: 20",
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

    if args.command == "reset-build":
        reset_scientific_db(
            args.database_url,
            confirm=args.confirm_reset,
        )
        return

    if args.command == "inspect-covid":
        inspect_covid_data(
            InspectCovidConfig(
                download_dir=Path(args.download_dir),
                max_covid_files=args.max_covid_files,
                max_covid_chunks_per_file=args.max_covid_chunks_per_file,
                top_n_airports=args.top_n_airports,
            )
        )
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
