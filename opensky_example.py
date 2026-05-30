"""
OpenSky long-haul flight and trajectory example.

This script demonstrates a two-step workflow for retrieving long-haul flights
from OpenSky or a local scientific PostgreSQL database and then building a
trajectory dataset for those flights.

Workflow
--------
Step 1:
- Query flight metadata for departures from one origin airport within a time
  window.
- Filter the returned flights by minimum duration and known arrival airport.
- Save the filtered flight list to ``long_haul_flights.csv``.

Step 2:
- Load the saved Step 1 flight metadata.
- Fetch trajectory/state-vector data for each saved flight.
- Append trajectory rows to ``long_distance_trajectories.csv``.
- Generate an altitude-vs-time-since-takeoff plot as
  ``flights_altitude_vs_time.png``.

Backends
--------
The script supports four backend modes:

- ``auto``:
  Prefer the local scientific PostgreSQL database when it is available. If the
  scientific DB is not available, fall back to Trino.

- ``scientific_db``:
  Use the local PostgreSQL database built by ``opensky_build_scientific_db.py``.
  This is the preferred option when that database has already been built.

- ``trino``:
  Uses the OpenSky Trino historical database.

- ``rest``:
  Uses the OpenSky REST API. This path supports chunking, retry handling, and
  resumable Step 2 execution, but it is still constrained by REST rate limits.

CLI options
-----------
The main user-facing options are:

- ``--step 1|2|both``
  Select which workflow step to run.

- ``--backend auto|scientific_db|rest|trino``
  Select the data backend. Default: ``auto``.

- ``--origin-airport ICAO``
  Origin airport ICAO identifier. Default: ``EGLL``.

- ``--start-time "YYYY-MM-DD HH:MM:SS"``
  Inclusive search start timestamp.

- ``--end-time "YYYY-MM-DD HH:MM:SS"``
  Inclusive search end timestamp.

- ``--minimum-duration-hours FLOAT``
  Minimum duration threshold used to classify long-haul flights.

Example commands
----------------
Run Step 1 with the default auto backend:

    python opensky_example.py --step 1

Run Step 1 with the local scientific DB:

    python opensky_example.py --step 1 --backend scientific_db

Run Step 1 with Trino:

    python opensky_example.py --step 1 --backend trino

Run Step 2 with Trino:

    python opensky_example.py --step 2 --backend trino

Run Step 1 with REST:

    python opensky_example.py --step 1 --backend rest

Resume a rate-limited REST Step 2 run:

    python opensky_example.py --step 2 --backend rest

Requirements
------------
Python dependencies:

- pandas
- matplotlib
- pyopensky
- httpx

Backend-specific requirements:

- Scientific DB backend:
  Build the local PostgreSQL dataset first with ``opensky_build_scientific_db.py``.
  The default database URL matches the local setup created by
  ``opensky_create_postgresql_db.sh``.

- REST backend:
  Configure OpenSky REST credentials if you want authenticated access.
  ``pyopensky.rest.REST`` reads credentials from the standard pyopensky config.

- Trino backend:
  You must have Trino access enabled by OpenSky and valid Trino credentials in
  the pyopensky settings file, typically:

      /Users/[username]/Library/Application Support/pyopensky/settings.conf

Output files
------------
- ``long_haul_flights.csv``:
  Filtered Step 1 flight metadata.

- ``long_distance_trajectories.csv``:
  Step 2 trajectory/state-vector output written in a Trino-style schema for
  both backends.

- ``flights_altitude_vs_time.png``:
  Plot generated from the saved Step 2 data. The plot prefers ``geoaltitude``
  when available and falls back to ``baroaltitude``.

Testing
-------
Unit tests live in ``tests/test_opensky_example.py``.

Run the built-in unittest suite with the project virtual environment:

    pipenv run python -m unittest discover -s tests -v

If using pytest, make sure the project root is importable:

    PYTHONPATH=. pipenv run pytest tests/test_opensky_example.py -v

Notes
-----
- Step 2 is resumable because completed flights are tracked via ``flight_key``
  values already present in the Step 2 output CSV.
- REST Step 2 may stop at a rate-limit checkpoint and print a retry/resume
  command.
- Trino Step 2 can return much richer state-vector data than REST, even though
  both backends are normalized into one common output schema.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Literal

import httpx
import matplotlib
import pandas as pd
from opensky_build_scientific_db import DEFAULT_DATABASE_URL
from opensky_build_scientific_db import get_engine
from opensky_build_scientific_db import query_scientific_db
from pyopensky.rest import REST
from pyopensky.trino import Trino
from sqlalchemy import text


BackendName = Literal["auto", "scientific_db", "rest", "trino"]
ResolvedBackendName = Literal["scientific_db", "rest", "trino"]
StepName = Literal["1", "2", "both"]


def configure_matplotlib_backend() -> bool:
    try:
        if sys.platform == "darwin":
            matplotlib.use("MacOSX")
        else:
            matplotlib.use("TkAgg")
        return True
    except Exception:
        matplotlib.use("Agg")
        return False


interactive_backend = configure_matplotlib_backend()
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).parent
DEFAULT_STEP_1_OUTPUT_PATH = SCRIPT_DIR / "opensky_results" /"long_haul_flights.csv"
DEFAULT_STEP_2_OUTPUT_PATH = SCRIPT_DIR / "opensky_results" /"long_distance_trajectories.csv"
DEFAULT_PLOT_OUTPUT_PATH = SCRIPT_DIR / "opensky_results" /"flights_altitude_vs_time.png"

REST_INTERVAL_LIMIT = pd.Timedelta(days=2)
REST_CHUNK_RETRY_ATTEMPTS = 3
REST_CHUNK_RETRY_SLEEP_SECONDS = 2
REST_TRACK_RETRY_ATTEMPTS = 3
REST_TRACK_RETRY_SLEEP_SECONDS = 2
REST_TRACK_MAX_RETRY_AFTER_SECONDS = 30
TRACK_REQUEST_COOLDOWN_SECONDS = 5
STEP_2_DEFAULT_RETRY_WAIT_SECONDS = 3600
SCIENTIFIC_DB_STEP_2_BATCH_SIZE = 500

DEFAULT_BACKEND: BackendName = "auto"
DEFAULT_STEP: StepName = "1"
DEFAULT_ORIGIN_AIRPORT = "EGLL"  # London Heathrow
DEFAULT_START_TIME = "2026-01-24 00:00:00"
DEFAULT_END_TIME = "2026-05-24 23:59:59"
DEFAULT_MINIMUM_DURATION_HOURS = 6
TRINO_CREDENTIALS_HINT = (
    "/Users/anthonydills/Library/Application Support/pyopensky/settings.conf"
)
STEP_2_SCHEMA_COLUMNS = [
    "time",
    "icao24",
    "lat",
    "lon",
    "velocity",
    "heading",
    "vertrate",
    "callsign",
    "onground",
    "alert",
    "spi",
    "squawk",
    "baroaltitude",
    "geoaltitude",
    "lastposupdate",
    "lastcontact",
    "serials",
    "hour",
    "flight_key",
    "origin",
    "destination",
    "first_seen",
    "last_seen",
    "duration_hours",
    "source_backend",
]


@dataclass(frozen=True)
class OutputPaths:
    step_1_output_path: Path = DEFAULT_STEP_1_OUTPUT_PATH
    step_2_output_path: Path = DEFAULT_STEP_2_OUTPUT_PATH
    plot_output_path: Path = DEFAULT_PLOT_OUTPUT_PATH


@dataclass(frozen=True)
class RunConfig:
    backend: ResolvedBackendName
    step: StepName
    origin_airport: str
    start_time: str
    end_time: str
    minimum_duration_hours: float
    database_url: str
    paths: OutputPaths


@dataclass(frozen=True)
class ScientificDbClient:
    database_url: str


class Step2RateLimitError(RuntimeError):
    def __init__(self, response: httpx.Response):
        super().__init__(f"Rate limited with HTTP {response.status_code}")
        self.response = response


def build_flight_key(
    icao24: str,
    callsign: str,
    first_seen: str | pd.Timestamp,
) -> str:
    callsign_clean = str(callsign).strip()
    first_seen_ts = pd.Timestamp(first_seen)
    return f"{icao24}|{callsign_clean}|{first_seen_ts.isoformat()}"


def make_backend_client(
    backend: ResolvedBackendName,
    database_url: str,
) -> REST | Trino | ScientificDbClient:
    if backend == "scientific_db":
        return ScientificDbClient(database_url=database_url)
    if backend == "rest":
        return REST()
    return Trino()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        backend=args.backend,
        step=args.step,
        origin_airport=args.origin_airport,
        start_time=args.start_time,
        end_time=args.end_time,
        minimum_duration_hours=args.minimum_duration_hours,
        database_url=args.database_url,
        paths=OutputPaths(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch long-haul flight metadata and trajectories from "
            "OpenSky REST or Trino."
        )
    )
    parser.add_argument(
        "--step",
        choices=["1", "2", "both"],
        default=DEFAULT_STEP,
        help="Which workflow step to run: 1, 2, or both.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "scientific_db", "rest", "trino"],
        default=DEFAULT_BACKEND,
        help=(
            "Data backend to use. 'auto' prefers the local scientific DB and "
            "falls back to Trino."
        ),
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help=(
            "SQLAlchemy database URL for the local scientific DB backend. "
            f"Default: {DEFAULT_DATABASE_URL}"
        ),
    )
    parser.add_argument(
        "--origin-airport",
        default=DEFAULT_ORIGIN_AIRPORT,
        help="Origin airport ICAO code for the departure search.",
    )
    parser.add_argument(
        "--start-time",
        default=DEFAULT_START_TIME,
        help="Inclusive departure search start timestamp.",
    )
    parser.add_argument(
        "--end-time",
        default=DEFAULT_END_TIME,
        help="Inclusive departure search end timestamp.",
    )
    parser.add_argument(
        "--minimum-duration-hours",
        type=float,
        default=DEFAULT_MINIMUM_DURATION_HOURS,
        help="Minimum duration in hours for a flight to count as long-haul.",
    )
    return parser.parse_args()


def scientific_db_is_available(
    database_url: str,
    step: StepName,
) -> bool:
    try:
        engine = get_engine(database_url)
        with engine.connect() as connection:
            status_row = connection.execute(
                text(
                    """
                    SELECT
                        to_regclass('public.scientific_flights') AS flights_table,
                        to_regclass('public.scientific_state_vectors') AS states_table
                    """
                )
            ).mappings().one()
    except Exception:
        return False

    if status_row["flights_table"] is None:
        return False
    if step in {"2", "both"} and status_row["states_table"] is None:
        return False
    return True


def resolve_backend(
    requested_backend: BackendName,
    *,
    database_url: str,
    step: StepName,
) -> ResolvedBackendName:
    if requested_backend != "auto":
        return requested_backend

    if scientific_db_is_available(database_url, step):
        print(
            "Using local scientific PostgreSQL database backend at "
            f"{database_url}."
        )
        return "scientific_db"

    print(
        "Local scientific PostgreSQL database not available. "
        "Falling back to Trino."
    )
    return "trino"


def plot_altitude_vs_time_since_takeoff(
    trajectory_df: pd.DataFrame,
    plot_output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for (_, callsign, destination), flight_df in trajectory_df.groupby(
        ["flight_key", "callsign", "destination"], dropna=False
    ):
        flight_df = flight_df.sort_values("time").copy()
        first_time = flight_df["time"].iloc[0]
        flight_df["plot_altitude_m"] = select_altitude_for_plot(flight_df)
        flight_df["hours_since_takeoff"] = (
            pd.to_datetime(flight_df["time"]) - pd.to_datetime(first_time)
        ).dt.total_seconds() / 3600

        label = f"{str(callsign).strip()} -> {destination}"
        ax.plot(
            flight_df["hours_since_takeoff"],
            flight_df["plot_altitude_m"],
            label=label,
        )

    ax.set_xlabel("Hours Since Takeoff")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude vs Time Since Takeoff")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_output_path, dpi=150)
    print(f"Saved plot to {plot_output_path}")
    if interactive_backend:
        plt.show()
    plt.close(fig)


def select_altitude_for_plot(trajectory_df: pd.DataFrame) -> pd.Series:
    geoaltitude = trajectory_df.get("geoaltitude")
    if geoaltitude is None:
        geoaltitude = pd.Series(pd.NA, index=trajectory_df.index, dtype="Float64")

    baroaltitude = trajectory_df.get("baroaltitude")
    if baroaltitude is None:
        baroaltitude = pd.Series(pd.NA, index=trajectory_df.index, dtype="Float64")

    return geoaltitude.fillna(baroaltitude)


def fetch_departure_chunk_rest(
    api: REST,
    airport: str,
    begin: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    chunk_df = pd.DataFrame()
    for attempt in range(1, REST_CHUNK_RETRY_ATTEMPTS + 1):
        try:
            chunk_df = api.departure(
                airport=airport,
                begin=begin,
                end=end,
            )
            return chunk_df
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 404:
                return pd.DataFrame()

            retriable = status_code in (429, 503)
            if not retriable or attempt == REST_CHUNK_RETRY_ATTEMPTS:
                raise

            sleep_seconds = REST_CHUNK_RETRY_SLEEP_SECONDS * attempt
            print(
                f"Chunk {begin} to {end} returned HTTP {status_code}. "
                f"Sleeping {sleep_seconds}s before retry "
                f"{attempt + 1}/{REST_CHUNK_RETRY_ATTEMPTS}."
            )
            time.sleep(sleep_seconds)

    return chunk_df


def fetch_departures_for_interval_rest(
    api: REST,
    airport: str,
    begin: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    begin_ts = pd.Timestamp(begin)
    end_ts = pd.Timestamp(end)

    if end_ts <= begin_ts:
        raise ValueError("end must be later than begin")

    interval = end_ts - begin_ts
    if interval <= REST_INTERVAL_LIMIT:
        print(f"Fetching departures from {begin_ts} to {end_ts}...")
        return fetch_departure_chunk_rest(
            api=api,
            airport=airport,
            begin=begin_ts,
            end=end_ts,
        )

    total_chunks = int(interval / REST_INTERVAL_LIMIT)
    if interval % REST_INTERVAL_LIMIT != pd.Timedelta(0):
        total_chunks += 1

    print(
        f"Requested interval is {interval}. "
        f"Chunking into {total_chunks} REST requests..."
    )

    all_chunks: list[pd.DataFrame] = []
    current_start = begin_ts
    chunk_index = 1

    while current_start < end_ts:
        current_end = min(current_start + REST_INTERVAL_LIMIT, end_ts)
        print(
            f"[{chunk_index}/{total_chunks}] Fetching departures from "
            f"{current_start} to {current_end}..."
        )

        chunk_df = fetch_departure_chunk_rest(
            api=api,
            airport=airport,
            begin=current_start,
            end=current_end,
        )

        if not chunk_df.empty:
            all_chunks.append(chunk_df)

        current_start = current_end
        chunk_index += 1

    if not all_chunks:
        return pd.DataFrame()

    stitched_df = pd.concat(all_chunks, ignore_index=True)
    dedupe_columns = [
        "firstSeen",
        "lastSeen",
        "icao24",
        "callsign",
        "estDepartureAirport",
        "estArrivalAirport",
    ]
    available_dedupe_columns = [
        column for column in dedupe_columns if column in stitched_df.columns
    ]
    if available_dedupe_columns:
        stitched_df = stitched_df.drop_duplicates(
            subset=available_dedupe_columns
        ).reset_index(drop=True)

    return stitched_df


def fetch_track_chunk_rest(
    api: REST,
    icao24: str,
    ts: str | pd.Timestamp,
) -> pd.DataFrame:
    ts_int = int(pd.Timestamp(ts).timestamp())
    url = f"https://opensky-network.org/api/tracks/?icao24={icao24}&time={ts_int}"

    for attempt in range(1, REST_TRACK_RETRY_ATTEMPTS + 1):
        response = api.client.get(url, headers=api.headers)
        status_code = response.status_code

        if status_code == 404:
            return pd.DataFrame()

        if status_code == 429 and attempt == REST_TRACK_RETRY_ATTEMPTS:
            raise Step2RateLimitError(response)

        if status_code in (429, 503):
            retry_after_header = response.headers.get(
                "X-Rate-Limit-Retry-After-Seconds"
            )
            fallback_sleep = REST_TRACK_RETRY_SLEEP_SECONDS * attempt
            sleep_seconds = fallback_sleep

            if retry_after_header is not None:
                try:
                    retry_after_seconds = int(retry_after_header)
                    if retry_after_seconds > 0:
                        sleep_seconds = min(
                            retry_after_seconds,
                            REST_TRACK_MAX_RETRY_AFTER_SECONDS,
                        )
                except ValueError:
                    sleep_seconds = fallback_sleep

            print(
                f"Track lookup for {icao24} returned HTTP {status_code}. "
                f"Sleeping {sleep_seconds}s before retry "
                f"{attempt + 1}/{REST_TRACK_RETRY_ATTEMPTS}."
            )
            time.sleep(sleep_seconds)
            continue

        response.raise_for_status()
        json = response.json()
        return (
            pd.DataFrame.from_records(
                json["path"],
                columns=[
                    "timestamp",
                    "latitude",
                    "longitude",
                    "altitude",
                    "track",
                    "onground",
                ],
            )
            .assign(
                timestamp=lambda df: pd.to_datetime(
                    df.timestamp, utc=True, unit="s"
                ),
                icao24=json["icao24"],
                callsign=json["callsign"],
            )
            .convert_dtypes(dtype_backend="pyarrow")
        )

    return pd.DataFrame()


def fetch_flights_step_1_rest(
    config: RunConfig,
    client: REST,
) -> pd.DataFrame:
    flights_df = fetch_departures_for_interval_rest(
        api=client,
        airport=config.origin_airport,
        begin=config.start_time,
        end=config.end_time,
    )
    if flights_df.empty:
        return flights_df
    return flights_df.assign(source_backend="rest")


def fetch_flights_step_1_trino(
    config: RunConfig,
    client: Trino,
) -> pd.DataFrame:
    flights_df = client.flightlist(
        start=config.start_time,
        stop=config.end_time,
        departure_airport=config.origin_airport,
    )
    if flights_df is None or flights_df.empty:
        return pd.DataFrame()

    normalized_df = flights_df.rename(
        columns={
            "departure": "estDepartureAirport",
            "arrival": "estArrivalAirport",
            "firstseen": "firstSeen",
            "lastseen": "lastSeen",
        }
    )
    return normalized_df.assign(source_backend="trino")


def fetch_flights_step_1_scientific_db(
    config: RunConfig,
    client: ScientificDbClient,
) -> pd.DataFrame:
    sql = """
    SELECT
        icao24,
        callsign,
        origin AS "estDepartureAirport",
        destination AS "estArrivalAirport",
        firstseen AS "firstSeen",
        lastseen AS "lastSeen",
        duration_hours,
        flight_key,
        'scientific_db' AS source_backend
    FROM scientific_flights
    WHERE origin = :origin_airport
      AND duration_hours >= :minimum_duration_hours
      AND firstseen >= :start_time
      AND lastseen <= :end_time
      AND destination IS NOT NULL
    ORDER BY firstseen
    """
    flights_df = pd.read_sql_query(
        text(sql),
        get_engine(client.database_url),
        params={
            "origin_airport": config.origin_airport,
            "minimum_duration_hours": config.minimum_duration_hours,
            "start_time": config.start_time,
            "end_time": config.end_time,
        },
    )
    if flights_df.empty:
        return flights_df

    flights_df["firstSeen"] = pd.to_datetime(flights_df["firstSeen"], utc=True)
    flights_df["lastSeen"] = pd.to_datetime(flights_df["lastSeen"], utc=True)
    flights_df["callsign"] = flights_df["callsign"].astype(str).str.strip()
    return flights_df


def prepare_long_haul_flights(
    flights_df: pd.DataFrame,
    minimum_duration_hours: float,
) -> pd.DataFrame:
    long_haul_flights = (
        flights_df.assign(
            duration_hours=(
                flights_df["lastSeen"] - flights_df["firstSeen"]
            ).dt.total_seconds()
            / 3600
        )
        .loc[lambda df: df["duration_hours"] >= minimum_duration_hours]
        .loc[lambda df: df["estArrivalAirport"].notna()]
        .copy()
    )

    long_haul_flights["callsign"] = (
        long_haul_flights["callsign"].astype(str).str.strip()
    )
    long_haul_flights["flight_key"] = long_haul_flights.apply(
        lambda row: build_flight_key(
            row["icao24"],
            row["callsign"],
            row["firstSeen"],
        ),
        axis=1,
    )
    return long_haul_flights.reset_index(drop=True)


def save_step_1_results(
    long_haul_flights: pd.DataFrame,
    output_path: Path,
) -> None:
    long_haul_flights.to_csv(output_path, index=False)
    print(
        f"Saved {len(long_haul_flights)} long-haul flight records to "
        f"{output_path}"
    )


def load_step_1_results(step_1_output_path: Path) -> pd.DataFrame:
    if not step_1_output_path.exists():
        raise FileNotFoundError(
            f"Step 1 output not found: {step_1_output_path}. Run Step 1 first."
        )

    return pd.read_csv(
        step_1_output_path,
        parse_dates=["firstSeen", "lastSeen"],
    )


def fetch_track_batch_scientific_db(
    database_url: str,
    flight_keys: list[str],
) -> pd.DataFrame:
    if not flight_keys:
        return pd.DataFrame(columns=STEP_2_SCHEMA_COLUMNS)

    placeholders = []
    sql_params: dict[str, Any] = {}
    for index, flight_key in enumerate(flight_keys):
        parameter_name = f"flight_key_{index}"
        placeholders.append(f":{parameter_name}")
        sql_params[parameter_name] = flight_key

    sql_query = f"""
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
    JOIN scientific_flights f
      ON sv.flight_key = f.flight_key
    WHERE sv.flight_key IN ({", ".join(placeholders)})
    ORDER BY f.firstseen, sv.time
    """
    return standardize_step_2_schema(
        query_scientific_db(
            database_url,
            sql_query=sql_query,
            sql_params=sql_params,
        )
    )


def normalize_rest_track_dataframe(
    track_df: pd.DataFrame,
    flight: pd.Series,
) -> pd.DataFrame:
    normalized_df = track_df.rename(
        columns={
            "timestamp": "time",
            "latitude": "lat",
            "longitude": "lon",
            "altitude": "baroaltitude",
            "track": "heading",
        }
    ).assign(
        flight_key=flight["flight_key"],
        origin=flight["estDepartureAirport"],
        destination=flight["estArrivalAirport"],
        first_seen=flight["firstSeen"],
        last_seen=flight["lastSeen"],
        duration_hours=flight["duration_hours"],
        source_backend="rest",
        velocity=pd.NA,
        vertrate=pd.NA,
        alert=pd.NA,
        spi=pd.NA,
        squawk=pd.NA,
        geoaltitude=pd.NA,
        lastposupdate=pd.NA,
        lastcontact=pd.NA,
        serials=pd.NA,
    )
    normalized_df["hour"] = pd.to_datetime(normalized_df["time"]).dt.floor("h")
    return standardize_step_2_schema(normalized_df)


def fetch_track_chunk_trino(
    client: Trino,
    flight: pd.Series,
) -> pd.DataFrame:
    history_df = client.history(
        start=flight["firstSeen"],
        stop=flight["lastSeen"],
        icao24=flight["icao24"],
        callsign=flight["callsign"],
    )
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    return history_df


def normalize_trino_track_dataframe(
    track_df: pd.DataFrame,
    flight: pd.Series,
) -> pd.DataFrame:
    normalized_df = track_df.assign(
        flight_key=flight["flight_key"],
        origin=flight["estDepartureAirport"],
        destination=flight["estArrivalAirport"],
        first_seen=flight["firstSeen"],
        last_seen=flight["lastSeen"],
        duration_hours=flight["duration_hours"],
        source_backend="trino",
    )
    return standardize_step_2_schema(normalized_df)


def standardize_step_2_schema(track_df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = track_df.copy()
    for column in STEP_2_SCHEMA_COLUMNS:
        if column not in normalized_df.columns:
            normalized_df[column] = pd.NA
    return normalized_df.loc[:, STEP_2_SCHEMA_COLUMNS]


def load_completed_flight_keys(step_2_output_path: Path) -> set[str]:
    if not step_2_output_path.exists():
        return set()

    completed_df = pd.read_csv(step_2_output_path, usecols=["flight_key"])
    return set(completed_df["flight_key"].dropna().astype(str).unique())


def append_tracks_to_output(
    track_df: pd.DataFrame,
    step_2_output_path: Path,
) -> None:
    write_header = not step_2_output_path.exists()
    track_df.to_csv(
        step_2_output_path,
        mode="a",
        header=write_header,
        index=False,
    )


def load_step_2_results(step_2_output_path: Path) -> pd.DataFrame:
    if not step_2_output_path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        step_2_output_path,
        parse_dates=["time", "first_seen", "last_seen"],
    )


def estimate_step_2_retry_time(
    response: httpx.Response | None,
) -> tuple[pd.Timestamp, str]:
    now_utc = pd.Timestamp.now(tz="UTC")
    retry_seconds = STEP_2_DEFAULT_RETRY_WAIT_SECONDS
    reason = "default 1 hour retry window"

    if response is not None:
        retry_after_header = response.headers.get(
            "X-Rate-Limit-Retry-After-Seconds"
        )
        if retry_after_header is not None:
            try:
                retry_after_seconds = int(retry_after_header)
                if 0 < retry_after_seconds <= 6 * 3600:
                    retry_seconds = retry_after_seconds
                    reason = "server retry-after header"
                else:
                    reason = (
                        "server retry-after header was unrealistic; "
                        "using default 1 hour retry window"
                    )
            except ValueError:
                reason = (
                    "server retry-after header was invalid; "
                    "using default 1 hour retry window"
                )

    return now_utc + pd.Timedelta(seconds=retry_seconds), reason


def print_step_2_checkpoint(
    current_flight: pd.Series,
    completed_count: int,
    remaining_count: int,
    response: httpx.Response | None,
    backend: BackendName,
) -> None:
    retry_timestamp_utc, retry_reason = estimate_step_2_retry_time(response)
    retry_command = f"python opensky_example.py --step 2 --backend {backend}"

    print("\n--- Step 2 Checkpoint ---")
    print(
        f"Stopped after rate limiting on {current_flight['callsign']} "
        f"({current_flight['icao24']})."
    )
    print(f"Completed flights with saved trajectories: {completed_count}")
    print(f"Flights remaining to fetch: {remaining_count}")
    print(
        "Estimated retry time: "
        f"{retry_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} "
        f"({retry_reason})."
    )
    print(f"Resume with: {retry_command}")


def print_step_2_summary(
    final_trajectory_df: pd.DataFrame,
    config: RunConfig,
    stopped_early: bool,
) -> None:
    if final_trajectory_df.empty:
        print("No trajectory rows have been saved yet.")
        return

    columns_to_show = [
        "time",
        "icao24",
        "callsign",
        "lat",
        "lon",
        "baroaltitude",
        "geoaltitude",
        "heading",
        "origin",
        "destination",
    ]
    print("\n--- Trajectory Sample ---")
    print(final_trajectory_df[columns_to_show].head())
    plot_altitude_vs_time_since_takeoff(
        final_trajectory_df,
        config.paths.plot_output_path,
    )
    print(f"Trajectory output is stored at {config.paths.step_2_output_path}")
    if stopped_early:
        print("Step 2 paused at a rate-limit checkpoint. Resume with:")
        print(f"python opensky_example.py --step 2 --backend {config.backend}")


def run_step_1_rest(
    config: RunConfig,
    client: REST,
) -> None:
    print(
        f"--- Step 1 ({config.backend}): Fetching departed flights from "
        f"{config.origin_airport} ---"
    )
    flights_df = fetch_flights_step_1_rest(config, client)
    if flights_df.empty:
        print("No flights returned by the REST API for that airport and time window.")
        raise SystemExit(0)

    long_haul_flights = prepare_long_haul_flights(
        flights_df,
        minimum_duration_hours=config.minimum_duration_hours,
    )
    print(
        f"Found {len(long_haul_flights)} flights with duration >= "
        f"{config.minimum_duration_hours} hours and a known arrival airport."
    )
    if long_haul_flights.empty:
        raise SystemExit(0)

    print(
        long_haul_flights[
            [
                "callsign",
                "icao24",
                "estDepartureAirport",
                "estArrivalAirport",
                "firstSeen",
                "lastSeen",
                "duration_hours",
                "flight_key",
                "source_backend",
            ]
        ].head()
    )
    save_step_1_results(long_haul_flights, config.paths.step_1_output_path)


def run_step_1_trino(
    config: RunConfig,
    client: Trino,
) -> None:
    print(
        f"--- Step 1 ({config.backend}): Fetching departed flights from "
        f"{config.origin_airport} ---"
    )
    try:
        flights_df = fetch_flights_step_1_trino(config, client)
    except Exception as exc:
        raise RuntimeError(
            "Trino Step 1 failed. Check that your Trino access is enabled and "
            f"your credentials are configured in {TRINO_CREDENTIALS_HINT}. "
            "You can retry with --backend rest if needed."
        ) from exc

    if flights_df.empty:
        print("No flights returned by Trino for that airport and time window.")
        raise SystemExit(0)

    long_haul_flights = prepare_long_haul_flights(
        flights_df,
        minimum_duration_hours=config.minimum_duration_hours,
    )
    print(
        f"Found {len(long_haul_flights)} flights with duration >= "
        f"{config.minimum_duration_hours} hours and a known arrival airport."
    )
    if long_haul_flights.empty:
        raise SystemExit(0)

    print(
        long_haul_flights[
            [
                "callsign",
                "icao24",
                "estDepartureAirport",
                "estArrivalAirport",
                "firstSeen",
                "lastSeen",
                "duration_hours",
                "flight_key",
                "source_backend",
            ]
        ].head()
    )
    save_step_1_results(long_haul_flights, config.paths.step_1_output_path)


def run_step_1_scientific_db(
    config: RunConfig,
    client: ScientificDbClient,
) -> None:
    print(
        f"--- Step 1 ({config.backend}): Loading long-haul flights from "
        f"local scientific DB for {config.origin_airport} ---"
    )
    try:
        long_haul_flights = fetch_flights_step_1_scientific_db(config, client)
    except Exception as exc:
        raise RuntimeError(
            "Scientific DB Step 1 failed. Check that the local PostgreSQL "
            "database exists, is reachable, and has been built with "
            "opensky_build_scientific_db.py."
        ) from exc

    if long_haul_flights.empty:
        print(
            "No flights returned by the local scientific DB for that airport "
            "and time window."
        )
        raise SystemExit(0)

    print(
        f"Found {len(long_haul_flights)} flights with duration >= "
        f"{config.minimum_duration_hours} hours and a known arrival airport."
    )
    print(
        long_haul_flights[
            [
                "callsign",
                "icao24",
                "estDepartureAirport",
                "estArrivalAirport",
                "firstSeen",
                "lastSeen",
                "duration_hours",
                "flight_key",
                "source_backend",
            ]
        ].head()
    )
    save_step_1_results(long_haul_flights, config.paths.step_1_output_path)


def run_step_2_rest(
    config: RunConfig,
    client: REST,
) -> None:
    print("\n--- Step 2 (rest): Fetching trajectories from saved Step 1 results ---")

    long_haul_flights = load_step_1_results(config.paths.step_1_output_path)
    completed_flight_keys = load_completed_flight_keys(
        config.paths.step_2_output_path
    )
    pending_flights = long_haul_flights[
        ~long_haul_flights["flight_key"].astype(str).isin(completed_flight_keys)
    ].copy()

    print(
        f"Loaded {len(long_haul_flights)} saved flights. "
        f"{len(completed_flight_keys)} already have trajectory rows. "
        f"{len(pending_flights)} remain."
    )

    stopped_early = False
    total_pending = len(pending_flights)
    for position, (_, flight) in enumerate(pending_flights.iterrows(), start=1):
        print(
            f"[{position}/{total_pending}] Fetching track for flight "
            f"{flight['callsign']} heading to {flight['estArrivalAirport']}..."
        )

        try:
            track_df = fetch_track_chunk_rest(
                api=client,
                icao24=flight["icao24"],
                ts=flight["firstSeen"],
            )
        except Step2RateLimitError as exc:
            stopped_early = True
            print_step_2_checkpoint(
                current_flight=flight,
                completed_count=len(completed_flight_keys),
                remaining_count=total_pending - position + 1,
                response=exc.response,
                backend="rest",
            )
            break
        except httpx.HTTPStatusError as exc:
            print(
                f"Stopping on {flight['callsign']} ({flight['icao24']}): "
                f"REST track lookup failed with HTTP {exc.response.status_code}."
            )
            print(
                "Step 2 is resumable. Re-run later and it will continue "
                "from the remaining flights."
            )
            break
        except Exception as exc:
            print(
                f"Skipping {flight['callsign']} ({flight['icao24']}): "
                f"{exc}"
            )
            continue

        if track_df.empty:
            continue

        normalized_track_df = normalize_rest_track_dataframe(track_df, flight)
        append_tracks_to_output(
            normalized_track_df,
            config.paths.step_2_output_path,
        )
        completed_flight_keys.add(str(flight["flight_key"]))

        print(
            f"Saved {len(normalized_track_df)} trajectory rows for "
            f"{flight['callsign']} to {config.paths.step_2_output_path}"
        )
        time.sleep(TRACK_REQUEST_COOLDOWN_SECONDS)

    final_trajectory_df = load_step_2_results(config.paths.step_2_output_path)
    print_step_2_summary(final_trajectory_df, config, stopped_early)


def run_step_2_trino(
    config: RunConfig,
    client: Trino,
) -> None:
    print("\n--- Step 2 (trino): Fetching trajectories from saved Step 1 results ---")

    long_haul_flights = load_step_1_results(config.paths.step_1_output_path)
    completed_flight_keys = load_completed_flight_keys(
        config.paths.step_2_output_path
    )
    pending_flights = long_haul_flights[
        ~long_haul_flights["flight_key"].astype(str).isin(completed_flight_keys)
    ].copy()

    print(
        f"Loaded {len(long_haul_flights)} saved flights. "
        f"{len(completed_flight_keys)} already have trajectory rows. "
        f"{len(pending_flights)} remain."
    )

    total_pending = len(pending_flights)
    for position, (_, flight) in enumerate(pending_flights.iterrows(), start=1):
        print(
            f"[{position}/{total_pending}] Fetching Trino history for flight "
            f"{flight['callsign']} heading to {flight['estArrivalAirport']}..."
        )

        try:
            track_df = fetch_track_chunk_trino(client, flight)
        except Exception as exc:
            raise RuntimeError(
                "Trino Step 2 failed. Check that your Trino access is enabled "
                f"and your credentials are configured in {TRINO_CREDENTIALS_HINT}. "
                "You can retry later with --backend trino."
            ) from exc

        if track_df.empty:
            continue

        normalized_track_df = normalize_trino_track_dataframe(track_df, flight)
        append_tracks_to_output(
            normalized_track_df,
            config.paths.step_2_output_path,
        )
        completed_flight_keys.add(str(flight["flight_key"]))

        print(
            f"Saved {len(normalized_track_df)} trajectory rows for "
            f"{flight['callsign']} to {config.paths.step_2_output_path}"
        )

    final_trajectory_df = load_step_2_results(config.paths.step_2_output_path)
    print_step_2_summary(final_trajectory_df, config, stopped_early=False)


def run_step_2_scientific_db(
    config: RunConfig,
    client: ScientificDbClient,
) -> None:
    print(
        "\n--- Step 2 (scientific_db): Loading trajectories from saved Step 1 "
        "results via local PostgreSQL ---"
    )

    long_haul_flights = load_step_1_results(config.paths.step_1_output_path)
    completed_flight_keys = load_completed_flight_keys(
        config.paths.step_2_output_path
    )
    pending_flights = long_haul_flights[
        ~long_haul_flights["flight_key"].astype(str).isin(completed_flight_keys)
    ].copy()

    print(
        f"Loaded {len(long_haul_flights)} saved flights. "
        f"{len(completed_flight_keys)} already have trajectory rows. "
        f"{len(pending_flights)} remain."
    )

    if pending_flights.empty:
        final_trajectory_df = load_step_2_results(config.paths.step_2_output_path)
        print_step_2_summary(final_trajectory_df, config, stopped_early=False)
        return

    pending_keys = pending_flights["flight_key"].astype(str).tolist()
    total_batches = (
        len(pending_keys) + SCIENTIFIC_DB_STEP_2_BATCH_SIZE - 1
    ) // SCIENTIFIC_DB_STEP_2_BATCH_SIZE

    total_rows_saved = 0
    for batch_index, start in enumerate(
        range(0, len(pending_keys), SCIENTIFIC_DB_STEP_2_BATCH_SIZE),
        start=1,
    ):
        batch_keys = pending_keys[start : start + SCIENTIFIC_DB_STEP_2_BATCH_SIZE]
        print(
            f"[{batch_index}/{total_batches}] Loading trajectories for "
            f"{len(batch_keys)} flights from local scientific DB..."
        )
        try:
            track_df = fetch_track_batch_scientific_db(
                client.database_url,
                batch_keys,
            )
        except Exception as exc:
            raise RuntimeError(
                "Scientific DB Step 2 failed. Check that the local PostgreSQL "
                "database exists, is reachable, and contains trajectory rows."
            ) from exc

        if track_df.empty:
            continue

        append_tracks_to_output(track_df, config.paths.step_2_output_path)
        total_rows_saved += len(track_df)
        print(
            f"Saved {len(track_df)} trajectory rows for batch {batch_index} "
            f"to {config.paths.step_2_output_path}"
        )

    print(f"Saved {total_rows_saved} total trajectory rows from local scientific DB.")
    final_trajectory_df = load_step_2_results(config.paths.step_2_output_path)
    print_step_2_summary(final_trajectory_df, config, stopped_early=False)


def run_step_1(
    config: RunConfig,
    client: REST | Trino | ScientificDbClient,
) -> None:
    if config.backend == "scientific_db":
        assert isinstance(client, ScientificDbClient)
        run_step_1_scientific_db(config, client)
        return
    if config.backend == "rest":
        assert isinstance(client, REST)
        run_step_1_rest(config, client)
        return

    assert isinstance(client, Trino)
    run_step_1_trino(config, client)


def run_step_2(
    config: RunConfig,
    client: REST | Trino | ScientificDbClient,
) -> None:
    if config.backend == "scientific_db":
        assert isinstance(client, ScientificDbClient)
        run_step_2_scientific_db(config, client)
        return
    if config.backend == "rest":
        assert isinstance(client, REST)
        run_step_2_rest(config, client)
        return

    assert isinstance(client, Trino)
    run_step_2_trino(config, client)


def main() -> None:
    args = parse_args()
    resolved_backend = resolve_backend(
        args.backend,
        database_url=args.database_url,
        step=args.step,
    )
    args.backend = resolved_backend
    config = build_run_config(args)
    client = make_backend_client(config.backend, config.database_url)

    if config.step in {"1", "both"}:
        run_step_1(config, client)

    if config.step in {"2", "both"}:
        run_step_2(config, client)


if __name__ == "__main__":
    main()
