import argparse
from pathlib import Path
import sys
import time

import httpx
import matplotlib
import pandas as pd
from pyopensky.rest import REST


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
STEP_1_OUTPUT_PATH = SCRIPT_DIR / "long_haul_flights.csv"
STEP_2_OUTPUT_PATH = SCRIPT_DIR / "long_distance_trajectories.csv"
PLOT_OUTPUT_PATH = SCRIPT_DIR / "flights_altitude_vs_time.png"

# STEP 1 metadata is saved to long_haul_flights.csv:
# icao24
# callsign
# estDepartureAirport
# estArrivalAirport
# firstSeen
# lastSeen
# duration_hours
# flight_key

# STEP 2 This step is time-consuming and should be run separately after Step 1 
# completes.
# time
# lat
# lon
# altitude_m
# true_track_deg

# OpenSky REST client.
# Authentication is optional for public endpoints, but authenticated access
# raises rate limits. pyopensky reads credentials from:
# - OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET for bearer-token auth
# - OPENSKY_USERNAME / OPENSKY_PASSWORD are still useful for your account,
#   but this REST client primarily uses client credentials when available.
api = REST()
REST_INTERVAL_LIMIT = pd.Timedelta(days=2)
REST_CHUNK_RETRY_ATTEMPTS = 3
REST_CHUNK_RETRY_SLEEP_SECONDS = 2
REST_TRACK_RETRY_ATTEMPTS = 3
REST_TRACK_RETRY_SLEEP_SECONDS = 2
REST_TRACK_MAX_RETRY_AFTER_SECONDS = 30
TRACK_REQUEST_COOLDOWN_SECONDS = 5
STEP_2_DEFAULT_RETRY_WAIT_SECONDS = 3600

DEFAULT_ORIGIN_AIRPORT = "EGLL"  # London Heathrow
DEFAULT_START_TIME = "2026-01-24 00:00:00"
DEFAULT_END_TIME = "2026-05-24 23:59:59"
DEFAULT_MINIMUM_DURATION_HOURS = 6


def build_flight_key(
    icao24: str,
    callsign: str,
    first_seen: str | pd.Timestamp,
) -> str:
    callsign_clean = str(callsign).strip()
    first_seen_ts = pd.Timestamp(first_seen)
    return f"{icao24}|{callsign_clean}|{first_seen_ts.isoformat()}"


def plot_altitude_vs_time_since_takeoff(trajectory_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for (_, callsign, destination), flight_df in trajectory_df.groupby(
        ["flight_key", "callsign", "destination"], dropna=False
    ):
        flight_df = flight_df.sort_values("time").copy()
        first_time = flight_df["time"].iloc[0]
        flight_df["hours_since_takeoff"] = (
            pd.to_datetime(flight_df["time"]) - pd.to_datetime(first_time)
        ).dt.total_seconds() / 3600

        label = f"{str(callsign).strip()} -> {destination}"
        ax.plot(
            flight_df["hours_since_takeoff"],
            flight_df["altitude_m"],
            label=label,
        )

    ax.set_xlabel("Hours Since Takeoff")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude vs Time Since Takeoff")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_OUTPUT_PATH}")
    if interactive_backend:
        plt.show()
    plt.close(fig)


def fetch_departures_for_interval(
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
        return fetch_departure_chunk(
            api=api,
            airport=airport,
            begin=begin_ts,
            end=end_ts,
        )

    print(
        f"Requested interval is {interval}. "
        f"Chunking into {REST_INTERVAL_LIMIT} REST requests..."
    )

    all_chunks: list[pd.DataFrame] = []
    current_start = begin_ts

    while current_start < end_ts:
        current_end = min(current_start + REST_INTERVAL_LIMIT, end_ts)
        print(f"Fetching departures from {current_start} to {current_end}...")

        chunk_df = fetch_departure_chunk(
            api=api,
            airport=airport,
            begin=current_start,
            end=current_end,
        )

        if not chunk_df.empty:
            all_chunks.append(chunk_df)

        current_start = current_end

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


def fetch_departure_chunk(
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


def fetch_track_chunk(
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

        if status_code in (429, 503):
            if attempt == REST_TRACK_RETRY_ATTEMPTS:
                response.raise_for_status()

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

    long_haul_flights["callsign"] = long_haul_flights["callsign"].astype(str).str.strip()
    long_haul_flights["flight_key"] = long_haul_flights.apply(
        lambda row: build_flight_key(
            row["icao24"],
            row["callsign"],
            row["firstSeen"],
        ),
        axis=1,
    )
    return long_haul_flights.reset_index(drop=True)


def save_step_1_results(long_haul_flights: pd.DataFrame) -> None:
    long_haul_flights.to_csv(STEP_1_OUTPUT_PATH, index=False)
    print(
        f"Saved {len(long_haul_flights)} long-haul flight records to "
        f"{STEP_1_OUTPUT_PATH}"
    )


def load_step_1_results() -> pd.DataFrame:
    if not STEP_1_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Step 1 output not found: {STEP_1_OUTPUT_PATH}. Run Step 1 first."
        )

    return pd.read_csv(
        STEP_1_OUTPUT_PATH,
        parse_dates=["firstSeen", "lastSeen"],
    )


def normalize_track_dataframe(
    track_df: pd.DataFrame,
    flight: pd.Series,
) -> pd.DataFrame:
    return track_df.rename(
        columns={
            "timestamp": "time",
            "latitude": "lat",
            "longitude": "lon",
            "altitude": "altitude_m",
            "track": "true_track_deg",
        }
    ).assign(
        flight_key=flight["flight_key"],
        origin=flight["estDepartureAirport"],
        destination=flight["estArrivalAirport"],
        first_seen=flight["firstSeen"],
        last_seen=flight["lastSeen"],
        duration_hours=flight["duration_hours"],
    )


def load_completed_flight_keys() -> set[str]:
    if not STEP_2_OUTPUT_PATH.exists():
        return set()

    completed_df = pd.read_csv(STEP_2_OUTPUT_PATH, usecols=["flight_key"])
    return set(completed_df["flight_key"].dropna().astype(str).unique())


def append_tracks_to_output(track_df: pd.DataFrame) -> None:
    write_header = not STEP_2_OUTPUT_PATH.exists()
    track_df.to_csv(
        STEP_2_OUTPUT_PATH,
        mode="a",
        header=write_header,
        index=False,
    )


def load_step_2_results() -> pd.DataFrame:
    if not STEP_2_OUTPUT_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        STEP_2_OUTPUT_PATH,
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
) -> None:
    retry_timestamp_utc, retry_reason = estimate_step_2_retry_time(response)
    retry_command = "python opensky_example.py --step 2"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch long-haul flight metadata and trajectories from OpenSky REST."
    )
    parser.add_argument(
        "--step",
        choices=["1", "2", "both"],
        default="1",
        help="Which workflow step to run: 1, 2, or both.",
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


def run_step_1(
    origin_airport: str,
    start_time: str,
    end_time: str,
    minimum_duration_hours: float,
) -> None:
    print(f"--- Step 1: Fetching departed flights from {origin_airport} via REST ---")

    try:
        flights_df = fetch_departures_for_interval(
            api=api,
            airport=origin_airport,
            begin=start_time,
            end=end_time,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            print("No flights found for the specified timeframe.")
        else:
            print(f"Error fetching departures: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"Unexpected error fetching departures: {exc}")
        raise SystemExit(1)

    if flights_df.empty:
        print("No flights returned by the REST API for that airport and time window.")
        raise SystemExit(0)

    try:
        long_haul_flights = prepare_long_haul_flights(
            flights_df,
            minimum_duration_hours=minimum_duration_hours,
        )
    except KeyError as exc:
        print("Expected flight timing columns were not returned by the REST API.")
        print(f"Error: {exc}")
        raise SystemExit(1)

    print(
        f"Found {len(long_haul_flights)} flights with duration >= "
        f"{minimum_duration_hours} hours and a known arrival airport."
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
            ]
        ].head()
    )
    save_step_1_results(long_haul_flights)


def run_step_2() -> None:
    print("\n--- Step 2: Fetching trajectories from saved Step 1 results ---")

    try:
        long_haul_flights = load_step_1_results()
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1)

    completed_flight_keys = load_completed_flight_keys()
    pending_flights = long_haul_flights[
        ~long_haul_flights["flight_key"].astype(str).isin(completed_flight_keys)
    ].copy()

    print(
        f"Loaded {len(long_haul_flights)} saved flights. "
        f"{len(completed_flight_keys)} already have trajectory rows. "
        f"{len(pending_flights)} remain."
    )

    step_2_stopped_early = False
    for position, (_, flight) in enumerate(pending_flights.iterrows(), start=1):
        print(
            "Fetching track for flight "
            f"{flight['callsign']} heading to {flight['estArrivalAirport']}..."
        )

        try:
            track_df = fetch_track_chunk(
                api=api,
                icao24=flight["icao24"],
                ts=flight["firstSeen"],
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                step_2_stopped_early = True
                print_step_2_checkpoint(
                    current_flight=flight,
                    completed_count=len(completed_flight_keys),
                    remaining_count=len(pending_flights) - position + 1,
                    response=exc.response,
                )
            else:
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

        normalized_track_df = normalize_track_dataframe(track_df, flight)
        append_tracks_to_output(normalized_track_df)
        completed_flight_keys.add(str(flight["flight_key"]))

        print(
            f"Saved {len(normalized_track_df)} trajectory rows for "
            f"{flight['callsign']} to {STEP_2_OUTPUT_PATH}"
        )
        time.sleep(TRACK_REQUEST_COOLDOWN_SECONDS)

    final_trajectory_df = load_step_2_results()
    if final_trajectory_df.empty:
        print("No trajectory rows have been saved yet.")
        return

    columns_to_show = [
        "time",
        "icao24",
        "callsign",
        "lat",
        "lon",
        "altitude_m",
        "true_track_deg",
        "origin",
        "destination",
    ]
    print("\n--- Trajectory Sample ---")
    print(final_trajectory_df[columns_to_show].head())
    plot_altitude_vs_time_since_takeoff(final_trajectory_df)
    print(f"Trajectory output is stored at {STEP_2_OUTPUT_PATH}")
    if step_2_stopped_early:
        print("Step 2 paused at a rate-limit checkpoint. Resume with:")
        print("python opensky_example.py --step 2")


def main() -> None:
    args = parse_args()

    if args.step in {"1", "both"}:
        run_step_1(
            origin_airport=args.origin_airport,
            start_time=args.start_time,
            end_time=args.end_time,
            minimum_duration_hours=args.minimum_duration_hours,
        )

    if args.step in {"2", "both"}:
        run_step_2()


if __name__ == "__main__":
    main()
