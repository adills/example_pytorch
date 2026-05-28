import httpx
import pandas as pd
from pyopensky.rest import REST
from pathlib import Path
import sys
import time
import matplotlib

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


def plot_altitude_vs_time_since_takeoff(trajectory_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    for (icao24, callsign, destination), flight_df in trajectory_df.groupby(
        ["icao24", "callsign", "destination"], dropna=False
    ):
        flight_df = flight_df.sort_values("time").copy()
        first_time = flight_df["time"].iloc[0]
        flight_df["hours_since_takeoff"] = (
            pd.to_datetime(flight_df["time"]) - pd.to_datetime(first_time)
        ).dt.total_seconds() / 3600

        label = f"{callsign.strip()} -> {destination}"
        plt.plot(
            flight_df["hours_since_takeoff"],
            flight_df["altitude_m"],
            label=label,
        )

    plt.xlabel("Hours Since Takeoff")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude vs Time Since Takeoff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if interactive_backend:
        plt.savefig("flights_altitude_vs_time.png", dpi=150)
        plt.show()
    else:
        plt.savefig("flights_altitude_vs_time.png", dpi=150)


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

# Define parameters
origin_airport = "EGLL"  # London Heathrow
start_time = "2026-01-24 00:00:00"
end_time = "2026-05-24 23:59:59"
minimum_duration_hours = 6

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
    flights_df = flights_df.assign(
        duration_hours=(
            flights_df["lastSeen"] - flights_df["firstSeen"]
        ).dt.total_seconds()
        / 3600
    )
    long_haul_flights = flights_df[
        flights_df["duration_hours"] >= minimum_duration_hours
    ].copy()
except KeyError as exc:
    print("Expected flight timing columns were not returned by the REST API.")
    print(f"Error: {exc}")
    raise SystemExit(1)

print(
    f"Found {len(long_haul_flights)} flights with duration >= "
    f"{minimum_duration_hours} hours."
)
if long_haul_flights.empty:
    raise SystemExit(0)

long_haul_flights = long_haul_flights[
    long_haul_flights["estArrivalAirport"].notna()
].copy()

print(
    f"Keeping {len(long_haul_flights)} flights with a known arrival airport."
)
if long_haul_flights.empty:
    print("No long-duration flights had a populated arrival airport.")
    raise SystemExit(0)

print(
    long_haul_flights[
        [
            "callsign",
            "icao24",
            "estArrivalAirport",
            "firstSeen",
            "lastSeen",
            "duration_hours",
        ]
    ].head()
)

print("\n--- Step 2: Fetching trajectories with REST tracks endpoint ---")
all_tracks = []

# Take up to 3 flights to keep the example light
for _, flight in long_haul_flights.head(3).iterrows():
    print(
        "Fetching track for flight "
        f"{flight['callsign']} heading to {flight['estArrivalAirport']}..."
    )

    try:
        track_df = api.tracks(
            icao24=flight["icao24"],
            ts=flight["firstSeen"],
        )
    except httpx.HTTPStatusError as exc:
        print(
            f"Skipping {flight['callsign']} ({flight['icao24']}): "
            f"REST track lookup failed with HTTP {exc.response.status_code}."
        )
        continue
    except Exception as exc:
        print(
            f"Skipping {flight['callsign']} ({flight['icao24']}): "
            f"{exc}"
        )
        continue

    if track_df.empty:
        continue

    normalized_track_df = track_df.rename(
        columns={
            "timestamp": "time",
            "latitude": "lat",
            "longitude": "lon",
            "altitude": "altitude_m",
            "track": "true_track_deg",
        }
    ).assign(
        origin=flight["estDepartureAirport"],
        destination=flight["estArrivalAirport"],
        first_seen=flight["firstSeen"],
        last_seen=flight["lastSeen"],
        duration_hours=flight["duration_hours"],
    )

    all_tracks.append(normalized_track_df)

if all_tracks:
    final_trajectory_df = pd.concat(all_tracks, ignore_index=True)
    print("\n--- Success! Sample trajectory view: ---")
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
    print(final_trajectory_df[columns_to_show].head())

    # Save to CSV for mapping tools or notebooks
    plot_altitude_vs_time_since_takeoff(final_trajectory_df)
    final_trajectory_df.to_csv(Path(__file__).parent / "long_distance_trajectories.csv", index=False)
else:
    print(
        "No tracks were returned. REST trajectories are lower resolution than "
        "Trino history and may be missing for some flights."
    )
