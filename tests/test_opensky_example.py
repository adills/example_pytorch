from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

import opensky_example


def make_rest_step_1_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firstSeen": pd.to_datetime(
                ["2026-01-01 10:00:00", "2026-01-01 11:00:00"], utc=True
            ),
            "lastSeen": pd.to_datetime(
                ["2026-01-01 17:30:00", "2026-01-01 15:00:00"], utc=True
            ),
            "icao24": ["abc123", "def456"],
            "callsign": ["LONG1  ", "MEDIUM2"],
            "estDepartureAirport": ["EGLL", "EGLL"],
            "estArrivalAirport": ["KJFK", "EDDF"],
        }
    )


def make_trino_step_1_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firstseen": pd.to_datetime(
                ["2026-01-01 08:00:00", "2026-01-01 09:00:00"], utc=True
            ),
            "lastseen": pd.to_datetime(
                ["2026-01-01 16:30:00", "2026-01-01 16:45:00"], utc=True
            ),
            "icao24": ["trn001", "trn002"],
            "callsign": ["TRINO1 ", "TRINO2 "],
            "departure": ["EGLL", "EGLL"],
            "arrival": ["KORD", "KLAX"],
        }
    )


def make_step_1_saved_frame() -> pd.DataFrame:
    first_seen = pd.to_datetime(
        ["2026-01-01 10:00:00", "2026-01-01 12:00:00"], utc=True
    )
    frame = pd.DataFrame(
        {
            "icao24": ["abc123", "def456"],
            "callsign": ["LONG1", "LONG2"],
            "estDepartureAirport": ["EGLL", "EGLL"],
            "estArrivalAirport": ["KJFK", "KLAX"],
            "firstSeen": first_seen,
            "lastSeen": pd.to_datetime(
                ["2026-01-01 17:00:00", "2026-01-01 20:00:00"], utc=True
            ),
            "duration_hours": [7.0, 8.0],
            "source_backend": ["rest", "rest"],
        }
    )
    frame["flight_key"] = frame.apply(
        lambda row: opensky_example.build_flight_key(
            row["icao24"],
            row["callsign"],
            row["firstSeen"],
        ),
        axis=1,
    )
    return frame


def make_rest_track_frame(
    icao24: str,
    callsign: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 10:00:00", "2026-01-01 10:30:00"], utc=True
            ),
            "latitude": [51.4, 52.0],
            "longitude": [-0.4, -5.0],
            "altitude": [0.0, 5000.0],
            "track": [260.0, 265.0],
            "onground": [True, False],
            "icao24": [icao24, icao24],
            "callsign": [callsign, callsign],
        }
    )


def make_trino_track_frame(
    icao24: str,
    callsign: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01 10:00:00", "2026-01-01 10:10:00"], utc=True
            ),
            "icao24": [icao24, icao24],
            "lat": [51.4, 51.8],
            "lon": [-0.4, -1.2],
            "velocity": [0.0, 220.0],
            "heading": [250.0, 255.0],
            "vertrate": [0.0, 10.0],
            "callsign": [callsign, callsign],
            "onground": [True, False],
            "baroaltitude": [0.0, 3500.0],
            "geoaltitude": [0.0, 3600.0],
        }
    )


def make_scientific_step_1_frame() -> pd.DataFrame:
    first_seen = pd.to_datetime(
        ["2026-01-01 10:00:00", "2026-01-01 12:00:00"], utc=True
    )
    frame = pd.DataFrame(
        {
            "icao24": ["sci123", "sci456"],
            "callsign": ["SCI1", "SCI2"],
            "estDepartureAirport": ["EGLL", "EGLL"],
            "estArrivalAirport": ["KJFK", "KLAX"],
            "firstSeen": first_seen,
            "lastSeen": pd.to_datetime(
                ["2026-01-01 18:00:00", "2026-01-01 21:00:00"], utc=True
            ),
            "duration_hours": [8.0, 9.0],
            "flight_key": [
                opensky_example.build_flight_key("sci123", "SCI1", first_seen[0]),
                opensky_example.build_flight_key("sci456", "SCI2", first_seen[1]),
            ],
            "source_backend": ["scientific_db", "scientific_db"],
        }
    )
    return frame


def make_scientific_track_frame(flight_key: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01 10:00:00", "2026-01-01 10:10:00"], utc=True
            ),
            "icao24": ["sci123", "sci123"],
            "lat": [51.4, 51.8],
            "lon": [-0.4, -1.2],
            "velocity": [0.0, 220.0],
            "heading": [250.0, 255.0],
            "vertrate": [0.0, 10.0],
            "callsign": ["SCI1", "SCI1"],
            "onground": [True, False],
            "alert": [pd.NA, pd.NA],
            "spi": [pd.NA, pd.NA],
            "squawk": [pd.NA, pd.NA],
            "baroaltitude": [0.0, 3500.0],
            "geoaltitude": [0.0, 3600.0],
            "lastposupdate": [pd.NA, pd.NA],
            "lastcontact": [pd.NA, pd.NA],
            "serials": [pd.NA, pd.NA],
            "hour": pd.to_datetime(
                ["2026-01-01 10:00:00", "2026-01-01 10:00:00"], utc=True
            ),
            "flight_key": [flight_key, flight_key],
            "origin": ["EGLL", "EGLL"],
            "destination": ["KJFK", "KJFK"],
            "first_seen": pd.to_datetime(
                ["2026-01-01 10:00:00", "2026-01-01 10:00:00"], utc=True
            ),
            "last_seen": pd.to_datetime(
                ["2026-01-01 18:00:00", "2026-01-01 18:00:00"], utc=True
            ),
            "duration_hours": [8.0, 8.0],
            "source_backend": ["scientific_db", "scientific_db"],
        }
    )
    return opensky_example.standardize_step_2_schema(frame)


class FakeRestClient:
    def __init__(self, departures_df: pd.DataFrame):
        self.departures_df = departures_df
        self.client = mock.Mock()
        self.headers: dict[str, str] = {}

    def departure(
        self,
        airport: str,
        begin: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        return self.departures_df.copy()


class FakeTrinoClient:
    def __init__(self, flights_df: pd.DataFrame | None = None):
        self.flights_df = flights_df

    def flightlist(
        self,
        start: str,
        stop: str,
        departure_airport: str,
    ) -> pd.DataFrame | None:
        if self.flights_df is None:
            return None
        return self.flights_df.copy()


class OpenSkyExampleTestCase(unittest.TestCase):
    def make_paths(self, tmpdir: str) -> opensky_example.OutputPaths:
        base = Path(tmpdir)
        return opensky_example.OutputPaths(
            step_1_output_path=base / "step1.csv",
            step_2_output_path=base / "step2.csv",
            plot_output_path=base / "plot.png",
        )

    def make_config(
        self,
        tmpdir: str,
        backend: opensky_example.ResolvedBackendName = "rest",
        step: opensky_example.StepName = "1",
        save_step_1_csv: bool | None = None,
        save_step_2_csv: bool | None = None,
        plot_min_observed_fraction: float = opensky_example.MIN_PLOT_OBSERVED_HOURS_FRACTION,
    ) -> opensky_example.RunConfig:
        if save_step_1_csv is None:
            save_step_1_csv = backend != "scientific_db"
        if save_step_2_csv is None:
            save_step_2_csv = backend != "scientific_db"
        return opensky_example.RunConfig(
            backend=backend,
            step=step,
            origin_airport="EGLL",
            start_time="2026-01-01 00:00:00",
            end_time="2026-01-01 23:59:59",
            minimum_duration_hours=6,
            database_url="postgresql+psycopg://tester@localhost/opensky_scientific",
            save_step_1_csv=save_step_1_csv,
            save_step_2_csv=save_step_2_csv,
            plot_min_observed_fraction=plot_min_observed_fraction,
            paths=self.make_paths(tmpdir),
        )

    def test_parse_args_defaults_to_auto_backend(self) -> None:
        with mock.patch("sys.argv", ["opensky_example.py"]):
            args = opensky_example.parse_args()

        self.assertEqual(args.backend, "auto")
        self.assertEqual(args.step, "1")
        self.assertEqual(args.origin_airport, opensky_example.DEFAULT_ORIGIN_AIRPORT)
        self.assertIsNone(args.start_time)
        self.assertIsNone(args.end_time)
        self.assertFalse(args.save_step_1_csv)
        self.assertFalse(args.save_step_2_csv)
        self.assertEqual(
            args.plot_min_observed_fraction,
            opensky_example.MIN_PLOT_OBSERVED_HOURS_FRACTION,
        )
        self.assertEqual(
            args.minimum_duration_hours,
            opensky_example.DEFAULT_MINIMUM_DURATION_HOURS,
        )

    def test_resolve_backend_auto_prefers_scientific_db(self) -> None:
        with mock.patch.object(
            opensky_example,
            "scientific_db_is_available",
            return_value=True,
        ):
            backend = opensky_example.resolve_backend(
                "auto",
                database_url="postgresql+psycopg://tester@localhost/opensky_scientific",
                step="1",
            )

        self.assertEqual(backend, "scientific_db")

    def test_resolve_time_window_uses_scientific_db_flight_window_defaults(self) -> None:
        with mock.patch.object(
            opensky_example,
            "get_scientific_db_flight_window",
            return_value=(
                pd.Timestamp("2018-12-31 08:48:25", tz="UTC"),
                pd.Timestamp("2022-12-31 23:58:29", tz="UTC"),
            ),
        ):
            start_time, end_time = opensky_example.resolve_time_window(
                None,
                None,
                backend="scientific_db",
                database_url="postgresql+psycopg://tester@localhost/opensky_scientific",
                origin_airport="EGLL",
            )

        self.assertEqual(start_time, "2018-12-31 08:48:25")
        self.assertEqual(end_time, "2022-12-31 23:58:29")

    def test_run_step_1_rest_writes_step_1_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir, backend="rest", step="1")
            client = FakeRestClient(make_rest_step_1_frame())

            opensky_example.run_step_1_rest(config, client)

            saved_df = pd.read_csv(config.paths.step_1_output_path)
            self.assertEqual(len(saved_df), 1)
            self.assertEqual(saved_df.loc[0, "callsign"], "LONG1")
            self.assertEqual(saved_df.loc[0, "source_backend"], "rest")
            self.assertIn("flight_key", saved_df.columns)

    def test_run_step_1_trino_writes_step_1_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir, backend="trino", step="1")
            client = FakeTrinoClient(make_trino_step_1_frame())

            opensky_example.run_step_1_trino(config, client)

            saved_df = pd.read_csv(config.paths.step_1_output_path)
            self.assertEqual(len(saved_df), 2)
            self.assertIn("estDepartureAirport", saved_df.columns)
            self.assertIn("estArrivalAirport", saved_df.columns)
            self.assertTrue((saved_df["source_backend"] == "trino").all())

    def test_run_step_1_scientific_db_skips_csv_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir, backend="scientific_db", step="1")
            client = opensky_example.ScientificDbClient(config.database_url)

            with mock.patch.object(
                opensky_example,
                "fetch_flights_step_1_scientific_db",
                return_value=make_scientific_step_1_frame(),
            ):
                opensky_example.run_step_1_scientific_db(config, client)

            self.assertFalse(config.paths.step_1_output_path.exists())

    def test_run_step_1_scientific_db_can_write_step_1_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(
                tmpdir,
                backend="scientific_db",
                step="1",
                save_step_1_csv=True,
            )
            client = opensky_example.ScientificDbClient(config.database_url)

            with mock.patch.object(
                opensky_example,
                "fetch_flights_step_1_scientific_db",
                return_value=make_scientific_step_1_frame(),
            ):
                opensky_example.run_step_1_scientific_db(config, client)

            saved_df = pd.read_csv(config.paths.step_1_output_path)
            self.assertEqual(len(saved_df), 2)
            self.assertTrue((saved_df["source_backend"] == "scientific_db").all())

    def test_run_step_2_rest_resume_skips_completed_flights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir, backend="rest", step="2")
            step_1_df = make_step_1_saved_frame()
            step_1_df.to_csv(config.paths.step_1_output_path, index=False)

            with mock.patch.object(
                opensky_example,
                "fetch_track_chunk_rest",
                side_effect=[
                    make_rest_track_frame("abc123", "LONG1"),
                    make_rest_track_frame("def456", "LONG2"),
                ],
            ) as fetch_mock, mock.patch.object(
                opensky_example,
                "plot_altitude_vs_time_since_takeoff",
            ), mock.patch.object(opensky_example.time, "sleep"):
                opensky_example.run_step_2_rest(config, FakeRestClient(pd.DataFrame()))
                self.assertEqual(fetch_mock.call_count, 2)

            saved_once = pd.read_csv(config.paths.step_2_output_path)
            self.assertEqual(saved_once["flight_key"].nunique(), 2)

            with mock.patch.object(
                opensky_example,
                "fetch_track_chunk_rest",
            ) as fetch_mock, mock.patch.object(
                opensky_example,
                "plot_altitude_vs_time_since_takeoff",
            ), mock.patch.object(opensky_example.time, "sleep"):
                opensky_example.run_step_2_rest(config, FakeRestClient(pd.DataFrame()))
                self.assertEqual(fetch_mock.call_count, 0)

    def test_run_step_2_trino_writes_step_2_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir, backend="trino", step="2")
            step_1_df = make_step_1_saved_frame().head(1)
            step_1_df.to_csv(config.paths.step_1_output_path, index=False)

            with mock.patch.object(
                opensky_example,
                "fetch_track_chunk_trino",
                return_value=make_trino_track_frame("abc123", "LONG1"),
            ), mock.patch.object(
                opensky_example,
                "plot_altitude_vs_time_since_takeoff",
            ):
                opensky_example.run_step_2_trino(config, FakeTrinoClient())

            saved_df = pd.read_csv(config.paths.step_2_output_path)
            self.assertEqual(len(saved_df), 2)
            self.assertTrue((saved_df["source_backend"] == "trino").all())
            self.assertIn("velocity", saved_df.columns)
            self.assertIn("heading", saved_df.columns)
            self.assertIn("geoaltitude", saved_df.columns)

    def test_run_step_2_scientific_db_uses_in_memory_results_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir, backend="scientific_db", step="2")
            client = opensky_example.ScientificDbClient(config.database_url)

            with mock.patch.object(
                opensky_example,
                "fetch_trajectories_step_2_scientific_db",
                return_value=make_scientific_track_frame(
                    make_scientific_step_1_frame().iloc[0]["flight_key"]
                ),
            ), mock.patch.object(
                opensky_example,
                "plot_altitude_vs_time_since_takeoff",
            ):
                opensky_example.run_step_2_scientific_db(config, client)

            self.assertFalse(config.paths.step_2_output_path.exists())

    def test_run_step_2_scientific_db_can_write_step_2_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(
                tmpdir,
                backend="scientific_db",
                step="2",
                save_step_2_csv=True,
            )
            client = opensky_example.ScientificDbClient(config.database_url)

            with mock.patch.object(
                opensky_example,
                "fetch_trajectories_step_2_scientific_db",
                return_value=make_scientific_track_frame(
                    make_scientific_step_1_frame().iloc[0]["flight_key"]
                ),
            ), mock.patch.object(
                opensky_example,
                "plot_altitude_vs_time_since_takeoff",
            ):
                opensky_example.run_step_2_scientific_db(config, client)

            saved_df = pd.read_csv(config.paths.step_2_output_path)
            self.assertEqual(len(saved_df), 2)
            self.assertTrue((saved_df["source_backend"] == "scientific_db").all())

    def test_select_altitude_for_plot_prefers_geoaltitude(self) -> None:
        frame = pd.DataFrame(
            {
                "geoaltitude": [1000.0, pd.NA, 3000.0],
                "baroaltitude": [900.0, 2000.0, 2900.0],
            }
        )

        result = opensky_example.select_altitude_for_plot(frame)

        self.assertEqual(list(result), [1000.0, 2000.0, 3000.0])

    def test_prepare_flight_for_altitude_plot_uses_first_seen_and_drops_onground(self) -> None:
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-01-01 11:00:00",
                        "2026-01-01 12:00:00",
                        "2026-01-01 13:00:00",
                    ],
                    utc=True,
                ),
                "first_seen": pd.to_datetime(
                    [
                        "2026-01-01 10:00:00",
                        "2026-01-01 10:00:00",
                        "2026-01-01 10:00:00",
                    ],
                    utc=True,
                ),
                "duration_hours": [8.0, 8.0, 8.0],
                "onground": [True, False, False],
                "geoaltitude": [0.0, 1000.0, 2000.0],
                "baroaltitude": [0.0, 900.0, 1900.0],
            }
        )

        result = opensky_example.prepare_flight_for_altitude_plot(
            frame,
            minimum_observed_hours_fraction=0.1,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(list(result["hours_since_takeoff"]), [2.0, 3.0])

    def test_prepare_flight_for_altitude_plot_requires_minimum_observed_fraction(self) -> None:
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-01-01 10:00:00",
                        "2026-01-01 10:30:00",
                    ],
                    utc=True,
                ),
                "first_seen": pd.to_datetime(
                    [
                        "2026-01-01 10:00:00",
                        "2026-01-01 10:00:00",
                    ],
                    utc=True,
                ),
                "duration_hours": [8.0, 8.0],
                "onground": [False, False],
                "geoaltitude": [1000.0, 2000.0],
                "baroaltitude": [900.0, 1900.0],
            }
        )

        result = opensky_example.prepare_flight_for_altitude_plot(
            frame,
            minimum_observed_hours_fraction=0.25,
        )

        self.assertTrue(result.empty)

    def test_normalize_rest_track_dataframe_maps_into_trino_schema(self) -> None:
        flight = make_step_1_saved_frame().iloc[0]

        normalized_df = opensky_example.normalize_rest_track_dataframe(
            make_rest_track_frame("abc123", "LONG1"),
            flight,
        )

        self.assertEqual(
            list(normalized_df.columns),
            opensky_example.STEP_2_SCHEMA_COLUMNS,
        )
        self.assertIn("baroaltitude", normalized_df.columns)
        self.assertIn("geoaltitude", normalized_df.columns)
        self.assertIn("heading", normalized_df.columns)
        self.assertTrue(normalized_df["geoaltitude"].isna().all())
        self.assertTrue(normalized_df["baroaltitude"].notna().all())

    def test_rest_soak_workflow_saves_all_pending_flights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir, backend="rest", step="2")
            flights = []
            for index in range(12):
                first_seen = pd.Timestamp(
                    f"2026-01-{index + 1:02d} 10:00:00",
                    tz="UTC",
                )
                flights.append(
                    {
                        "icao24": f"rest{index:03d}",
                        "callsign": f"REST{index:03d}",
                        "estDepartureAirport": "EGLL",
                        "estArrivalAirport": "KJFK",
                        "firstSeen": first_seen,
                        "lastSeen": first_seen + pd.Timedelta(hours=7),
                        "duration_hours": 7.0,
                        "source_backend": "rest",
                        "flight_key": opensky_example.build_flight_key(
                            f"rest{index:03d}",
                            f"REST{index:03d}",
                            first_seen,
                        ),
                    }
                )

            pd.DataFrame(flights).to_csv(config.paths.step_1_output_path, index=False)

            def fake_fetch(api: object, icao24: str, ts: object) -> pd.DataFrame:
                return make_rest_track_frame(icao24, f"{icao24.upper()}")

            with mock.patch.object(
                opensky_example,
                "fetch_track_chunk_rest",
                side_effect=fake_fetch,
            ), mock.patch.object(
                opensky_example,
                "plot_altitude_vs_time_since_takeoff",
            ), mock.patch.object(opensky_example.time, "sleep"):
                opensky_example.run_step_2_rest(config, FakeRestClient(pd.DataFrame()))

            saved_df = pd.read_csv(config.paths.step_2_output_path)
            self.assertEqual(saved_df["flight_key"].nunique(), 12)

    def test_trino_soak_workflow_saves_all_pending_flights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.make_config(tmpdir, backend="trino", step="2")
            flights = []
            for index in range(10):
                first_seen = pd.Timestamp(
                    f"2026-02-{index + 1:02d} 08:00:00",
                    tz="UTC",
                )
                flights.append(
                    {
                        "icao24": f"trino{index:03d}",
                        "callsign": f"TRI{index:03d}",
                        "estDepartureAirport": "EGLL",
                        "estArrivalAirport": "KLAX",
                        "firstSeen": first_seen,
                        "lastSeen": first_seen + pd.Timedelta(hours=8),
                        "duration_hours": 8.0,
                        "source_backend": "trino",
                        "flight_key": opensky_example.build_flight_key(
                            f"trino{index:03d}",
                            f"TRI{index:03d}",
                            first_seen,
                        ),
                    }
                )

            pd.DataFrame(flights).to_csv(config.paths.step_1_output_path, index=False)

            def fake_fetch(client: object, flight: pd.Series) -> pd.DataFrame:
                return make_trino_track_frame(
                    str(flight["icao24"]),
                    str(flight["callsign"]),
                )

            with mock.patch.object(
                opensky_example,
                "fetch_track_chunk_trino",
                side_effect=fake_fetch,
            ), mock.patch.object(
                opensky_example,
                "plot_altitude_vs_time_since_takeoff",
            ):
                opensky_example.run_step_2_trino(config, FakeTrinoClient())

            saved_df = pd.read_csv(config.paths.step_2_output_path)
            self.assertEqual(saved_df["flight_key"].nunique(), 10)


if __name__ == "__main__":
    unittest.main()
