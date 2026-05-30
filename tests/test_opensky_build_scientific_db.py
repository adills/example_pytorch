import unittest

import pandas as pd

import opensky_build_scientific_db as scientific_db


class OpenSkyBuildScientificDbTestCase(unittest.TestCase):
    def test_validate_required_columns_raises_on_missing_columns(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            scientific_db.validate_required_columns(
                ["callsign", "icao24"],
                scientific_db.REQUIRED_COVID_COLUMNS,
                source_label="COVID test file",
            )

        self.assertIn("missing required columns", str(context.exception))
        self.assertIn("COVID test file", str(context.exception))

    def test_build_parser_defaults_to_local_opensky_scientific_database(self) -> None:
        parser = scientific_db.build_parser()

        build_args = parser.parse_args(
            ["build", "--download-dir", "/tmp/opensky"]
        )
        query_args = parser.parse_args(["query"])
        reset_args = parser.parse_args(["reset-build"])
        inspect_args = parser.parse_args(["inspect-covid", "--download-dir", "/tmp/opensky"])

        self.assertIn("opensky_scientific", build_args.database_url)
        self.assertIn("@localhost/", build_args.database_url)
        self.assertEqual(build_args.database_url, scientific_db.DEFAULT_DATABASE_URL)
        self.assertEqual(query_args.database_url, scientific_db.DEFAULT_DATABASE_URL)
        self.assertEqual(reset_args.database_url, scientific_db.DEFAULT_DATABASE_URL)
        self.assertFalse(reset_args.confirm_reset)
        self.assertEqual(inspect_args.max_covid_files, 2)
        self.assertEqual(inspect_args.max_covid_chunks_per_file, 1)
        self.assertEqual(inspect_args.top_n_airports, 20)
        self.assertEqual(
            build_args.origin_airports,
            ",".join(scientific_db.DEFAULT_ORIGIN_AIRPORTS),
        )

    def test_reset_scientific_db_requires_confirmation(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            scientific_db.reset_scientific_db(
                scientific_db.DEFAULT_DATABASE_URL,
                confirm=False,
            )

        self.assertIn("confirm-reset", str(context.exception))

    def test_filter_covid_chunk_keeps_only_selected_origins_and_long_flights(self) -> None:
        chunk = pd.DataFrame(
            {
                "callsign": ["BAW001 ", "SIA305", "AFR123"],
                "number": ["BA1", "SQ305", "AF123"],
                "icao24": ["abc123", "def456", "ghi789"],
                "registration": ["G-TEST", "9V-AAA", "F-TEST"],
                "typecode": ["B744", "A359", "A320"],
                "origin": ["EGLL", "WSSS", "LFPG"],
                "destination": ["KJFK", "EGLL", "KJFK"],
                "firstseen": [1704103200, 1704106800, 1704110400],
                "lastseen": [1704132000, 1704132000, 1704121200],
            }
        )

        filtered = scientific_db.filter_covid_chunk(
            chunk,
            origin_airports=("EGLL", "WSSS"),
            minimum_duration_hours=6.0,
            source_file="flightlist_20200101_20200131.csv.gz",
        )

        self.assertEqual(len(filtered), 2)
        self.assertTrue((filtered["origin"].isin(["EGLL", "WSSS"])).all())
        self.assertTrue((filtered["duration_hours"] >= 6.0).all())
        self.assertTrue((filtered["source_dataset"] == "covid_flight_dataset").all())
        self.assertIn("flight_key", filtered.columns)

    def test_build_default_trajectory_query_without_sampling(self) -> None:
        sql, params = scientific_db.build_default_trajectory_query(
            origin_airport="EGLL",
            minimum_duration_hours=6.0,
            start_time="2020-01-01 00:00:00",
            end_time="2020-12-31 23:59:59",
            sample_trajectories=None,
        )

        self.assertIn("WITH filtered_flights AS", sql)
        self.assertIn("scientific_state_vectors", sql)
        self.assertNotIn("sampled_flights", sql)
        self.assertEqual(params["origin_airport"], "EGLL")
        self.assertEqual(params["minimum_duration_hours"], 6.0)
        self.assertEqual(params["start_time"], "2020-01-01 00:00:00")
        self.assertEqual(params["end_time"], "2020-12-31 23:59:59")

    def test_build_default_trajectory_query_with_sampling(self) -> None:
        sql, params = scientific_db.build_default_trajectory_query(
            origin_airport="WSSS",
            minimum_duration_hours=7.0,
            sample_trajectories=25,
        )

        self.assertIn("sampled_flights", sql)
        self.assertIn("ORDER BY RANDOM()", sql)
        self.assertEqual(params["sample_trajectories"], 25)

    def test_extract_date_from_name_supports_multiple_patterns(self) -> None:
        self.assertEqual(
            scientific_db.extract_date_from_name("states_2020-03-23_00.csv.tar"),
            pd.Timestamp("2020-03-23").date(),
        )
        self.assertEqual(
            scientific_db.extract_date_from_name("states_20200323_00.csv.tar"),
            pd.Timestamp("2020-03-23").date(),
        )
        self.assertIsNone(
            scientific_db.extract_date_from_name("states_unknown.csv.tar")
        )

    def test_extract_archive_hour_from_name_returns_utc_hour(self) -> None:
        self.assertEqual(
            scientific_db.extract_archive_hour_from_name(
                "states_2020-03-23-14.csv.tar"
            ),
            pd.Timestamp("2020-03-23 14:00:00", tz="UTC"),
        )
        self.assertIsNone(
            scientific_db.extract_archive_hour_from_name("states_unknown.csv.tar")
        )

    def test_select_relevant_state_archives_filters_to_candidate_hours(self) -> None:
        archives = [
            {
                "name": "states_2020-03-23-14.csv.tar",
                "url": "https://example.com/states_2020-03-23-14.csv.tar",
            },
            {
                "name": "states_2020-03-23-15.csv.tar",
                "url": "https://example.com/states_2020-03-23-15.csv.tar",
            },
            {
                "name": "states_2020-03-24-10.csv.tar",
                "url": "https://example.com/states_2020-03-24-10.csv.tar",
            },
        ]

        selected = scientific_db.select_relevant_state_archives(
            archives,
            candidate_days={pd.Timestamp("2020-03-23").date()},
            candidate_hours={pd.Timestamp("2020-03-23 14:00:00", tz="UTC")},
        )

        self.assertEqual([item["name"] for item in selected], ["states_2020-03-23-14.csv.tar"])

    def test_summarize_covid_filter_counts_reports_filter_stages(self) -> None:
        chunk = pd.DataFrame(
            {
                "callsign": ["BAW001 ", "SIA305", "AFR123"],
                "number": ["BA1", "SQ305", "AF123"],
                "icao24": ["abc123", "def456", "ghi789"],
                "registration": ["G-TEST", "9V-AAA", "F-TEST"],
                "typecode": ["B744", "A359", "A320"],
                "origin": ["EGLL", "WSSS", "LFPG"],
                "destination": ["KJFK", "EGLL", "KJFK"],
                "firstseen": [1704103200, 1704106800, 1704110400],
                "lastseen": [1704132000, 1704132000, 1704121200],
            }
        )

        summary = scientific_db.summarize_covid_filter_counts(
            chunk,
            origin_airports=("EGLL", "WSSS"),
            minimum_duration_hours=6.0,
        )

        self.assertEqual(summary["total_rows"], 3)
        self.assertEqual(summary["origin_match_rows"], 2)
        self.assertEqual(summary["duration_match_rows"], 2)
        self.assertEqual(summary["final_match_rows"], 2)

    def test_build_origin_duration_summary_orders_by_long_haul_counts(self) -> None:
        summary_df = pd.DataFrame(
            {
                "origin": ["EGLL", "EGLL", "OMDB", "OMDB", "OMDB", "EDDF"],
                "duration_hours": [7.5, 5.0, 9.0, 6.5, 3.0, 4.5],
            }
        )

        result = scientific_db.build_origin_duration_summary(
            summary_df,
            top_n_airports=3,
        )

        self.assertEqual(result.iloc[0]["origin"], "OMDB")
        self.assertEqual(int(result.iloc[0]["ge_6h"]), 2)
        self.assertEqual(int(result.iloc[1]["ge_6h"]), 1)

    def test_state_row_to_record_converts_into_database_shape(self) -> None:
        flight = {
            "flight_key": "abc123|TEST123|2020-01-01T00:00:00+00:00",
            "origin": "EGLL",
            "destination": "KJFK",
            "firstseen": pd.Timestamp("2020-01-01 00:00:00", tz="UTC"),
            "lastseen": pd.Timestamp("2020-01-01 08:00:00", tz="UTC"),
            "duration_hours": 8.0,
        }
        row = {
            "time": "1577836800",
            "icao24": "ABC123",
            "callsign": " TEST123 ",
            "lat": "51.47",
            "lon": "-0.45",
            "velocity": "250.0",
            "heading": "270.0",
            "vertrate": "5.0",
            "onground": "false",
            "alert": "false",
            "spi": "false",
            "squawk": "1234",
            "baroaltitude": "300.0",
            "geoaltitude": "320.0",
            "lastposupdate": "1577836790",
            "lastcontact": "1577836795",
        }

        record = scientific_db.state_row_to_record(
            row,
            flight,
            source_file="states_20200101_00.csv.tar",
            source_member="states_20200101_00.csv",
        )

        self.assertEqual(record["icao24"], "abc123")
        self.assertEqual(record["callsign"], "TEST123")
        self.assertEqual(record["source_dataset"], "weekly_state_vectors")
        self.assertEqual(record["source_file"], "states_20200101_00.csv.tar")
        self.assertEqual(record["source_member"], "states_20200101_00.csv")
        self.assertEqual(record["hour"], pd.Timestamp("2020-01-01 00:00:00", tz="UTC"))


if __name__ == "__main__":
    unittest.main()
