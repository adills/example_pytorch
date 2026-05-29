import unittest

import pandas as pd

import opensky_build_scientific_db as scientific_db


class OpenSkyBuildScientificDbTestCase(unittest.TestCase):
    def test_build_parser_defaults_to_local_opensky_scientific_database(self) -> None:
        parser = scientific_db.build_parser()

        build_args = parser.parse_args(
            ["build", "--download-dir", "/tmp/opensky"]
        )
        query_args = parser.parse_args(["query"])

        self.assertIn("opensky_scientific", build_args.database_url)
        self.assertIn("@localhost/", build_args.database_url)
        self.assertEqual(build_args.database_url, scientific_db.DEFAULT_DATABASE_URL)
        self.assertEqual(query_args.database_url, scientific_db.DEFAULT_DATABASE_URL)

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
