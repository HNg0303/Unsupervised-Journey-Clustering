from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from Rule_based.config import PipelineConfig
from Rule_based.production import canonicalize_frame
from Rule_based import segment, tokens


class ProductionPipelineTest(unittest.TestCase):
    def fixture(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"session_id": "s1", "platform": "Android", "key": "view", "segmentation_name": "home", "screen_id": "home", "client_time": 1775001600000},
            {"session_id": "s1", "platform": "Android", "key": "action", "segmentation_name": "click_submit", "client_time": 1775001603000},
            {"session_id": "s1", "platform": "Android", "key": "view", "segmentation_name": "payment", "screen_id": "payment", "client_time": 1775001700001},
        ])

    def test_mixed_time_token_and_gap(self) -> None:
        frame, report = canonicalize_frame(self.fixture(), source_file="fixture.csv")
        self.assertEqual(report["invalid_client_time"], 0)
        self.assertEqual(frame.event_token.tolist(), ["view@home", "action@home#click_submit", "view@payment"])
        self.assertEqual(frame.gap_prev_seconds.iloc[1], 3.0)
        self.assertAlmostEqual(frame.gap_next_seconds.iloc[1], 97.001)
        self.assertTrue(pd.isna(frame.gap_prev_seconds.iloc[0]))
        self.assertTrue(pd.isna(frame.gap_next_seconds.iloc[-1]))
        self.assertNotIn("duration", frame.columns)

    def test_screen_context_uses_only_prior_view_in_same_stream(self) -> None:
        frame, _ = canonicalize_frame(pd.DataFrame([
            {"session_id": "s1", "platform": "Android", "key": "action", "segmentation_name": "before_view", "client_time": 1000},
            {"session_id": "s1", "platform": "Android", "key": "view", "segmentation_name": "home", "screen_id": "home", "client_time": 1000},
            {"session_id": "s1", "platform": "Android", "key": "action", "segmentation_name": "explicit", "screen_id": "payment", "client_time": 2000},
            {"session_id": "s1", "platform": "Android", "key": "action", "segmentation_name": "inferred", "client_time": 2000},
            {"session_id": "s2", "platform": "Android", "key": "action", "segmentation_name": "other_session", "client_time": 1000},
            {"session_id": "s1", "platform": "iOS", "key": "action", "segmentation_name": "other_platform", "client_time": 1000},
        ]))
        self.assertEqual(
            frame.event_token.tolist(),
            [
                "action@<missing>#before_view",
                "view@home",
                "action@payment#explicit",
                "action@home#inferred",
                "action@<missing>#other_session",
                "action@<missing>#other_platform",
            ],
        )

    def test_screen_context_resets_after_idle_and_auth_journey_boundaries(self) -> None:
        frame, _ = canonicalize_frame(pd.DataFrame([
            {"session_id": "s1", "platform": "Android", "key": "view", "segmentation_name": "home", "screen_id": "home", "client_time": 1000},
            {"session_id": "s1", "platform": "Android", "key": "action", "segmentation_name": "before_idle", "client_time": 2000},
            {"session_id": "s1", "platform": "Android", "key": "action", "segmentation_name": "after_idle", "client_time": 100000},
            {"session_id": "s1", "platform": "Android", "key": "action", "segmentation_name": "continue_login", "client_time": 101000},
            {"session_id": "s1", "platform": "Android", "key": "action", "segmentation_name": "after_auth_boundary", "client_time": 102000},
        ]))
        self.assertEqual(
            frame.event_token.tolist(),
            [
                "view@home",
                "action@home#before_idle",
                "action@<missing>#after_idle",
                "action@<missing>#continue_login",
                "action@<missing>#after_auth_boundary",
            ],
        )

    def test_idle_boundary(self) -> None:
        frame, _ = canonicalize_frame(self.fixture(), source_file="fixture.csv")
        cfg = PipelineConfig()
        assigned = segment.assign_journeys(tokens.build_tokens(frame, cfg.canonize), cfg.segment)
        self.assertEqual(assigned.boundary_reason.tolist(), ["session_start", "", "idle_gap"])

    def test_flattened_export_column_aliases_are_accepted(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "_id.$oid": "record-1",
                    "session_id": "s1",
                    "device_id": "d1",
                    "key": "[CLY]_view",
                    "segmentation.segment": "iOS",
                    "segmentation.name": "HomeVC",
                    "segmentation.screen_id": "HomeVC",
                    "timestamp": 1775001600000,
                }
            ]
        )
        frame, report = canonicalize_frame(source, source_file="flattened.csv")
        self.assertEqual(report["rows_output"], 1)
        self.assertEqual(frame.loc[0, "record_id"], "record-1")
        self.assertEqual(frame.loc[0, "platform"], "ios")
        self.assertEqual(frame.loc[0, "event_token"], "view@HomeVC")


if __name__ == "__main__":
    unittest.main()
