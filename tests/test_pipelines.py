from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from journey_clustering.config import PipelineConfig  # noqa: E402
from journey_clustering.pipelines import (  # noqa: E402
    prepare_event_partition,
    read_journey_partitions,
    sequence_value,
)
from journey_clustering.storage import (  # noqa: E402
    stable_session_bucket,
    write_parquet,
)


class PipelineTest(unittest.TestCase):
    def test_session_bucket_is_stable_and_platform_scoped(self) -> None:
        first = stable_session_bucket("Android", "session-1", 128)
        second = stable_session_bucket("android", "session-1", 128)
        other_platform = stable_session_bucket("ios", "session-1", 128)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_platform)

    def test_sequence_value_handles_parquet_lists_and_missing_values(self) -> None:
        self.assertEqual(sequence_value(["a", "b"]), ["a", "b"])
        self.assertEqual(sequence_value("a -> b"), ["a", "b"])
        self.assertEqual(sequence_value(pd.NA), [])

    def test_partition_preparation_makes_journey_ids_global(self) -> None:
        rows = []
        for session_id in ("s1", "s2"):
            for position, (key, name) in enumerate(
                [("view", "home"), ("action", "tap_one"), ("view", "detail"), ("action", "tap_two")]
            ):
                rows.append(
                    {
                        "session_id": session_id,
                        "platform": "android",
                        "key": key,
                        "segmentation_name": name,
                        "screen_id": name if key == "view" else None,
                        "client_time": 1_775_001_600_000 + position * 1_000,
                    }
                )
        cfg = PipelineConfig()
        cfg.segment.min_journey_length = 4
        cfg.post.drop_chrome = False
        prepared = prepare_event_partition(
            pd.DataFrame(rows),
            partition_id="platform=android/bucket=003",
            platform="android",
            cfg=cfg,
        )
        self.assertEqual(len(prepared.journeys), 2)
        self.assertEqual(len(set(prepared.journeys["journey_id"])), 2)
        self.assertTrue(all("platform=android_bucket=003::" in value for value in prepared.journeys["journey_id"]))
        self.assertTrue(prepared.journeys["model_eligible"].all())
        self.assertEqual(len(prepared.channels["coarse"]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journeys.parquet"
            write_parquet(prepared.journeys, path)
            round_trip = pd.read_parquet(path)
        self.assertEqual(sequence_value(round_trip.loc[0, "sequence_tokens"]), prepared.sequences[0])
        self.assertEqual(sequence_value(round_trip.loc[0, "channel_coarse"]), prepared.channels["coarse"][0])

    def test_drop_boot_and_chrome_remove_structural_events_and_keep_channels_aligned(self) -> None:
        rows = [
            {"session_id": "s1", "platform": "android", "key": "view", "segmentation_name": "SplashActivity", "screen_id": "SplashActivity", "client_time": 1_775_001_600_000},
            {"session_id": "s1", "platform": "android", "key": "view", "segmentation_name": "MainAppActivity", "screen_id": "MainAppActivity", "client_time": 1_775_001_601_000},
            {"session_id": "s1", "platform": "android", "key": "view", "segmentation_name": "HomeVC", "screen_id": "HomeVC", "client_time": 1_775_001_602_000},
            {"session_id": "s1", "platform": "android", "key": "action", "segmentation_name": "click_submit", "screen_id": None, "client_time": 1_775_001_603_000},
            {"session_id": "s1", "platform": "android", "key": "view", "segmentation_name": "HiWebViewActivity", "screen_id": "HiWebViewActivity", "client_time": 1_775_001_604_000},
            {"session_id": "s1", "platform": "android", "key": "view", "segmentation_name": "payment", "screen_id": "payment", "client_time": 1_775_001_605_000},
        ]
        cfg = PipelineConfig()
        cfg.segment.min_journey_length = 2
        cfg.post.drop_chrome = True
        cfg.post.drop_boot = True

        prepared = prepare_event_partition(
            pd.DataFrame(rows),
            partition_id="platform=android/bucket=004",
            platform="android",
            cfg=cfg,
        )

        sequence = prepared.sequences[0]
        self.assertEqual(prepared.journeys.loc[0, "n_dropped_screens"], 3)
        self.assertNotIn("view@SplashActivity", sequence)
        self.assertNotIn("view@MainAppActivity", sequence)
        self.assertNotIn("view@HiWebViewActivity", sequence)
        self.assertIn("view@HomeVC", sequence)
        self.assertIn("view@payment", sequence)
        self.assertEqual(len(sequence), len(prepared.channels["coarse"][0]))
        self.assertEqual(len(sequence), len(prepared.channels["intent"][0]))
        self.assertEqual(len(sequence), len(prepared.channels["operation"][0]))

    def test_journey_partitions_can_be_loaded_as_one_logical_dataset(self) -> None:
        frame = pd.DataFrame(
            {
                "journey_id": ["p::j1", "p::j2"],
                "platform": ["android", "android"],
                "model_eligible": [True, False],
                "n_events_final": [4, 2],
                "sequence_tokens": [["a", "b", "c", "d"], ["a", "b"]],
                "backoff_sequence": [["a", "b", "c", "d"], ["a", "b"]],
                "channel_coarse": [["x", "y", "z", "q"], ["x", "y"]],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.parquet"
            write_parquet(frame, path)
            loaded = read_journey_partitions([path], platform="android")
        self.assertEqual(loaded["journey_id"].tolist(), ["p::j1"])

    def test_customer_stratified_sampling_preserves_customer_coverage(self) -> None:
        frame = pd.DataFrame(
            {
                "journey_id": [f"j{i}" for i in range(12)],
                "platform": ["android"] * 12,
                "customer_id": ["large"] * 8 + ["medium"] * 3 + ["small"],
                "model_eligible": [True] * 12,
                "n_events_final": [4] * 12,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.parquet"
            write_parquet(frame, path)
            loaded = read_journey_partitions(
                [path], platform="android", max_journeys=6, seed=7
            )
        self.assertEqual(len(loaded), 6)
        self.assertEqual(set(loaded["customer_id"]), {"large", "medium", "small"})
        self.assertEqual(
            loaded["customer_id"].value_counts().to_dict(),
            {"large": 3, "medium": 2, "small": 1},
        )

    def test_customer_stratified_sampling_rejects_too_small_cap(self) -> None:
        frame = pd.DataFrame(
            {
                "journey_id": ["j1", "j2", "j3"],
                "platform": ["android"] * 3,
                "customer_id": ["a", "b", "c"],
                "model_eligible": [True] * 3,
                "n_events_final": [4] * 3,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.parquet"
            write_parquet(frame, path)
            with self.assertRaisesRegex(ValueError, "cannot preserve all customers"):
                read_journey_partitions(
                    [path], platform="android", max_journeys=2
                )


if __name__ == "__main__":
    unittest.main()
