from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from journey_clustering.config import PipelineConfig
from journey_clustering.experiment import (
    build_prepared_data_slug,
    build_run_slug,
    split_sessions_chronologically,
)


class ChronologicalSessionSplitTest(unittest.TestCase):
    def test_latest_complete_sessions_are_held_out_without_leakage(self) -> None:
        rows = []
        for day, session_id, event_count in (
            (1, "s1", 2),
            (2, "s2", 3),
            (3, "s3", 2),
            (4, "s4", 4),
        ):
            for offset in range(event_count):
                rows.append(
                    {
                        "session_id": session_id,
                        "event_time": pd.Timestamp(
                            year=2026, month=1, day=day, minute=offset, tz="UTC"
                        ),
                        "event": offset,
                    }
                )
        frame = pd.DataFrame(rows)

        train, test, report = split_sessions_chronologically(frame, test_size=0.25)

        self.assertEqual(set(train.session_id), {"s1", "s2", "s3"})
        self.assertEqual(set(test.session_id), {"s4"})
        self.assertEqual(len(test), 4)
        self.assertFalse(set(train.session_id) & set(test.session_id))
        self.assertEqual(report["strategy"], "chronological_complete_session")
        self.assertEqual(report["session_overlap"], 0)

    def test_split_requires_at_least_two_sessions(self) -> None:
        frame = pd.DataFrame(
            {
                "session_id": ["only", "only"],
                "event_time": pd.to_datetime(
                    ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"]
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "at least two sessions"):
            split_sessions_chronologically(frame, test_size=0.2)


class ExperimentPathTest(unittest.TestCase):
    def test_run_slug_changes_with_hyperparameters(self) -> None:
        base = PipelineConfig()
        tuned = PipelineConfig()
        tuned.features.ngram_range = (2, 4)
        tuned.features.svd_components = 128
        tuned.cluster.min_cluster_size = 50
        tuned.cluster.min_samples = 3

        base_slug = build_run_slug(base, test_size=0.2)
        tuned_slug = build_run_slug(tuned, test_size=0.2)

        self.assertNotEqual(base_slug, tuned_slug)
        self.assertIn("ng2-4", tuned_slug)
        self.assertIn("svd128", tuned_slug)
        self.assertIn("mcs50", tuned_slug)
        self.assertIn("ms3", tuned_slug)

    def test_cluster_tuning_reuses_the_same_prepared_data_cache(self) -> None:
        source = ROOT / "data" / "train_data"
        base = PipelineConfig()
        tuned = PipelineConfig()
        tuned.cluster.min_cluster_size = 30
        tuned.features.svd_components = 128

        self.assertEqual(
            build_prepared_data_slug(base, input_path=source, test_size=0.2),
            build_prepared_data_slug(tuned, input_path=source, test_size=0.2),
        )


if __name__ == "__main__":
    unittest.main()
