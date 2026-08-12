from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from Rule_based.hierarchical_score import HierarchicalJourneyScorer  # noqa: E402
from Rule_based.score import JourneyScorer, ScoreThresholds  # noqa: E402


class FakeConfig:
    def to_dict(self) -> dict:
        return {name: {} for name in ("canonize", "tokens", "post", "segment")}


class FakeVectorizer:
    channel_weights: dict[str, float] = {}

    def transform(
        self,
        journeys: pd.DataFrame,
        sequences: list[list[str]],
        channels: dict[str, list[list[str]]] | None = None,
    ) -> np.ndarray:
        assert not channels, "this fake was built without semantic channels"
        return journeys[["x"]].to_numpy(dtype=float)


class FakeMarkov:
    vocab: list[str] = []

    def score(self, sequence: list[str], label: int) -> float:
        return -1.0

    def score_all(self, sequences: list[list[str]], labels: np.ndarray) -> np.ndarray:
        return np.full(len(sequences), -1.0)

    def predict_next(self, sequence: list[str], label: int, top_k: int = 1) -> list:
        return []


def scorer(centroids: dict[int, np.ndarray]) -> JourneyScorer:
    return JourneyScorer(
        FakeConfig(),
        FakeVectorizer(),
        centroids,
        FakeMarkov(),
        ScoreThresholds(
            distance_p95=100.0,
            markov_p05=-2.0,
            markov_p01=-3.0,
            back_rate_p90=0.5,
            loops_p90=1.0,
            revisit_p90=0.5,
            span_p95=100.0,
        ),
    )


def journeys(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "journey_id": [f"j{i}" for i in range(len(values))],
            "x": values,
            "back_rate": 0.0,
            "n_loop_removed": 0,
            "revisit_ratio": 0.0,
            "span_seconds": 1.0,
        }
    )


class HierarchicalScoreTest(unittest.TestCase):
    def test_cluster_specific_distance_limits_override_global_threshold(self) -> None:
        model = scorer({1: np.array([0.0]), 2: np.array([10.0])})
        frame = journeys([0.5, 9.5])
        result = model.score_prepared(
            frame,
            [["a"], ["b"]],
            distance_limits={1: 0.4, 2: 1.0},
        )
        self.assertEqual(result["nearest_cluster"].tolist(), [1, 2])
        self.assertEqual(result["cluster"].tolist(), [-1, 2])

    def test_C_first_then_B_fallback(self) -> None:
        model = HierarchicalJourneyScorer(
            scorer({1: np.array([0.0])}),
            scorer({2: np.array([1.0]), 3: np.array([5.0])}),
            c_distance_limits={1: 0.2},
            b_distance_limits={2: 0.5, 3: 0.5},
            b_markov_limits={2: -2.0, 3: -2.0},
            min_distance_margin=0.1,
            name_mapping={
                "C:1": {"cluster_name": "C one", "business_family": "c", "naming_confidence": "high"},
                "B:2": {"cluster_name": "B two", "business_family": "b", "naming_confidence": "high"},
            },
        )
        frame = journeys([0.1, 1.1, 9.0])
        result = model.score_prepared(frame, [["a"], ["b"], ["c"]])
        self.assertEqual(
            result["effective_cluster_key"].tolist(), ["C:1", "B:2", "UNKNOWN"]
        )
        self.assertEqual(
            result["assignment_type"].tolist(),
            ["C_primary", "B_secondary_inference", "unassigned_novel"],
        )
        self.assertEqual(
            result["cluster_name"].tolist(),
            ["C one", "B two", "Hành trình chưa phân loại / hỗn hợp"],
        )


if __name__ == "__main__":
    unittest.main()
