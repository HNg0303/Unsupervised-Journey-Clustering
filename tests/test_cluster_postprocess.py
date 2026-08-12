from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from Rule_based.cluster_postprocess import (  # noqa: E402
    align_journeys,
    apply_hierarchy,
    nearest_two_centroids,
)


class ClusterPostprocessTest(unittest.TestCase):
    def test_align_journeys_uses_id_not_row_order(self) -> None:
        c = pd.DataFrame(
            {"journey_id": ["j1", "j2"], "cluster": [-1, 4], "sequence": ["a", "b"]}
        )
        b = pd.DataFrame(
            {"journey_id": ["j2", "j1"], "cluster": [8, 3], "sequence": ["b", "a"]}
        )
        aligned = align_journeys(c, b)
        self.assertEqual(aligned["b_cluster"].tolist(), [3, 8])
        self.assertEqual(aligned["c_cluster"].tolist(), [-1, 4])

    def test_nearest_two_centroids_returns_ordered_pair(self) -> None:
        matrix = np.array([[0.1, 0.0], [9.0, 0.0]])
        centroids = {4: np.array([0.0, 0.0]), 8: np.array([10.0, 0.0])}
        first, distance, second, second_distance = nearest_two_centroids(
            matrix, centroids, chunk_size=1
        )
        self.assertEqual(first.tolist(), [4, 8])
        self.assertEqual(second.tolist(), [8, 4])
        np.testing.assert_allclose(distance, [0.1, 1.0])
        np.testing.assert_allclose(second_distance, [9.9, 9.0])

    def test_hierarchy_preserves_C_and_namespaces_B(self) -> None:
        frame = pd.DataFrame(
            {
                "journey_id": ["primary", "b-fitted", "b-soft", "repeat", "friction"],
                "c_cluster": [7, -1, -1, -1, -1],
                "b_cluster": [2, 3, -1, -1, -1],
                "nearest_b_cluster": [2, 3, 9, 5, 6],
                "exact_sequence_noise_support": [0, 1, 1, 12, 1],
                "friction_flags": ["", "", "", "", "navigation_loop"],
            }
        )
        result = apply_hierarchy(
            frame,
            soft_pass=np.array([False, False, True, False, False]),
            recurring_support=10,
        )
        self.assertEqual(
            result["assignment_type"].tolist(),
            [
                "C_primary",
                "B_existing_secondary",
                "B_borderline_secondary",
                "unassigned_recurring",
                "unassigned_friction",
            ],
        )
        self.assertEqual(result["effective_cluster_key"].tolist(), ["C:7", "B:3", "B:9", "UNKNOWN", "UNKNOWN"])


if __name__ == "__main__":
    unittest.main()
