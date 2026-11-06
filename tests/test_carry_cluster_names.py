from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import carry_cluster_names as carry  # noqa: E402

SITEMAP = {
    "M05": ("Internet", "", ""),
    "M05.06": ("Internet", "Wi-Fi", ""),
    "M05.06.01": ("Internet", "Wi-Fi", "Đổi mật khẩu"),
    "M05.06.02": ("Internet", "Wi-Fi", "Đổi tên"),
    "M05.07": ("Internet", "Modem", ""),
    "M05.07.01": ("Internet", "Modem", "Bật tắt modem"),
    "M10": ("Thanh toán", "", ""),
    "M10.01": ("Thanh toán", "Hóa đơn", ""),
    "M10.01.01": ("Thanh toán", "Hóa đơn", "Thanh toán hóa đơn"),
}


def name(taxonomy_id: str, cluster_name: str | None = None) -> dict[str, str]:
    levels = SITEMAP[taxonomy_id]
    return {"taxonomy_id": taxonomy_id,
            "cluster_name": cluster_name or " | ".join(level for level in levels if level),
            "business_family": levels[0], "business_submodule": levels[1], "business_detail": levels[2]}


NAMES = {
    1: name("M05.06.01", "Đổi mật khẩu Wi-Fi"),
    2: name("M05.06.01"),
    3: name("M05.06.02"),
    4: name("M05.07.01"),
    5: name("M10.01.01"),
    6: name("M05.07"),
}


def decide(counts: dict[int, int]) -> dict[str, object]:
    return carry.decide(Counter(counts), NAMES, SITEMAP, threshold=0.7, min_coverage=0.5)


class DecideTest(unittest.TestCase):
    def test_one_old_cluster_above_threshold_keeps_its_own_name(self) -> None:
        row = decide({1: 80, 3: 20})
        self.assertEqual(row["carry_step"], "old_cluster")
        self.assertEqual(row["cluster_name"], "Đổi mật khẩu Wi-Fi")
        self.assertEqual(row["carried_from_cluster"], 1)

    def test_old_clusters_with_the_same_taxonomy_id_are_summed(self) -> None:
        row = decide({1: 40, 2: 40, 3: 20})
        self.assertEqual((row["carry_step"], row["taxonomy_id"]), ("taxonomy_id", "M05.06.01"))
        self.assertEqual(row["cluster_name"], "Internet | Wi-Fi | Đổi mật khẩu")
        self.assertAlmostEqual(row["carry_share"], 0.8)

    def test_falls_back_to_submodule_then_module(self) -> None:
        row = decide({1: 45, 3: 45, 5: 10})
        self.assertEqual((row["carry_step"], row["taxonomy_id"]), ("submodule", "M05.06"))
        row = decide({1: 40, 4: 30, 6: 10, 5: 20})
        self.assertEqual((row["carry_step"], row["taxonomy_id"]), ("module", "M05"))

    def test_undecided_cluster_gets_a_suggestion_for_review(self) -> None:
        row = decide({1: 50, 5: 50})
        self.assertEqual(row["carry_step"], "review")
        self.assertTrue(row["taxonomy_id"])

    def test_ties_never_decide_a_level(self) -> None:
        row = carry.decide(Counter({1: 50, 5: 50}), NAMES, SITEMAP, threshold=0.5, min_coverage=0.5)
        self.assertEqual(row["carry_step"], "review")

    def test_old_noise_and_unnamed_clusters_do_not_count_toward_shares(self) -> None:
        row = decide({-1: 40, 99: 5, 1: 55})
        self.assertEqual(row["carry_step"], "old_cluster")
        self.assertAlmostEqual(row["carry_coverage"], 0.55)

    def test_low_coverage_is_new_behaviour(self) -> None:
        row = decide({-1: 70, 1: 30})
        self.assertEqual(row["carry_step"], "new")
        self.assertEqual(row["taxonomy_id"], "M05.06.01")
        row = decide({-1: 100})
        self.assertEqual((row["carry_step"], row["cluster_name"]), ("new", ""))


class TokensTest(unittest.TestCase):
    def test_reads_numpy_repr_arrow_strings_and_lists(self) -> None:
        self.assertEqual(carry.tokens("['view@HOME' 'view@android/Home'\n 'abort:back']"),
                         ["view@HOME", "view@android/Home", "abort:back"])
        self.assertEqual(carry.tokens("['a' \"it's\"]")[-1], "it's")
        self.assertEqual(carry.tokens("view@HOME -> action@x#y"), ["view@HOME", "action@x#y"])
        self.assertEqual(carry.tokens(np.array(["a", "b"])), ["a", "b"])
        self.assertEqual(carry.tokens(float("nan")), [])


class CarryPlatformTest(unittest.TestCase):
    def test_writes_names_and_report_and_samples_per_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "old" / "android").mkdir(parents=True)
            (tmp / "old" / "android" / "android_journey_scorer.pkl").write_bytes(b"")
            names = pd.DataFrame([{"platform": "android", "cluster_id": c, **n} for c, n in NAMES.items()])
            names.to_csv(tmp / "old" / "android" / "android_named_clusters.csv", index=False)
            sitemap = tmp / "sitemap.csv"
            pd.DataFrame([{"taxonomy_id": k, "business_family": v[0], "business_submodule": v[1],
                           "business_detail": v[2]} for k, v in SITEMAP.items()]).to_csv(sitemap, index=False)
            (tmp / "new" / "android").mkdir(parents=True)
            # new cluster 0 was old cluster 5, new cluster 1 is unseen behaviour
            journeys = pd.DataFrame({"cluster": [0] * 30 + [1] * 10 + [-1] * 5,
                                     "old": [5] * 30 + [-1] * 10 + [-1] * 5,
                                     "sequence": "a -> b"})
            journeys.to_csv(tmp / "new" / "android" / "android_journeys.csv", index=False)

            args = carry.argparse.Namespace(
                old_run=tmp / "old", old_names=tmp / "old", new_run=tmp / "new", output_dir=tmp / "new",
                threshold=0.7, min_coverage=0.5, max_per_cluster=8, chunksize=7, seed=0, overwrite=False)
            with mock.patch.object(carry.JourneyScorer, "load", return_value=None), \
                    mock.patch.object(carry, "old_clusters", side_effect=lambda _, frame: frame["old"].to_numpy()):
                carry.carry_platform(args, "android", carry.load_sitemap(sitemap))

            report = pd.read_csv(tmp / "new" / "android" / "android_carry_report.csv").set_index("cluster_id")
            self.assertEqual(report.loc[0, "carry_step"], "old_cluster")
            self.assertEqual(report.loc[0, "taxonomy_id"], "M10.01.01")
            self.assertEqual(report.loc[0, "size"], 30)
            self.assertEqual(report.loc[0, "sampled"], 8)
            self.assertEqual(report.loc[1, "carry_step"], "new")
            self.assertTrue(report.loc[1, "needs_review"])
            self.assertEqual(report.loc[-1, "carry_step"], "noise")
            named = pd.read_csv(tmp / "new" / "android" / "android_named_clusters.csv")
            self.assertEqual(sorted(named["cluster_id"]), [-1, 0])
            with self.assertRaises(SystemExit):
                carry.carry_platform(args, "android", carry.load_sitemap(sitemap))


if __name__ == "__main__":
    unittest.main()
