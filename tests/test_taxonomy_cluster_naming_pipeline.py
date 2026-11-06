from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from journey_clustering import naming as MODULE  # noqa: E402


class TaxonomyParserTest(unittest.TestCase):
    def test_parse_taxonomy_csv(self) -> None:
        with self.subTest("valid CSV"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                taxonomy = Path(directory) / "taxonomy.csv"
                taxonomy.write_text(
                    "taxonomy_id,business_family,business_submodule,business_detail\n"
                    "M01.01.01,Truy cập,Đăng nhập,Đăng nhập FPT ID\n",
                    encoding="utf-8",
                )
                self.assertEqual(MODULE.parse_taxonomy(taxonomy), [
                    MODULE.TaxonomyLeaf(
                        "M01.01.01", "Truy cập", "Đăng nhập", "Đăng nhập FPT ID"
                    )
                ])

    def test_parse_taxonomy_csv_requires_columns(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            taxonomy = Path(directory) / "taxonomy.csv"
            taxonomy.write_text(
                "taxonomy_id,name\nM01.01.01,FPT ID\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "missing taxonomy columns"):
                MODULE.parse_taxonomy(taxonomy)


# The repository copy of the Hi FPT sitemap, feature rows only.
LEAVES = [
    item for item in MODULE.parse_taxonomy(ROOT / "taxonomy" / "hifpt_sitemap_taxonomy.csv")
    if item.detail
]


def ngram(cluster: int, rank: int, text: str, mass: float) -> dict[str, object]:
    return {
        "platform": "android", "cluster": cluster, "rank": rank,
        "ngram": text, "lift": 100.0, "mass": mass,
    }


def names(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    _, mappings = MODULE.analyse(LEAVES, rows, 2, 0.15, 0.60)
    return {int(row["cluster_id"]): row for row in mappings}


class EvidenceTest(unittest.TestCase):
    def test_generic_event_falls_back_to_screen(self) -> None:
        self.assertEqual(MODULE.evidence_text("action@android/home/payment/my_cards#btn_back"), "android/home/payment/my_cards")
        self.assertEqual(MODULE.evidence_text("action@HomeVC#Home/Nav_payement"), "HomeVC")
        self.assertEqual(MODULE.evidence_text("action@HomeVC#Home/change_contract"), "Home/change_contract")
        self.assertEqual(MODULE.evidence_text("<eos>"), "")

    def test_deepest_path_segment_wins(self) -> None:
        text = (
            "internet_fprotect_screen/management_device/management_device_screen/"
            "devices_details/device_information/locking_internet"
        )
        self.assertEqual(MODULE.deepest_match(text, MODULE.SUBMODULE_EVIDENCE), "M05.05")
        self.assertEqual(MODULE.deepest_match("hi.fpt.vn/web/shop/non-ftel/prepaid-wifi/introducing", MODULE.SUBMODULE_EVIDENCE), "M08.07")

    def test_card_tokens_never_name_vpbank(self) -> None:
        # Android cluster 299 of run 678: autopay, linked cards and payment
        # settings. "card" alone used to validate "Mở thẻ tín dụng VPBank".
        rows = [
            ngram(299, 1, "view@android/PAYMENT/LISTAUTOPAY view@HOME", 68),
            ngram(299, 2, "view@android/home/payment/setting view@HOME", 29),
            ngram(299, 3, "view@android/home/payment/my_cards view@HOME", 26),
            ngram(299, 4, "action@android/home/payment/my_cards#android/home/payment/my_cards/link_card", 5),
        ]
        named = names(rows)[299]
        self.assertNotIn("VPBank", str(named["cluster_name"]))
        self.assertEqual(named["cluster_name"], "Hóa đơn và thanh toán")
        self.assertEqual(named["naming_source"], "validated_taxonomy_module")

    def test_home_names_only_a_cluster_with_nothing_else(self) -> None:
        rows = [
            ngram(1, 1, "view@HomeVC", 90), ngram(1, 2, "action@HomeVC#btn_back view@HomeVC", 80),
            ngram(1, 3, "view@android/home/payment/my_cards", 30), ngram(1, 4, "action@android/home/payment/my_cards#btn_back", 20),
            ngram(2, 1, "view@HomeVC", 90), ngram(2, 2, "action@HomeVC#btn_back view@HomeVC", 80),
        ]
        named = names(rows)
        self.assertEqual(named[1]["taxonomy_id"], "M10.05.01")
        self.assertEqual(named[2]["taxonomy_id"], "M02.01.01")

    def test_evidence_tables_reference_sitemap_ids(self) -> None:
        ids = {item.taxonomy_id for item in LEAVES}
        submodules = {taxonomy_id[:6] for taxonomy_id in ids}
        self.assertLessEqual(set(MODULE.SUBMODULE_EVIDENCE), submodules)
        self.assertLessEqual(set(MODULE.FEATURE_EVIDENCE), ids)

    def test_preserve_reviewed_names(self) -> None:
        import tempfile

        rows = [ngram(299, 1, "view@android/home/payment/my_cards", 30), ngram(299, 2, "view@android/home/payment/my_cards view@HOME", 20)]
        _, mappings = MODULE.analyse(LEAVES, rows, 2, 0.15, 0.60)
        with tempfile.TemporaryDirectory() as directory:
            named = Path(directory) / "android_named_clusters.csv"
            named.write_text(
                "platform,cluster_id,taxonomy_id,cluster_name,business_family,business_submodule,"
                "business_detail,naming_confidence,naming_source,needs_review\n"
                "android,299,M10.05.03,Hóa đơn và thanh toán | Ví/thẻ liên kết | Liên kết thẻ nội địa,"
                "Hóa đơn và thanh toán,Ví/thẻ liên kết,Liên kết thẻ nội địa,high,business_review,false\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.preserve_reviewed_names(mappings, [named]), 1)
        self.assertEqual(mappings[0]["taxonomy_id"], "M10.05.03")
        self.assertEqual(mappings[0]["naming_source"], "business_review")


if __name__ == "__main__":
    unittest.main()
