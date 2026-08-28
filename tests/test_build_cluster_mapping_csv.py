from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_cluster_mapping_csv import (
    _detail_from_evidence,
    _score_evidence,
    build_mapping,
)


class BuildClusterMappingCsvTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.sitemap = self.root / "sitemap.csv"
        with self.sitemap.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows([
                ["Hi FPT"],
                [""],
                ["Home", "Đăng ký dịch vụ"],
                ["Noti + search + Quét QR", "Gợi ý cho bạn"],
                ["Banner", "Menu dịch vụ"],
            ])
        for platform in ("android", "ios"):
            model = (
                self.root / platform / "model_version=latest"
                / f"platform={platform}" / "model_output"
            )
            model.mkdir(parents=True)
            catalog = [
                {"cluster": -1, "size": 10, "share": 0.1, "medoid_path": "mixed"},
                {
                    "cluster": 0,
                    "size": 90,
                    "share": 0.9,
                    "medoid_path": "view@notification -> action@notification#open",
                },
            ]
            (model / f"{platform}_cluster_catalog.json").write_text(
                json.dumps(catalog), encoding="utf-8"
            )
            with (model / f"{platform}_cluster_ngrams.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["cluster", "rank", "ngram", "lift", "cluster_mass"],
                )
                writer.writeheader()
                for cluster in (-1, 0):
                    for rank in range(1, 9):
                        writer.writerow({
                            "cluster": cluster,
                            "rank": rank,
                            "ngram": (
                                f"view@notification action@notification#open_{rank}"
                                if cluster == 0 else f"mixed_{rank}"
                            ),
                            "lift": 2 + rank,
                            # Rank 8 deliberately has the largest mass.
                            "cluster_mass": 100 if rank == 8 else 10 - rank,
                        })

    def tearDown(self):
        self.directory.cleanup()

    def test_builds_directly_from_evidence_and_orders_ngrams_by_mass(self):
        rows = build_mapping(self.root, self.sitemap)
        self.assertEqual(len(rows), 4)
        mapped = next(
            row for row in rows
            if row["platform"] == "android" and row["cluster_id"] == 0
        )
        self.assertEqual(mapped["canonical_level_2_code"], "engagement.notification")
        self.assertEqual(mapped["cluster_name"], "Home | Noti + search + Quét QR")
        self.assertEqual(mapped["naming_source"], "sitemap_exact")
        self.assertEqual(mapped["needs_review"], "false")
        self.assertGreater(float(mapped["evidence_share"]), 0.0)
        self.assertTrue(mapped["top_ngrams"].startswith("mass=100.0000; rank=8;"))

    def test_dominant_notification_mass_beats_rare_transaction_callback(self):
        ngrams = [
            {
                "rank": "3", "cluster_mass": "75.1421",
                "ngram": "view@HomeVC view@DetailsNotiVC",
            },
            {
                "rank": "2", "cluster_mass": "1.8403",
                "ngram": "action@HomeVC#transaction_result/home view@DetailsNotiVC",
            },
            {
                "rank": "4", "cluster_mass": "3.0434",
                "ngram": "action@HomeVC#Home/Home_pull_to_refresh view@DetailsNotiVC",
            },
            {
                "rank": "5", "cluster_mass": "2.1419",
                "ngram": "action@DetailsNotiVC#adsview view@HomeVC",
            },
        ]
        medoid = (
            "view@HomeVC -> view@DetailsNotiVC -> "
            "action@DetailsNotiVC#view_os_noti -> view@HomeVC"
        )

        scoring = _score_evidence(ngrams, medoid)

        self.assertEqual(scoring["code"], "engagement.notification")
        self.assertNotEqual(scoring["ngram_primary"], "payment.transaction")
        self.assertGreater(scoring["coverage"], 0.9)
        self.assertEqual(
            _detail_from_evidence(
                str(scoring["code"]), str(scoring["confidence"]), ngrams, medoid
            ),
            "Xem chi tiết thông báo",
        )

    def test_non_noise_cluster_without_rule_match_gets_sitemap_style_name(self):
        model = (
            self.root / "android" / "model_version=latest"
            / "platform=android" / "model_output"
        )
        catalog_path = model / "android_cluster_catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog.append({
            "cluster": 1, "size": 5, "share": 0.05,
            "medoid_path": "view@UnknownController -> action@UnknownController#open",
        })
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        with (model / "android_cluster_ngrams.csv").open(
            "a", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["cluster", "rank", "ngram", "lift", "cluster_mass"],
            )
            for rank in range(1, 9):
                writer.writerow({
                    "cluster": 1, "rank": rank,
                    "ngram": f"unknown_controller_{rank} unknown_action_{rank}",
                    "lift": 10, "cluster_mass": 9 - rank,
                })

        rows = build_mapping(self.root, self.sitemap)
        mapped = next(
            row for row in rows
            if row["platform"] == "android" and row["cluster_id"] == 1
        )

        self.assertTrue(mapped["cluster_name"].strip())
        self.assertEqual(mapped["cluster_name"], "Home | Hành trình tổng quát")
        self.assertEqual(mapped["naming_source"], "sitemap_style_approximation")
        self.assertEqual(mapped["business_detail"], "")


if __name__ == "__main__":
    unittest.main()
