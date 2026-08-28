from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.apply_cluster_business import apply_csv, apply_input, load_mapping


MAPPING_COLUMNS = [
    "platform", "cluster_id", "cluster_name", "business_family",
    "business_submodule", "business_detail", "canonical_level_2_code",
    "naming_confidence",
]


class ApplyClusterBusinessTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.mapping_path = self.root / "mapping.csv"
        rows = []
        for platform in ("android", "ios"):
            rows.extend([
                {
                    "platform": platform,
                    "cluster_id": -1,
                    "cluster_name": "Chưa phân loại | Journey hỗn hợp/nhiễu",
                    "business_family": "Chưa phân loại",
                    "business_submodule": "Journey hỗn hợp/nhiễu",
                    "business_detail": "",
                    "canonical_level_2_code": "unclassified.dynamic",
                    "naming_confidence": "low",
                },
                {
                    "platform": platform,
                    "cluster_id": 7,
                    "cluster_name": "Thanh toán | Lịch sử thanh toán",
                    "business_family": "Thanh toán",
                    "business_submodule": "Lịch sử thanh toán",
                    "business_detail": "",
                    "canonical_level_2_code": "payment.history",
                    "naming_confidence": "high",
                },
            ])
        self._write(self.mapping_path, MAPPING_COLUMNS, rows)

    def tearDown(self):
        self.directory.cleanup()

    def test_applies_platform_cluster_mapping_and_preserves_raw_columns(self):
        source = self.root / "raw.csv"
        output = self.root / "named.csv"
        catalog = self.root / "shareholder_cluster_catalog.json"
        self._write(source, ["journey_id", "cluster", "distance"], [
            {"journey_id": "j1", "cluster": 7, "distance": 0.2},
            {"journey_id": "j2", "cluster": -1, "distance": 1.4},
        ])

        summary = apply_csv(
            source,
            output,
            load_mapping(self.mapping_path),
            "android",
            catalog_output=catalog,
        )
        with output.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(rows[0]["cluster_id"], "7")
        self.assertEqual(rows[0]["business_family"], "Thanh toán")
        self.assertEqual(rows[0]["distance"], "0.2")
        self.assertEqual(rows[1]["business_family"], "Chưa phân loại")
        self.assertEqual(summary["shareholder_catalog"], str(catalog))

        payload = json.loads(catalog.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "shareholder-cluster-catalog-business-v1")
        self.assertEqual(payload["platforms"][0]["platform"], "android")
        catalog_rows = payload["platforms"][0]["clusters"]
        self.assertEqual([row["cluster_id"] for row in catalog_rows], [-1, 7])
        self.assertEqual([row["row_count_in_input"] for row in catalog_rows], [1, 1])
        self.assertEqual(catalog_rows[1]["mapping_name"], "Thanh toán | Lịch sử thanh toán")
        self.assertNotIn("function_code", catalog_rows[1])

    def test_rejects_cluster_absent_from_platform_mapping(self):
        source = self.root / "raw.csv"
        self._write(source, ["cluster"], [{"cluster": 999}])
        with self.assertRaisesRegex(ValueError, "absent from mapping"):
            apply_csv(source, self.root / "named.csv", load_mapping(self.mapping_path), "ios")

    def test_applies_mapping_to_parquet_input_in_batches(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")

        source = self.root / "raw.parquet"
        output = self.root / "named_from_parquet.csv"
        catalog = self.root / "catalog_from_parquet.json"
        pq.write_table(
            pa.table({
                "journey_id": ["j1", "j2", "j3"],
                "cluster": [7, -1, 7],
                "distance": [0.2, 1.4, 0.3],
            }),
            source,
        )

        summary = apply_input(
            source,
            output,
            load_mapping(self.mapping_path),
            "ios",
            catalog_output=catalog,
            parquet_batch_size=1,
        )
        with output.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        payload = json.loads(catalog.read_text(encoding="utf-8"))
        catalog_rows = payload["platforms"][0]["clusters"]

        self.assertEqual(summary["input_format"], "parquet")
        self.assertEqual([row["cluster_id"] for row in rows], ["7", "-1", "7"])
        self.assertEqual([row["business_family"] for row in rows], ["Thanh toán", "Chưa phân loại", "Thanh toán"])
        self.assertEqual([row["row_count_in_input"] for row in catalog_rows], [1, 2])

    def test_rejects_blank_non_noise_business_name(self):
        bad = self.root / "bad.csv"
        rows = []
        for platform in ("android", "ios"):
            rows.append({
                "platform": platform,
                "cluster_id": -1,
                "cluster_name": "noise",
                "business_family": "Chưa phân loại",
                "business_submodule": "noise",
                "business_detail": "",
                "canonical_level_2_code": "unclassified.dynamic",
                "naming_confidence": "low",
            })
        rows.append({
            "platform": "android",
            "cluster_id": 3,
            "cluster_name": "",
            "business_family": "",
            "business_submodule": "",
            "business_detail": "",
            "canonical_level_2_code": "x.y",
            "naming_confidence": "low",
        })
        self._write(bad, MAPPING_COLUMNS, rows)
        with self.assertRaisesRegex(ValueError, "empty columns"):
            load_mapping(bad)

    @staticmethod
    def _write(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
