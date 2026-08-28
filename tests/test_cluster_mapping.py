from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cluster_mapping import apply_cluster_name_mapping, load_cluster_name_mapping


class ClusterNameMappingTest(unittest.TestCase):
    def test_load_shareholder_catalog_and_apply(self):
        catalog = {
            "platforms": [
                {
                    "platform": "android",
                    "clusters": [
                        {
                            "cluster_id": -1,
                            "cluster_name": "Chưa phân loại",
                            "business_family": "chưa phân loại",
                            "naming_confidence": "not_applicable",
                        },
                        {
                            "cluster_id": 7,
                            "cluster_name": "Thông báo",
                            "business_family": "tương tác",
                            "naming_confidence": "high",
                        },
                    ],
                }
            ]
        }
        path = self._write_catalog(catalog)

        mapping = load_cluster_name_mapping(path, platform="android")
        result = apply_cluster_name_mapping(pd.DataFrame({"cluster": [7, 999, -1]}), mapping)

        self.assertEqual(result["cluster_name"].tolist(), ["Thông báo", "Chưa phân loại", "Chưa phân loại"])
        self.assertEqual(result["business_family"].tolist(), ["tương tác", "chưa phân loại", "chưa phân loại"])
        self.assertEqual(
            result.columns.tolist(), ["cluster", "business_family", "cluster_name", "naming_confidence"]
        )

    def test_catalog_requires_platform_when_multiple_platforms(self):
        path = self._write_catalog(
            {
                "platforms": [
                    {"platform": "android", "clusters": [{"cluster_id": -1, "cluster_name": "a"}]},
                    {"platform": "ios", "clusters": [{"cluster_id": -1, "cluster_name": "i"}]},
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "platform is required"):
            load_cluster_name_mapping(path)

    @staticmethod
    def _write_catalog(catalog):
        import tempfile

        handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        handle.write(json.dumps(catalog, ensure_ascii=False).encode("utf-8"))
        handle.close()
        return handle.name
