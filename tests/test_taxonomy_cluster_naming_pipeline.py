from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "output/scores/taxonomy_cluster_naming_pipeline.py"
SPEC = importlib.util.spec_from_file_location("taxonomy_cluster_naming_pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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


if __name__ == "__main__":
    unittest.main()
