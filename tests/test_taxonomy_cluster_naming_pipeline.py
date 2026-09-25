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


if __name__ == "__main__":
    unittest.main()
