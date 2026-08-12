"""Attach names from the shareholder cluster catalog to scored journeys.

    python scripts/apply_cluster_name_mapping.py --platform android --input output\test\test_data_android_android_scored.csv --catalog output\journey_runs\EXACT_ch-c45i30o20_ng1-3_svd64_fdf3_mf20000_nw0p35_mcs100_ms5_sel-eom_gap90_jmin4_tdf3_ent0_chr1_boot0_test0p2\shareholder_cluster_catalog_vi.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based.cluster_mapping import apply_cluster_name_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=["android", "ios"])
    parser.add_argument("--input", required=True, help="scored journeys as CSV")
    parser.add_argument(
        "--catalog",
        default="output/clusters_with_screen/shareholder_cluster_catalog_vi.json",
        help="shareholder catalog JSON",
    )
    parser.add_argument("--output", help="default: <input stem>_named.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    catalog_path = Path(args.catalog)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not catalog_path.is_absolute():
        catalog_path = REPO_ROOT / catalog_path

    frame = pd.read_csv(input_path, low_memory=False)
    named = apply_cluster_name_mapping(frame, catalog_path, platform=args.platform)
    output = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_named.csv")
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    named.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"mapped {len(named):,} rows with shareholder names -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
