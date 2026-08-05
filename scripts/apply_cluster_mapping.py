"""Attach business class names to a catalog or newly scored journeys.

Examples:
    python scripts/apply_cluster_mapping.py
    python scripts/apply_cluster_mapping.py --input output/android_scored.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based.cluster_mapping import apply_cluster_mapping, load_cluster_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="android", choices=["android", "ios"])
    parser.add_argument("--input", help="default: output/<platform>_cluster_catalog.json")
    parser.add_argument("--mapping", help="default: output/<platform>_cluster_class_mapping.json")
    parser.add_argument("--output", help="default: <input stem>_named.csv")
    parser.add_argument("--mapping-csv", help="optional flat CSV export of the mapping")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = REPO_ROOT / (args.input or f"output/{args.platform}_cluster_catalog.json")
    mapping_path = REPO_ROOT / (
        args.mapping or f"output/{args.platform}_cluster_class_mapping.json"
    )
    output = REPO_ROOT / args.output if args.output else source.with_name(f"{source.stem}_named.csv")
    frame = pd.read_json(source) if source.suffix.lower() == ".json" else pd.read_csv(source)
    mapping = load_cluster_mapping(mapping_path)
    named = apply_cluster_mapping(frame, mapping)
    output.parent.mkdir(parents=True, exist_ok=True)
    # BOM keeps Vietnamese labels intact when a CSV is opened directly in Excel.
    named.to_csv(output, index=False, encoding="utf-8-sig")
    if args.mapping_csv:
        mapping_csv = REPO_ROOT / args.mapping_csv
        mapping_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(mapping.values()).to_csv(mapping_csv, index=False, encoding="utf-8-sig")
        print(f"mapping table -> {mapping_csv}")
    print(f"mapped {len(named):,} rows with {len(mapping):,} class definitions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
