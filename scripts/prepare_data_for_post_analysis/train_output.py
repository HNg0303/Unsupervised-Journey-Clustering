#!/usr/bin/env python3
"""Export compact metadata for the canonical partitioned training output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", type=Path, default=Path("output/partitioned_runs/pca48_svd48_ngrams12_500k/latest"))
    parser.add_argument("--output", type=Path, default=Path("output/scores/pca48_ngrams12_500/post_analysis/train_output_summary.json"))
    args = parser.parse_args()
    train_run = args.train_run if args.train_run.is_absolute() else ROOT / args.train_run
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not train_run.exists():
        raise SystemExit(f"train run not found: {train_run}")

    platforms = {}
    for platform in ("android", "ios"):
        cfg_path = train_run / platform / f"{platform}_run_config.json"
        catalog_path = train_run / platform / f"{platform}_cluster_catalog.json"
        if not cfg_path.exists():
            continue
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []
        platforms[platform] = {
            "run_config": config,
            "cluster_catalog": str(catalog_path),
            "cluster_catalog_rows": len(catalog),
        }
    payload = {
        "schema_version": "train-output-summary-v1",
        "source": str(train_run),
        "platforms": platforms,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote train output summary -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
