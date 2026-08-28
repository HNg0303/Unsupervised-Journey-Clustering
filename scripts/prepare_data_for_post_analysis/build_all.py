#!/usr/bin/env python3
"""Prepare the independent data products consumed by dashboard and HTML.

By default this command only registers existing train/inference/post-analysis
outputs.  Raw EDA is opt-in because it is the expensive source-data scan the
user may want to run separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("output/scores/pca48_ngrams12_500"))
    parser.add_argument("--train-run", type=Path, default=Path("output/partitioned_runs/pca48_svd48_ngrams12_500k/latest"))
    parser.add_argument("--android", type=Path, help="named Android inference CSV for summary regeneration")
    parser.add_argument("--ios", type=Path, help="named iOS inference CSV for summary regeneration")
    parser.add_argument("--mapping", type=Path, help="authoritative Cluster_naming.csv")
    parser.add_argument("--run-eda", action="store_true", help="run the raw EDA scan; omit for registration-only")
    parser.add_argument("--eda-max-rows", type=int, help="pass a row cap to EDA for smoke testing")
    parser.add_argument("--build-html", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(command: list[str]) -> None:
    print("$", " ".join(str(item) for item in command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    bundle = resolve(args.bundle)
    train_run = resolve(args.train_run)
    if not bundle.exists():
        raise SystemExit(f"bundle not found: {bundle}")
    if not train_run.exists():
        raise SystemExit(f"train run not found: {train_run}")

    post_dir = bundle / "post_analysis"
    post_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        str(ROOT / "scripts/prepare_data_for_post_analysis/train_output.py"),
        "--train-run", str(train_run),
        "--output", str(post_dir / "train_output_summary.json"),
    ])

    if args.run_eda:
        eda_command = [
            sys.executable,
            str(ROOT / "scripts/prepare_data_for_post_analysis/eda_raw.py"),
            "--output-dir", str(post_dir / "eda"),
        ]
        if args.eda_max_rows is not None:
            eda_command.extend(["--max-rows", str(args.eda_max_rows)])
        run(eda_command)

    summary_dir = bundle / "html_dashboard_summary"
    if args.android and args.ios:
        mapping = resolve(args.mapping) if args.mapping else bundle / "Cluster_naming.csv"
        run([
            sys.executable,
            str(ROOT / "scripts/prepare_data_for_post_analysis/inference_summary.py"),
            "--android", str(resolve(args.android)),
            "--ios", str(resolve(args.ios)),
            "--naming", str(mapping),
            "--output-dir", str(summary_dir),
        ])

    if args.build_html:
        run([
            sys.executable,
            str(ROOT / "scripts/prepare_data_for_post_analysis/html_dashboard.py"),
            "--input-dir", str(summary_dir),
            "--output", str(summary_dir / "index.html"),
        ])

    contract = {
        "schema_version": "post-analysis-preparation-v1",
        "train_run": str(train_run),
        "inference_bundle": str(bundle),
        "raw_eda": str(post_dir / "eda" / "eda_summary.json"),
        "train_output": str(post_dir / "train_output_summary.json"),
        "inference_summary": str(summary_dir),
        "customer_analysis": str(bundle / "shareholder_analysis"),
        "named_anchor": "*_all_named.csv",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Raw EDA is absent until --run-eda is explicitly used.",
            "Full inference aggregates are read from html_dashboard_summary, never from journey_examples.csv.",
            "Customer and business post-analysis starts from all_named.csv.",
        ],
    }
    (post_dir / "preparation_manifest.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote preparation contract -> {post_dir / 'preparation_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
