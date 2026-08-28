#!/usr/bin/env python3
"""Run the independent customer, focus, VNeID and Loyalty analyses in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android", type=Path, required=True)
    parser.add_argument("--ios", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-focus", action="store_true")
    parser.add_argument("--skip-vneid", action="store_true")
    parser.add_argument("--skip-loyalty", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(script: str, args: list[str]) -> None:
    command = [sys.executable, str(ROOT / script), *args]
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    android, ios, output = resolve(args.android), resolve(args.ios), resolve(args.output_dir)
    common = ["--android", str(android), "--ios", str(ios), "--output-dir", str(output), "--memory-limit", args.memory_limit, "--threads", str(args.threads)]
    run("scripts/post_analysis/analyze_customer_metrics.py", common)
    if not args.skip_focus:
        run("scripts/post_analysis/analyze_focus_journeys.py", common)
    if not args.skip_vneid:
        run("scripts/post_analysis/analyze_post_vneid_econtract.py", ["--android", str(android), "--ios", str(ios), "--output-dir", str(output)])
    if not args.skip_loyalty:
        run("scripts/post_analysis/analyze_loyalty.py", common)
    print(f"completed independent post-analysis outputs -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
