#!/usr/bin/env python3
"""Run taxonomy-based cluster naming for one Android/iOS scoring run.

The score run and training run are expected to share the same relative path,
for example::

    output/scores/678/678_20260922_105837
    output/partitioned_runs/678/678_20260922_105837

This wrapper keeps the path discovery and output naming in one place while
delegating the actual evidence-based naming to
``output/scores/taxonomy_cluster_naming_pipeline.py``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_RUN = REPO_ROOT / "output/scores/678/678_20260922_105837"
DEFAULT_TAXONOMY = (
    REPO_ROOT / "output/scores/taxonomy_naming/hifpt_journey_taxonomy_3_levels.csv"
)
PIPELINE = REPO_ROOT / "output/scores/taxonomy_cluster_naming_pipeline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "score_run", nargs="?", type=Path, default=DEFAULT_SCORE_RUN,
        help=f"scoring run directory (default: {DEFAULT_SCORE_RUN})",
    )
    parser.add_argument(
        "--training-run", type=Path,
        help="matching training run; inferred from output/scores when omitted",
    )
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument(
        "--output-dir", type=Path,
        help="audit/mapping output directory (default: SCORE_RUN/taxonomy_naming)",
    )
    parser.add_argument("--min-evidence-rows", type=int, default=2)
    parser.add_argument("--min-mass-coverage", type=float, default=0.15)
    parser.add_argument("--min-score-share", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate inputs and print the pipeline command without running it",
    )
    return parser.parse_args()


def infer_training_run(score_run: Path) -> Path:
    scores_root = (REPO_ROOT / "output/scores").resolve()
    try:
        relative = score_run.resolve().relative_to(scores_root)
    except ValueError as exc:
        raise ValueError(
            "cannot infer --training-run because score_run is outside output/scores"
        ) from exc
    return REPO_ROOT / "output/partitioned_runs" / relative


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def build_command(args: argparse.Namespace) -> list[str]:
    score_run = args.score_run.resolve()
    training_run = (
        args.training_run.resolve()
        if args.training_run else infer_training_run(score_run).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir else score_run / "taxonomy_naming"
    )

    command = [
        sys.executable, str(require_file(PIPELINE, "taxonomy naming pipeline")),
        "--taxonomy", str(require_file(args.taxonomy, "taxonomy")),
        "--android-ngrams", str(require_file(
            training_run / "android/android_cluster_ngrams.csv", "Android n-grams"
        )),
        "--ios-ngrams", str(require_file(
            training_run / "ios/ios_cluster_ngrams.csv", "iOS n-grams"
        )),
        "--output-dir", str(output_dir),
        "--android-input", str(require_file(
            score_run / "android/android_scores.csv", "Android scores"
        )),
        "--android-output", str(score_run / "android/android_scores_taxonomy_named.csv"),
        "--android-catalog-output", str(
            score_run / "android/android_taxonomy_shareholder_catalog.json"
        ),
        "--ios-input", str(require_file(
            score_run / "ios/ios_scores.csv", "iOS scores"
        )),
        "--ios-output", str(score_run / "ios/ios_scores_taxonomy_named.csv"),
        "--ios-catalog-output", str(
            score_run / "ios/ios_taxonomy_shareholder_catalog.json"
        ),
        "--min-evidence-rows", str(args.min_evidence_rows),
        "--min-mass-coverage", str(args.min_mass_coverage),
        "--min-score-share", str(args.min_score_share),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = parse_args()
    try:
        command = build_command(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(shlex.join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
