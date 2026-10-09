"""Score session-safe raw parquet partitions with a frozen journey model.

Examples:

    journey-infer \
        --input data/lake/raw_events \
        --platform android \
        --model-run output/partitioned_runs/latest/android \
        --output-root output/scores

    journey-infer \
        --input data/giga_data/android_events_t3-2026.csv \
        --platform android \
        --model-run output/partitioned_runs/latest/android \
        --output-root output/scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..storage import parquet_files, safe_partition_id, write_parquet
from ..preprocessing import normalize_platform, normalize_production_columns
from ..score import JourneyScorer


# Row schema of the scored journey output, in the column order of the
# `journey_summary` database table. Cluster names are not repeated per row: they
# live in the per-cluster taxonomy table joined on (model_version, platform,
# cluster_id). Columns that duplicate or can be derived from these are left out:
# os (= platform), span_seconds (= end_ts - start_ts), n_unique_tokens
# (from revisit_ratio), n_dropped_screens/n_dedup_removed (diagnostics),
# nearest_cluster/distance_limit (model internals), source_partition (prefix of
# journey_id) and the effective_* copies of the single-model scores.
OUTPUT_COLUMNS = (
    "model_version", "platform", "cluster_id", "journey_id", "session_id",
    "device_id", "customer_id", "start_ts", "end_ts", "boundary_reason",
    "n_events_raw", "n_events_final", "n_loop_removed",
    "action_ratio", "back_rate", "revisit_ratio",
    "median_gap_s", "p90_gap_s", "max_gap_s",
    "entry_token", "exit_token", "sequence",
    "distance_to_centroid", "markov_logprob",
    "geometric_anomaly", "generative_anomaly", "severe_anomaly",
    "friction_flags", "next_action", "next_action_share", "scored_at",
)
# Already expressed by the geometric_anomaly / generative_anomaly columns.
MODEL_FLAGS = frozenset({"unknown_archetype", "improbable_transitions"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="raw CSV file for direct scoring or session-safe parquet dataset",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="also write direct CSV scores as parquet; large parquet inputs always stay parquet",
    )
    parser.add_argument("--platform", required=True, choices=["android", "ios"])
    parser.add_argument(
        "--model-run", required=True, type=Path,
        help="training output folder containing <platform>_journey_scorer.pkl",
    )
    parser.add_argument("--model-version", help="default: fitted run folder name")
    parser.add_argument("--output-root", type=Path, default=Path("output/scores"))
    parser.add_argument("--output", type=Path, help="direct output file for CSV input; overrides --output-root")
    parser.add_argument("--resume", action="store_true", help="skip result partitions that already exist")
    parser.add_argument("--max-files", type=int, help="optional smoke-test limit")
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")
    if args.output and args.max_files:
        parser.error("--output cannot be combined with --max-files")
    return args


def resolve(path: Path) -> Path:
    return path.resolve()


def partition_platform(path: Path) -> str:
    for part in path.parts:
        if part.startswith("platform="):
            return part.split("=", 1)[1].lower()
    frame = pd.read_parquet(path, columns=["_partition_platform"])
    return str(frame["_partition_platform"].dropna().iloc[0]).lower()


def read_csv_input(path: Path, platform: str) -> pd.DataFrame:
    """Read and platform-filter one small CSV without creating raw parquet."""
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError(f"CSV input must be a .csv file: {path}")
    raw = normalize_production_columns(pd.read_csv(path, low_memory=False))
    if "platform" not in raw.columns:
        raise ValueError(f"{path} is missing platform/segmentation.segment")
    normalized = normalize_platform(raw["platform"])
    filtered = raw.loc[normalized.eq(platform)].copy()
    if filtered.empty:
        raise ValueError(f"no {platform} rows found in CSV input: {path}")
    return filtered


def score_raw_partition(
    raw: pd.DataFrame,
    *,
    scorer: JourneyScorer,
    platform: str,
    model_version: str,
    partition_id: str,
) -> pd.DataFrame:
    """Prepare and score one raw input while keeping partition metadata aligned."""
    journeys, sequences, channels = scorer.prepare(raw)
    scored = scorer.score_prepared(journeys, sequences, channels=channels)
    if not scored.empty:
        prefix = safe_partition_id(partition_id)
        scored["journey_id"] = [f"{prefix}::{value}" for value in scored["journey_id"]]
        scored["platform"] = platform
        scored["model_version"] = model_version
        scored["scored_at"] = pd.Timestamp.now(tz="UTC")
    return to_output_schema(scored)


def to_output_schema(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep only the published columns, in `journey_summary` order."""
    if scored.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    out = scored.rename(columns={"cluster": "cluster_id"})
    out["friction_flags"] = out["friction_flags"].fillna("").map(
        lambda value: "|".join(
            flag for flag in str(value).split("|") if flag and flag not in MODEL_FLAGS
        )
    )
    return out.loc[:, list(OUTPUT_COLUMNS)]


def main() -> int:
    args = parse_args()
    input_root = resolve(args.input)
    output_root = resolve(args.output_root)
    run_dir = resolve(args.model_run)
    scorer = JourneyScorer.load(run_dir / f"{args.platform}_journey_scorer.pkl")
    default_version = run_dir.parent.name if run_dir.name == args.platform else run_dir.name
    model_version = args.model_version or default_version

    model_version = safe_partition_id(model_version)
    # Keep inference grouped by platform first, matching the canonical bundle layout:
    # output/scores/<run>/<platform>/model_version=<version>/platform=<platform>/...
    result_root = output_root / args.platform / f"model_version={model_version}" / f"platform={args.platform}"
    result_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    total_journeys = 0

    if input_root.is_file() and input_root.suffix.lower() == ".csv":
        partition_id = input_root.stem
        suffix = ".parquet" if args.parquet else ".csv"
        target = resolve(args.output) if args.output else result_root / f"{safe_partition_id(partition_id)}{suffix}"
        if args.output and args.parquet and target.suffix.lower() != ".parquet":
            raise SystemExit("--output must end in .parquet when --parquet is used")
        if args.output and not args.parquet and target.suffix.lower() != ".csv":
            raise SystemExit("--output must end in .csv for direct CSV scoring")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if args.resume:
                print(f"resume: skip {target.name}")
                return 0
            raise SystemExit(f"result already exists: {target}; use --resume or a new model version")

        raw = read_csv_input(input_root, args.platform)
        scored = score_raw_partition(
            raw,
            scorer=scorer,
            platform=args.platform,
            model_version=model_version,
            partition_id=partition_id,
        )
        if args.parquet:
            write_parquet(scored, target)
        else:
            scored.to_csv(target, index=False)
        total_journeys = len(scored)
        manifest_rows.append(
            {
                "source_partition": partition_id,
                "source_file": str(input_root),
                "output": str(target),
                "status": "written",
                "events": len(raw),
                "journeys": len(scored),
            }
        )
        print(
            f"{args.platform}: {input_root.name} -> {target.name}; "
            f"events={len(raw):,}, journeys={len(scored):,}"
        )
        manifest = {
            "schema_version": "scored-journeys-csv-v1" if not args.parquet else "scored-journeys-parquet-v1",
            "input_root": str(input_root),
            "output_root": str(result_root),
            "platform": args.platform,
            "model_version": model_version,
            "partitions": manifest_rows,
            "total_journeys": total_journeys,
        }
        manifest_path = target.with_suffix(".manifest.json") if args.output else result_root / "inference_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {total_journeys:,} scored journeys to {result_root}")
        return 0

    raw_paths = [path for path in parquet_files(input_root) if partition_platform(path) == args.platform]
    if args.max_files is not None:
        raw_paths = raw_paths[: args.max_files]
    if not raw_paths:
        raise SystemExit(f"no {args.platform} parquet partitions found below {input_root}")

    for raw_path in raw_paths:
        relative = raw_path.relative_to(input_root) if input_root.is_dir() else Path(raw_path.name)
        partition_id = str(relative.with_suffix(""))
        target = result_root / f"{safe_partition_id(partition_id)}.parquet"
        if target.exists():
            if args.resume:
                print(f"resume: skip {target.name}")
                manifest_rows.append({"source_partition": partition_id, "output": str(target), "status": "skipped_existing"})
                continue
            raise SystemExit(f"result already exists: {target}; use --resume or a new model version")

        raw = pd.read_parquet(raw_path)
        scored = score_raw_partition(
            raw,
            scorer=scorer,
            platform=args.platform,
            model_version=model_version,
            partition_id=partition_id,
        )
        write_parquet(scored, target)
        total_journeys += len(scored)
        row = {
            "source_partition": partition_id,
            "source_file": str(raw_path),
            "output": str(target),
            "status": "written",
            "events": len(raw),
            "journeys": len(scored),
        }
        manifest_rows.append(row)
        print(f"{args.platform}: {raw_path.name} -> {target.name}; events={len(raw):,}, journeys={len(scored):,}")

    manifest = {
        "schema_version": "scored-journeys-parquet-v1",
        "input_root": str(input_root),
        "output_root": str(result_root),
        "platform": args.platform,
        "model_version": model_version,
        "partitions": manifest_rows,
        "total_journeys": total_journeys,
    }
    (result_root / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {total_journeys:,} scored journeys to {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
