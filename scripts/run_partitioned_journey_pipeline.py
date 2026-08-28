"""Run the existing journey logic over session-safe parquet partitions.

Preparation writes one journey parquet for each raw parquet partition.  Training
then reads the journey partitions as one logical dataset and fits one global
model per platform; it never fits a separate model per storage partition.

Examples:

    python scripts/run_partitioned_journey_pipeline.py \
        --input data/lake/raw_events \
        --journeys-root data/lake/journeys \
        --output-root output/partitioned_runs \
        --mode all

    python scripts/run_partitioned_journey_pipeline.py \
        --input data/lake/raw_events \
        --journeys-root data/lake/journeys \
        --mode train \
        --max-training-journeys 500000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PipelineConfig  # noqa: E402
from src.experiment import split_sessions_chronologically  # noqa: E402
from src.large_data import (  # noqa: E402
    fit_global_journey_model,
    parquet_files,
    prepare_event_partition,
    read_journey_partitions,
    safe_partition_id,
    write_parquet,
)
from src import tokens as T  # noqa: E402

LOGGER = logging.getLogger("partitioned_journey_pipeline")


def configure_logging(args: argparse.Namespace) -> None:
    """Configure console logging and, optionally, a persistent log file."""
    level = getattr(logging, args.log_level.upper())
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        log_path = args.log_file if args.log_file.is_absolute() else REPO_ROOT / args.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def parse_channel_weights(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in (part.strip() for part in spec.split(",")):
        if not item:
            continue
        name, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"expected name=weight, got {item!r}")
        name = name.strip().lower()
        if name not in T.CHANNEL_COLUMNS:
            raise ValueError(f"unknown channel {name!r}")
        value = float(raw)
        if value < 0:
            raise ValueError(f"channel {name!r} weight cannot be negative")
        if value:
            weights[name] = value
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default=Path("data/lake/raw_events"),
        type=Path,
        help="raw parquet dataset (default: data/lake/raw_events)",
    )
    parser.add_argument("--journeys-root", type=Path, default=Path("data/lake/journeys"))
    parser.add_argument("--output-root", type=Path, default=Path("output/partitioned_runs"))
    parser.add_argument("--run-name", default="latest", help="model run folder name")
    parser.add_argument("--mode", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--platforms", nargs="+", choices=["android", "ios"], default=["android", "ios"])
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="optional log file; relative paths are resolved from the repository root",
    )
    parser.add_argument("--max-training-journeys", type=int, help="deterministic reservoir cap; default uses all")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2, help="latest complete-session holdout share; 0 disables holdout")
    parser.add_argument("--level", default="EXACT", choices=["EXACT", "L3", "L2", "L1"])
    parser.add_argument("--backoff-level", default="L2", choices=["EXACT", "L3", "L2", "L1"])
    parser.add_argument("--channel-weights", default="coarse=0.45,intent=0.30,operation=0.20")
    parser.add_argument("--idle-gap", type=float, default=90.0)
    parser.add_argument("--min-journey-length", type=int, default=4)
    parser.add_argument("--token-min-df", type=int, default=3)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--feature-min-df", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--svd-dim", type=int, default=48)
    parser.add_argument("--global-pca-components", type=int, default=48)
    parser.add_argument("--pca-whiten", action="store_true")
    parser.add_argument("--numeric-weight", type=float, default=0.35)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"])
    parser.add_argument("--algorithm", default="auto", choices=["auto", "ball_tree", "kd_tree", "brute"])
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--keep-chrome", action="store_true")
    parser.add_argument("--drop-boot", action="store_true", default=True)
    parser.add_argument("--keep-boot", action="store_false", dest="drop_boot", help="retain boot/splash events")
    args = parser.parse_args()
    if args.max_training_journeys is not None and args.max_training_journeys < 1:
        parser.error("--max-training-journeys must be positive")
    if not 0 <= args.test_size < 1:
        parser.error("--test-size must be 0 or strictly less than 1")
    if args.min_journey_length < 1 or args.min_cluster_size < 1 or args.min_samples < 1:
        parser.error("journey and clustering sizes must be positive")
    if args.global_pca_components is not None and args.global_pca_components < 1:
        parser.error("--global-pca-components must be positive")
    if args.ngram_min < 1 or args.ngram_min > args.ngram_max:
        parser.error("invalid ngram range")
    try:
        args.channel_weights = parse_channel_weights(args.channel_weights)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.tokens.level = args.level
    cfg.tokens.backoff_level = args.backoff_level
    cfg.tokens.min_journey_df = args.token_min_df
    cfg.segment.idle_gap_seconds = args.idle_gap
    cfg.segment.min_journey_length = args.min_journey_length
    cfg.segment.use_entropy_boundaries = args.entropy
    cfg.features.ngram_range = (args.ngram_min, args.ngram_max)
    cfg.features.min_df = args.feature_min_df
    cfg.features.max_features = args.max_features
    cfg.features.svd_components = args.svd_dim
    cfg.features.global_pca_components = args.global_pca_components
    cfg.features.global_pca_whiten = args.pca_whiten
    cfg.features.numeric_block_weight = args.numeric_weight
    cfg.features.channel_weights = args.channel_weights
    cfg.cluster.min_cluster_size = args.min_cluster_size
    cfg.cluster.min_samples = args.min_samples
    cfg.cluster.cluster_selection_method = args.cluster_selection_method
    cfg.cluster.algorithm = args.algorithm
    cfg.post.drop_chrome = not args.keep_chrome
    cfg.post.drop_boot = args.drop_boot
    return cfg


def partition_platform(path: Path) -> str:
    for part in path.parts:
        if part.startswith("platform="):
            return part.split("=", 1)[1].lower()
    frame = pd.read_parquet(path, columns=["_partition_platform"])
    return str(frame["_partition_platform"].dropna().iloc[0]).lower()


def prepare(args: argparse.Namespace, cfg: PipelineConfig) -> dict[str, int]:
    input_root = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    journeys_root = args.journeys_root if args.journeys_root.is_absolute() else REPO_ROOT / args.journeys_root
    raw_paths = parquet_files(input_root)
    counts: dict[str, int] = defaultdict(int)

    # Loop over all raw parquet partitions and write one journey parquet per partition.
    for raw_path in raw_paths:
        platform = partition_platform(raw_path)
        if platform not in args.platforms:
            continue
        relative = raw_path.relative_to(input_root) if input_root.is_dir() else raw_path.name
        partition_id = str(relative.with_suffix(""))
        target = journeys_root / f"platform={platform}" / f"{safe_partition_id(partition_id)}.parquet"
        if target.exists():
            print(f"skip existing journey partition: {target}")
            continue
        raw = pd.read_parquet(raw_path)
        prepared = prepare_event_partition(
            raw,
            partition_id=partition_id,
            platform=platform,
            cfg=cfg,
        )
        # Write journeys parquet.
        write_parquet(prepared.journeys, target)
        counts[platform] += len(prepared.journeys)
        print(
            f"{platform}: {raw_path.name} -> {target.name}; "
            f"events={len(raw):,}, journeys={len(prepared.journeys):,}, "
            f"eligible={int(prepared.journeys['model_eligible'].sum()):,}"
        )
    manifest = {
        "schema_version": "journeys-parquet-v1",
        "input_root": str(input_root),
        "journeys_root": str(journeys_root),
        "platforms": args.platforms,
        "journeys_by_platform": dict(counts),
        "config": cfg.to_dict(),
    }
    journeys_root.mkdir(parents=True, exist_ok=True)
    (journeys_root / "journey_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return counts


def split_holdout(frame: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object]]:
    # Split the journeys frame into a training set and a holdout set based on session_id and start_ts.
    if test_size == 0:
        return frame, None, {"strategy": "none", "train_journeys": len(frame), "test_journeys": 0}
    required = {"session_id", "start_ts"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cannot create holdout; missing columns: {sorted(missing)}")
    session_frame = frame[["session_id", "start_ts"]].rename(columns={"start_ts": "event_time"})
    train_sessions, test_sessions, report = split_sessions_chronologically(
        session_frame, test_size=test_size
    )
    train_ids = set(train_sessions["session_id"])
    test_ids = set(test_sessions["session_id"])
    train = frame.loc[frame["session_id"].isin(train_ids)].reset_index(drop=True)
    holdout = frame.loc[frame["session_id"].isin(test_ids)].reset_index(drop=True)
    report["train_journeys"] = len(train)
    report["test_journeys"] = len(holdout)
    return train, holdout, report


def train(args: argparse.Namespace, cfg: PipelineConfig) -> None:
    journeys_root = args.journeys_root if args.journeys_root.is_absolute() else REPO_ROOT / args.journeys_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    started = time.perf_counter()
    LOGGER.info(
        "starting partitioned training: journeys_root=%s output_root=%s platforms=%s "
        "test_size=%.3f max_training_journeys=%s",
        journeys_root,
        output_root,
        ",".join(args.platforms),
        args.test_size,
        args.max_training_journeys if args.max_training_journeys is not None else "all",
    )
    all_paths = parquet_files(journeys_root)
    by_platform: dict[str, list[Path]] = defaultdict(list)
    for path in all_paths:
        platform = partition_platform(path)
        if platform in args.platforms:
            by_platform[platform].append(path)
    LOGGER.info(
        "discovered %d journey parquet partitions for training (%s)",
        sum(len(paths) for paths in by_platform.values()),
        ", ".join(f"{platform}={len(by_platform.get(platform, []))}" for platform in args.platforms),
    )

    for platform in args.platforms:
        paths = by_platform.get(platform, [])
        if not paths:
            LOGGER.warning("%s: no journey partitions found; skipping", platform)
            continue
        platform_started = time.perf_counter()
        LOGGER.info(
            "%s: loading eligible journeys from %d partitions%s",
            platform,
            len(paths),
            f" (reservoir cap={args.max_training_journeys:,})"
            if args.max_training_journeys is not None
            else "",
        )
        frame = read_journey_partitions(
            paths,
            platform=platform,
            max_journeys=args.max_training_journeys,
            seed=args.sample_seed,
        )
        LOGGER.info(
            "%s: loaded %d eligible journeys from %d partitions",
            platform,
            len(frame),
            len(paths),
        )
        train_frame, holdout, split_report = split_holdout(frame, args.test_size)
        LOGGER.info(
            "%s: session split complete: train_journeys=%d holdout_journeys=%d "
            "train_sessions=%s holdout_sessions=%s",
            platform,
            len(train_frame),
            len(holdout) if holdout is not None else 0,
            split_report.get("train_sessions", "unknown"),
            split_report.get("test_sessions", "unknown"),
        )
        run_dir = output_root / args.run_name / platform
        manifest = {
            "platform": platform,
            "journey_partitions": [str(path) for path in paths],
            "selected_journeys": len(frame),
            "train_journeys": len(train_frame),
            "holdout_journeys": len(holdout) if holdout is not None else 0,
            "max_training_journeys": args.max_training_journeys,
            "sample_seed": args.sample_seed,
            "split": split_report,
        }
        LOGGER.info(
            "%s: fitting one global model on %d training journeys; output=%s",
            platform,
            len(train_frame),
            run_dir,
        )
        result = fit_global_journey_model(
            train_frame,
            platform=platform,
            cfg=cfg,
            output_dir=run_dir,
            holdout=holdout,
            training_manifest=manifest,
        )
        LOGGER.info(
            "%s: training complete: journeys=%d clusters=%d noise=%.1f%% "
            "elapsed=%.1fs run=%s",
            platform,
            len(train_frame),
            result["hdbscan"].get("n_clusters", 0),
            result["hdbscan"].get("noise_share", 0) * 100,
            time.perf_counter() - platform_started,
            run_dir,
        )
    LOGGER.info("partitioned training complete in %.1fs", time.perf_counter() - started)


def main() -> int:
    args = parse_args()
    configure_logging(args)
    cfg = build_config(args)
    LOGGER.info("starting partitioned journey pipeline: mode=%s", args.mode)
    if args.mode in {"prepare", "all"}:
        prepare(args, cfg)
    if args.mode in {"train", "all"}:
        train(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
