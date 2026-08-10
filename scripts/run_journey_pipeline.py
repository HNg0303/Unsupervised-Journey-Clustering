"""End-to-end journey pipeline: canonize -> tokenize -> segment -> cluster.

    python scripts/run_journey_pipeline.py --preprocess
    python scripts/run_journey_pipeline.py --min-cluster-size 50 --min-samples 3 \
        --ngram-min 1 --ngram-max 4 --svd-dim 128

The output directory is generated from the complete experiment configuration.
Canonical train/test data is cached separately and reused across clustering
experiments that share the same preprocessing settings.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based import canonize as C  # noqa: E402
from Rule_based import cluster as CL  # noqa: E402
from Rule_based import features as F  # noqa: E402
from Rule_based import postprocess as P  # noqa: E402
from Rule_based import segment as S  # noqa: E402
from Rule_based import tokens as T  # noqa: E402
from Rule_based.config import PipelineConfig  # noqa: E402
from Rule_based.experiment import (  # noqa: E402
    build_prepared_data_slug,
    build_run_slug,
    split_sessions_chronologically,
)
from Rule_based.score import JourneyScorer  # noqa: E402
from Rule_based.production import read_production_folder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preprocess",
        action="store_true",
        help="force rebuilding the automatically cached canonical train/test split",
    )
    p.add_argument("--input", default="data/train_data/raw_data_production/data_raw_sample")
    p.add_argument(
        "--output-root",
        "--output",
        dest="output_root",
        default="output/journey_runs",
        help="root directory; the run-specific subdirectory is generated from hyperparameters",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="share of latest sessions held out")
    p.add_argument("--level", default="L2", choices=["L1", "L2", "L3"])
    p.add_argument("--idle-gap", type=float, default=90.0)
    p.add_argument("--min-journey-length", type=int, default=4)
    p.add_argument("--token-min-df", type=int, default=3)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=3)
    p.add_argument("--feature-min-df", type=int, default=3)
    p.add_argument("--max-features", type=int, default=20_000)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--numeric-weight", type=float, default=0.35)
    p.add_argument("--min-cluster-size", type=int, default=100)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument(
        "--cluster-selection-method", default="eom", choices=["eom", "leaf"]
    )
    p.add_argument("--entropy", action="store_true", help="add branching-entropy boundaries")
    p.add_argument("--keep-chrome", action="store_true", help="do not drop OS chrome screens")
    p.add_argument("--drop-boot", action="store_true", help="drop boot/splash screens")
    p.add_argument("--no-stability", action="store_true")
    args = p.parse_args()
    if not 0.0 < args.test_size < 1.0:
        p.error("--test-size must be strictly between 0 and 1")
    positive = {
        "--min-journey-length": args.min_journey_length,
        "--token-min-df": args.token_min_df,
        "--ngram-min": args.ngram_min,
        "--ngram-max": args.ngram_max,
        "--feature-min-df": args.feature_min_df,
        "--max-features": args.max_features,
        "--svd-dim": args.svd_dim,
        "--min-cluster-size": args.min_cluster_size,
        "--min-samples": args.min_samples,
    }
    for flag, value in positive.items():
        if value < 1:
            p.error(f"{flag} must be at least 1")
    if args.ngram_min > args.ngram_max:
        p.error("--ngram-min cannot be greater than --ngram-max")
    if args.idle_gap <= 0:
        p.error("--idle-gap must be positive")
    if args.numeric_weight < 0:
        p.error("--numeric-weight cannot be negative")
    return args


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def run_cluster_journey(
    raw: pd.DataFrame,
    platform: str,
    cfg: PipelineConfig,
    out_dir: Path,
    *,
    holdout_raw: pd.DataFrame | None = None,
    check_stability: bool = True,
) -> dict[str, object]:
    """Run the complete, platform-agnostic clustering pipeline."""
    platform = platform.lower().strip()
    prefix = f"{platform}_"
    reports: dict[str, pd.DataFrame] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    banner(f"{platform.upper()} - STAGE 0 - input")
    print(f"rows={len(raw):,}  sessions={raw.session_id.nunique():,}  columns={len(raw.columns)}")

    banner(f"{platform.upper()} - STAGE 1 - canonize")
    canon = C.canonize_events(raw, cfg.canonize)
    reports["canonization"] = pd.DataFrame([{
        "stage": "canonical event_type@segment_name",
        "vocabulary": int(canon.event_token.nunique()),
        "singletons": int((canon.event_token.value_counts() == 1).sum()),
        "rows": int(len(canon)),
    }])
    print(reports["canonization"].to_string(index=False))

    banner(f"{platform.upper()} - STAGE 2 - tokenize")
    tok = T.build_tokens(canon, cfg.canonize)
    for level in ("L1", "L2", "L3"):
        col = {"L1": "token_l1", "L2": "token_l2", "L3": "token_l3"}[level]
        vc = tok[col].value_counts()
        coverage = vc.head(100).sum() / len(tok) if len(tok) else 0.0
        print(
            f"{level}: vocab={len(vc):>5}  singletons={(vc == 1).sum():>4} "
            f"({(vc == 1).mean():.1%})  top100_coverage={coverage:.1%}"
        )
    T.token_dictionary(tok, cfg.tokens.level).to_csv(
        out_dir / f"{prefix}token_dictionary.csv", index=False
    )

    # ------------------------------------------------------------- segment
    banner(f"{platform.upper()} - STAGE 3 - segment sessions into journeys")
    sweep_events = tok if len(tok) <= 200_000 else tok.iloc[:200_000].copy()
    reports["idle_gap_sweep"] = S.sweep_idle_gap(sweep_events, cfg.segment)
    print("idle-gap sensitivity:")
    print(reports["idle_gap_sweep"].to_string(index=False))

    seg = S.assign_journeys(tok, cfg.segment)
    seg = S.refine_with_entropy(seg, cfg.segment)
    reports["segmentation"] = S.segmentation_report(seg, cfg.segment)
    print("\nchosen segmentation:")
    print(reports["segmentation"].to_string(index=False))

    # ---------------------------------------------------------- postprocess
    banner("STAGE 4 - post-tokenization cleanup")
    token_col = {"L1": "token_l1", "L2": "token_l2", "L3": "token_l3"}[cfg.tokens.level]
    journeys, sequences, extras = P.build_journey_sequences(
        seg, cfg.post, token_col=token_col, extra_token_cols=("token_l1",)
    )
    reports["postprocess"] = P.postprocess_report(journeys)
    print(reports["postprocess"].to_string(index=False))

    sequences, fold_stats = T.fold_rare_tokens(sequences, extras["token_l1"], cfg.tokens)
    reports["rare_folding"] = fold_stats
    print("\nrare-token folding:")
    print(fold_stats.to_string(index=False))

    keep = np.array([len(s) >= cfg.segment.min_journey_length for s in sequences])
    print(
        f"\njourneys total={len(journeys):,}  "
        f"kept (len>={cfg.segment.min_journey_length})={int(keep.sum()):,}  "
        f"dropped={int((~keep).sum()):,}"
    )
    journeys_all = journeys.copy()
    journeys = journeys.loc[keep].reset_index(drop=True)
    sequences = [s for s, k in zip(sequences, keep) if k]
    if journeys.empty:
        raise ValueError(f"{platform}: no journeys meet the minimum length")

    # -------------------------------------------------------------- features
    banner("STAGE 5 - journey representation")
    vectorizer = F.JourneyVectorizer(cfg.features)
    matrix, info = vectorizer.fit_transform(journeys, sequences)
    print(json.dumps(info, indent=2))

    # -------------------------------------------------------------- cluster
    banner("STAGE 6 - clustering")
    labels, model, hstats = CL.fit_hdbscan(matrix, cfg.cluster)
    print("HDBSCAN:", json.dumps(hstats, indent=2))

    if len(matrix) > 50_000:
        rng = np.random.default_rng(cfg.cluster.random_state)
        sweep_idx = np.sort(rng.choice(len(matrix), 50_000, replace=False))
        sweep_matrix = matrix[sweep_idx]
    else:
        sweep_matrix = matrix
    reports["kmeans_sweep"] = CL.sweep_kmeans(sweep_matrix, cfg.cluster)
    print("\nKMeans baseline sweep:")
    print(reports["kmeans_sweep"].to_string(index=False))

    # if check_stability:
    #     reports["stability"] = CL.stability_check(matrix, cfg.cluster)
    #     print("\nstability (subsample ARI):")
    #     print(reports["stability"].to_string(index=False))

    catalog = CL.cluster_catalog(journeys, sequences, labels, matrix)
    reports["cluster_catalog"] = catalog.drop(columns=["os_mix"])
    print("\ncluster catalog:")
    print(
        catalog[["cluster", "size", "share", "median_length", "mean_back_rate", "medoid_path"]]
        .head(25)
        .to_string(index=False)
    )

    ngrams = F.top_ngrams_per_group(vectorizer, sequences, labels)
    reports["cluster_ngrams"] = ngrams

    # ------------------------------------------------------------- anomaly
    banner("STAGE 7 - Markov likelihood (anomaly companion)")
    journeys["cluster"] = labels
    bank = CL.MarkovBank(cfg.cluster.markov_smoothing).fit(sequences, labels)
    if cfg.cluster.fit_markov:
        journeys["markov_logprob"] = bank.score_all(sequences, labels)
        journeys["markov_logprob_global"] = bank.score_all(
            sequences, np.full(len(sequences), -999)
        )
        valid = journeys["markov_logprob"].notna()
        threshold = float(journeys.loc[valid, "markov_logprob"].quantile(0.05))
        journeys["anomaly_flag"] = (journeys["markov_logprob"] < threshold) | (labels == -1)
        print(
            f"p05 log-prob threshold = {threshold:.4f}\n"
            f"flagged anomalous = {int(journeys['anomaly_flag'].sum()):,} "
            f"({journeys['anomaly_flag'].mean():.1%})"
        )
        print("\nmost anomalous journeys:")
        print(
            journeys.nsmallest(8, "markov_logprob")[
                ["journey_id", "cluster", "n_events_final", "back_rate", "n_loop_removed", "markov_logprob"]
            ].to_string(index=False)
        )

    # -------------------------------------------------- fitted scorer (new data)
    banner("STAGE 8 - fit scorer and verify the new-data path")
    scorer = JourneyScorer.from_training_run(
        cfg, vectorizer, matrix, labels, journeys, sequences, bank
    )
    scorer.save(out_dir / f"{prefix}journey_scorer.pkl")

    # The holdout contains complete sessions excluded before fitting.
    if holdout_raw is not None and len(holdout_raw) > 500:
        scored = scorer.score(holdout_raw)
        print(
            f"scored {len(scored):,} journeys from the chronological holdout "
            f"({len(holdout_raw):,} events)\n"
            f"  matched a known archetype : {int((scored.cluster != -1).sum()):,} "
            f"({(scored.cluster != -1).mean():.1%})\n"
            f"  geometric anomalies       : {int(scored.geometric_anomaly.sum()):,}\n"
            f"  generative anomalies      : {int(scored.generative_anomaly.sum()):,}\n"
            f"  severe (both)             : {int(scored.severe_anomaly.sum()):,}"
        )
        print("\ntop friction flags:")
        print(
            scored.loc[scored.friction_flags.ne(""), "friction_flags"]
            .value_counts()
            .head(10)
            .to_string()
        )
        scored.to_csv(out_dir / f"{prefix}scored_holdout.csv", index=False)

    # ---------------------------------------------------------------- save
    banner("STAGE 9 - write artefacts")
    journeys["sequence"] = [" -> ".join(s) for s in sequences]
    journeys.to_csv(out_dir / f"{prefix}journeys.csv", index=False)
    journeys_all.to_csv(out_dir / f"{prefix}journeys_all_including_short.csv", index=False)
    catalog.to_json(out_dir / f"{prefix}cluster_catalog.json", orient="records", indent=2)
    ngrams.to_csv(out_dir / f"{prefix}cluster_ngrams.csv", index=False)
    seg[
        [
            "record_id", "session_id", "journey_id", "journey_pos", "boundary_reason",
            "event_time", "event_type", "segment_name", "event_token", "platform",
            "token_l1", "gap_prev_seconds", "gap_next_seconds",
        ]
    ].to_csv(out_dir / f"{prefix}events_canonical.csv", index=False)

    with (out_dir / f"{prefix}sequences.jsonl").open("w", encoding="utf-8") as fh:
        for jid, seq in zip(journeys["journey_id"], sequences):
            fh.write(json.dumps({"journey_id": jid, "tokens": seq}) + "\n")

    for name, frame in reports.items():
        frame.to_csv(out_dir / f"{prefix}report_{name}.csv", index=False)

    (out_dir / f"{prefix}run_config.json").write_text(
        json.dumps({"config": cfg.to_dict(), "feature_info": info, "hdbscan": hstats}, indent=2),
        encoding="utf-8",
    )

    lines = [f"# {platform.title()} Journey Pipeline Run\n"]
    for name, frame in reports.items():
        lines.append(f"\n## {name}\n\n{frame.head(40).to_markdown(index=False)}\n")
    (out_dir / f"{prefix}RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "journeys": journeys,
        "sequences": sequences,
        "labels": labels,
        "model": model,
        "scorer": scorer,
        "catalog": catalog,
        "reports": reports,
    }


def main() -> int:
    args = parse_args()
    config = PipelineConfig(input_csv=Path(args.input))
    config.tokens.level = args.level
    config.tokens.min_journey_df = args.token_min_df
    config.segment.idle_gap_seconds = args.idle_gap
    config.segment.min_journey_length = args.min_journey_length
    config.segment.use_entropy_boundaries = args.entropy
    config.features.ngram_range = (args.ngram_min, args.ngram_max)
    config.features.min_df = args.feature_min_df
    config.features.max_features = args.max_features
    config.features.svd_components = args.svd_dim
    config.features.numeric_block_weight = args.numeric_weight
    config.cluster.min_cluster_size = args.min_cluster_size
    config.cluster.min_samples = args.min_samples
    config.cluster.cluster_selection_method = args.cluster_selection_method
    config.post.drop_chrome = not args.keep_chrome
    config.post.drop_boot = args.drop_boot

    input_path = REPO_ROOT / config.input_csv
    output_root = REPO_ROOT / Path(args.output_root)
    out_dir = output_root / build_run_slug(config, test_size=args.test_size)
    prepared_dir = output_root / "_prepared" / build_prepared_data_slug(
        config, input_path=input_path, test_size=args.test_size
    )
    config.output_dir = (
        out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    android_train_path = prepared_dir / "canonical_events_android.csv"
    ios_train_path = prepared_dir / "canonical_events_ios.csv"

    android_test_path = prepared_dir / "canonical_events_android_test.csv"
    ios_test_path = prepared_dir / "canonical_events_ios_test.csv"
    cache_complete_path = prepared_dir / "_SUCCESS.json"
    prepared_paths = [
        android_train_path,
        ios_train_path,
        android_test_path,
        ios_test_path,
        cache_complete_path,
    ]

    if args.preprocess or not all(path.exists() for path in prepared_paths):
        banner("PREPROCESSING")
        combined, validation = read_production_folder(
            input_path,
            cfg=config.canonize,
            segment_cfg=config.segment,
        )
        validation.to_csv(prepared_dir / "preprocessing_report.csv", index=False)
        split_rows: list[dict[str, object]] = []
        destinations = {
            "android": (android_train_path, android_test_path),
            "ios": (ios_train_path, ios_test_path),
        }
        for platform, (train_path, test_path) in destinations.items():
            platform_df = combined.loc[combined.platform.eq(platform)].copy()
            train_df, test_df, split_report = split_sessions_chronologically(
                platform_df, test_size=args.test_size
            )
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            split_rows.append({"platform": platform, **split_report})
        pd.DataFrame(split_rows).to_csv(prepared_dir / "split_report.csv", index=False)
        cache_complete_path.write_text(
            json.dumps(
                {
                    "strategy": "chronological_complete_session",
                    "test_size": args.test_size,
                    "input": str(input_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        del combined
        gc.collect()
    else:
        print(f"reusing prepared train/test data: {prepared_dir}")

    preprocessing_report = prepared_dir / "preprocessing_report.csv"
    split_report = prepared_dir / "split_report.csv"
    if preprocessing_report.exists():
        pd.read_csv(preprocessing_report).to_csv(out_dir / "preprocessing_report.csv", index=False)
    if split_report.exists():
        pd.read_csv(split_report).to_csv(out_dir / "split_report.csv", index=False)

    manifest = {
        "run_slug": out_dir.name,
        "output_dir": str(out_dir),
        "prepared_data_dir": str(prepared_dir),
        "input": str(input_path),
        "test_size": args.test_size,
        "config": config.to_dict(),
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"experiment output: {out_dir}")

    platform_paths = (
        ("android", android_train_path, android_test_path),
        ("ios", ios_train_path, ios_test_path),
    )
    for platform, train_path, test_path in platform_paths:
        platform_df = pd.read_csv(train_path, low_memory=False, parse_dates=["event_time"])
        holdout_df = pd.read_csv(test_path, low_memory=False, parse_dates=["event_time"])
        overlap = set(platform_df["session_id"]) & set(holdout_df["session_id"])
        if overlap:
            raise RuntimeError(
                f"{platform}: prepared train/test data leaks {len(overlap)} sessions"
            )
        run_cluster_journey(
            platform_df,
            platform,
            config,
            out_dir,
            holdout_raw=holdout_df,
            check_stability=not args.no_stability,
        )
        del platform_df, holdout_df
        gc.collect()
    print(f"\nall artefacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
