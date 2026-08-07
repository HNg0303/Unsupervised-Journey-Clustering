"""End-to-end journey pipeline: canonize -> tokenize -> segment -> cluster.

    python scripts/run_journey_pipeline.py
    python scripts/run_journey_pipeline.py --level L3 --idle-gap 120 --entropy

Writes every intermediate artefact plus a markdown validation report to
outputs/journey/.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
from Rule_based.score import JourneyScorer  # noqa: E402
from Rule_based.production import read_production_folder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preprocess", action="store_true", help="canonicalize production CSV files first")
    p.add_argument("--input", default="data/train_data/raw_data_production/data_raw_sample")
    p.add_argument("--output", default="output/clusters_with_screen")
    p.add_argument("--level", default="L2", choices=["L1", "L2", "L3"])
    p.add_argument("--idle-gap", type=float, default=90.0)
    p.add_argument("--min-cluster-size", type=int, default=100)
    p.add_argument("--entropy", action="store_true", help="add branching-entropy boundaries")
    p.add_argument("--keep-chrome", action="store_true", help="do not drop OS chrome screens")
    p.add_argument("--no-stability", action="store_true")
    return p.parse_args()


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def run_cluster_journey(
    raw: pd.DataFrame,
    platform: str,
    cfg: PipelineConfig,
    out_dir: Path,
    *,
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

    # hold-out check: score the last calendar day as if it were unseen data
    event_time = pd.to_datetime(raw["event_time"], utc=True, format="mixed")
    last_day = event_time.max().normalize()
    unseen = raw[event_time >= last_day]
    if len(unseen) > 500:
        scored = scorer.score(unseen)
        print(
            f"scored {len(scored):,} journeys from the final day ({len(unseen):,} events)\n"
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
    config = PipelineConfig(input_csv=Path(args.input), output_dir=Path(args.output))
    config.tokens.level = args.level
    config.segment.idle_gap_seconds = args.idle_gap
    config.segment.use_entropy_boundaries = args.entropy
    config.cluster.min_cluster_size = args.min_cluster_size
    config.post.drop_chrome = not args.keep_chrome

    out_dir = REPO_ROOT / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    android_train_path = out_dir / "canonical_events_android.csv"
    ios_train_path = out_dir / "canonical_events_ios.csv"

    android_test_path = out_dir / "canonical_events_android_test.csv"
    ios_test_path = out_dir / "canonical_events_ios_test.csv"

    if args.preprocess:
        banner("PREPROCESSING")
        combined, validation = read_production_folder(
            REPO_ROOT / config.input_csv,
            cfg=config.canonize,
            segment_cfg=config.segment,
        )
        validation.to_csv(out_dir / "preprocessing_report.csv", index=False)
        android_df = combined.loc[combined.platform.eq("android")].copy()
        ios_df = combined.loc[combined.platform.eq("ios")].copy()
        android_df_train, android_df_test = train_test_split(android_df, test_size=0.2, random_state=42)
        ios_df_train, ios_df_test = train_test_split(ios_df, test_size=0.2, random_state=42)

        android_df_train.to_csv(android_train_path, index=False)
        ios_df_train.to_csv(ios_train_path, index=False)

        android_df_test.to_csv(android_test_path, index=False) 
        ios_df_test.to_csv(ios_test_path, index=False)

    if not android_train_path.exists() or not ios_train_path.exists():
        raise FileNotFoundError("canonical files are missing; run with --preprocess first")
    for platform, path in (("android", android_train_path), ("ios", ios_train_path)):
        platform_df = pd.read_csv(path, low_memory=False, parse_dates=["event_time"])
        run_cluster_journey(
            platform_df, platform, config, out_dir, check_stability=not args.no_stability
        )
        del platform_df
        gc.collect()
    print(f"\nall artefacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
