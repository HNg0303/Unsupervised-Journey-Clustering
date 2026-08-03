"""End-to-end journey pipeline: canonize -> tokenize -> segment -> cluster.

    python scripts/run_journey_pipeline.py
    python scripts/run_journey_pipeline.py --level L3 --idle-gap 120 --entropy

Writes every intermediate artefact plus a markdown validation report to
outputs/journey/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based import canonize as C  # noqa: E402
from Rule_based import cluster as CL  # noqa: E402
from Rule_based import features as F  # noqa: E402
from Rule_based import postprocess as P  # noqa: E402
from Rule_based import segment as S  # noqa: E402
from Rule_based import tokens as T  # noqa: E402
from Rule_based.config import PipelineConfig  # noqa: E402
from Rule_based.score import JourneyScorer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/final_clean_events.csv")
    p.add_argument("--output", default="outputs/journey")
    p.add_argument("--level", default="L2", choices=["L1", "L2", "L3"])
    p.add_argument("--idle-gap", type=float, default=90.0)
    p.add_argument("--min-cluster-size", type=int, default=15)
    p.add_argument("--entropy", action="store_true", help="add branching-entropy boundaries")
    p.add_argument("--keep-chrome", action="store_true", help="do not drop OS chrome screens")
    p.add_argument("--no-stability", action="store_true")
    return p.parse_args()


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    args = parse_args()
    cfg = PipelineConfig(input_csv=Path(args.input), output_dir=Path(args.output))
    cfg.tokens.level = args.level
    cfg.segment.idle_gap_seconds = args.idle_gap
    cfg.segment.use_entropy_boundaries = args.entropy
    cfg.cluster.min_cluster_size = args.min_cluster_size
    cfg.post.drop_chrome = not args.keep_chrome

    out_dir = REPO_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, pd.DataFrame] = {}

    # ---------------------------------------------------------------- load
    banner("STAGE 0 - load")
    raw = pd.read_csv(REPO_ROOT / cfg.input_csv, low_memory=False)
    print(f"rows={len(raw):,}  sessions={raw.session_id.nunique():,}  columns={len(raw.columns)}")

    # ------------------------------------------------------------ canonize
    banner("STAGE 1 - canonize (role-correct screen/target)")
    canon = C.canonize_events(raw, cfg.canonize)
    reports["canonization"] = C.canonization_report(raw, canon)
    print(reports["canonization"].to_string(index=False))

    # ------------------------------------------------------------ tokenize
    banner("STAGE 2 - tokenize (multi-resolution)")
    tok = T.build_tokens(canon, cfg.canonize)
    for level in ("L1", "L2", "L3"):
        col = {"L1": "token_l1", "L2": "token_l2", "L3": "token_l3"}[level]
        vc = tok[col].value_counts()
        print(
            f"{level}: vocab={len(vc):>5}  singletons={(vc == 1).sum():>4} "
            f"({(vc == 1).mean():.1%})  top100_coverage={vc.head(100).sum() / len(tok):.1%}"
        )
    dictionary = T.token_dictionary(tok, cfg.tokens.level)
    dictionary.to_csv(out_dir / "token_dictionary.csv", index=False)

    # ------------------------------------------------------------- segment
    banner("STAGE 3 - segment sessions into journeys")
    reports["idle_gap_sweep"] = S.sweep_idle_gap(tok, cfg.segment)
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

    # -------------------------------------------------------------- features
    banner("STAGE 5 - journey representation")
    vectorizer = F.JourneyVectorizer(cfg.features)
    matrix, info = vectorizer.fit_transform(journeys, sequences)
    print(json.dumps(info, indent=2))

    # -------------------------------------------------------------- cluster
    banner("STAGE 6 - clustering")
    labels, model, hstats = CL.fit_hdbscan(matrix, cfg.cluster)
    print("HDBSCAN:", json.dumps(hstats, indent=2))

    reports["kmeans_sweep"] = CL.sweep_kmeans(matrix, cfg.cluster)
    print("\nKMeans baseline sweep:")
    print(reports["kmeans_sweep"].to_string(index=False))

    if not args.no_stability:
        reports["stability"] = CL.stability_check(matrix, cfg.cluster)
        print("\nstability (subsample ARI):")
        print(reports["stability"].to_string(index=False))

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
    scorer.save(out_dir / "journey_scorer.pkl")

    # hold-out check: score the last calendar day as if it were unseen data
    cutoff = raw["created_at"].max()
    last_day = pd.to_datetime(cutoff, utc=True, format="mixed").normalize()
    unseen = raw[pd.to_datetime(raw["created_at"], utc=True, format="mixed") >= last_day]
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
        scored.to_csv(out_dir / "scored_holdout.csv", index=False)

    # ---------------------------------------------------------------- save
    banner("STAGE 9 - write artefacts")
    journeys["sequence"] = [" -> ".join(s) for s in sequences]
    journeys.to_csv(out_dir / "journeys.csv", index=False)
    journeys_all.to_csv(out_dir / "journeys_all_including_short.csv", index=False)
    catalog.to_json(out_dir / "cluster_catalog.json", orient="records", indent=2)
    ngrams.to_csv(out_dir / "cluster_ngrams.csv", index=False)
    seg[
        [
            "record_id", "session_id", "journey_id", "journey_pos", "boundary_reason",
            "ts", "event_type", "OS", "screen", "target", "screen_class",
            "token_l1", "token_l2", "token_l3", "duration_clip", "gap_prev_s",
        ]
    ].to_csv(out_dir / "events_canonical.csv", index=False)

    with (out_dir / "sequences.jsonl").open("w", encoding="utf-8") as fh:
        for jid, seq in zip(journeys["journey_id"], sequences):
            fh.write(json.dumps({"journey_id": jid, "tokens": seq}) + "\n")

    for name, frame in reports.items():
        frame.to_csv(out_dir / f"report_{name}.csv", index=False)

    (out_dir / "run_config.json").write_text(
        json.dumps({"config": cfg.to_dict(), "feature_info": info, "hdbscan": hstats}, indent=2),
        encoding="utf-8",
    )

    lines = ["# Journey Pipeline Run\n"]
    for name, frame in reports.items():
        lines.append(f"\n## {name}\n\n{frame.head(40).to_markdown(index=False)}\n")
    (out_dir / "RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nartefacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
