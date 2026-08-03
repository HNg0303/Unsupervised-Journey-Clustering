"""Inference: score a new batch of raw clickstream events.

    python scripts/score_new_events.py --platform android --input data/new_events.csv
    python scripts/score_new_events.py --platform ios --holdout-days 1
    python scripts/score_new_events.py --platform ios --holdout-days 1 --predict-next

The input is RAW events in exactly the schema the training data used:

    record_id, device_id, customer_id, session_id, created_at, timestamp,
    key, segmentation.name, segmentation.segment, segmentation.screen_id, duration

The fitted scorer re-runs the whole pipeline itself - canonize, tokenize,
segment, cleanup, vectorize - so training and inference can never drift apart.
Nothing is re-fitted: the vocabulary, SVD basis, scaler, centroids, Markov
chains and thresholds all come from the training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from utils.json_to_csv import json_to_csv
from utils.clean_data import drop_columns, rename_columns, map_values


import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based.score import JourneyScorer  # noqa: E402

MODEL_DIR = REPO_ROOT / "outputs" / "clusters"
DEFAULT_RAW = {
    "android": "data/clean_july_events_android.csv",
    "ios": "data/clean_july_events_ios.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platform", required=True, choices=["android", "ios"])
    p.add_argument("--input", help="raw events CSV to score")
    p.add_argument(
        "--holdout-days",
        type=float,
        help="score the last N days of the training file instead (smoke test)",
    )
    p.add_argument("--output", help="where to write scored journeys")
    p.add_argument("--predict-next", action="store_true", help="show next-action predictions")
    p.add_argument("--top", type=int, default=10, help="rows to print")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    scorer = JourneyScorer.load(MODEL_DIR / f"{args.platform}_scorer.pkl")
    print(
        f"loaded scorer: {len(scorer.centroids)} archetypes, "
        f"{len(scorer.markov.vocab):,} tokens in vocabulary"
    )

    path = Path(args.input) if args.input else REPO_ROOT / DEFAULT_RAW[args.platform]
    raw = pd.read_csv(path, low_memory=False)

    if args.holdout_days:
        ts = pd.to_datetime(raw["created_at"], utc=True, format="mixed")
        cutoff = ts.max() - pd.Timedelta(days=args.holdout_days)
        raw = raw.loc[ts >= cutoff]
        print(f"holdout: last {args.holdout_days} day(s) -> {len(raw):,} events since {cutoff}")
    print(f"input: {path.name}  events={len(raw):,}  sessions={raw.session_id.nunique():,}")

    scored = scorer.score(raw)
    if scored.empty:
        print("no journey long enough to score")
        return 0

    n = len(scored)
    print(f"\nscored journeys: {n:,}")
    print(f"  matched a known archetype : {int((scored.cluster != -1).sum()):,} "
          f"({(scored.cluster != -1).mean():.1%})")
    print(f"  geometric anomalies       : {int(scored.geometric_anomaly.sum()):,} "
          f"({scored.geometric_anomaly.mean():.1%})")
    print(f"  generative anomalies      : {int(scored.generative_anomaly.sum()):,} "
          f"({scored.generative_anomaly.mean():.1%})")
    print(f"  severe (both channels)    : {int(scored.severe_anomaly.sum()):,}")

    flags = scored.loc[scored.friction_flags.ne(""), "friction_flags"].str.split("|").explode()
    if not flags.empty:
        print("\nfriction signals:")
        print(flags.value_counts().to_string())

    print(f"\nmost anomalous journeys (lowest Markov log-prob):")
    cols = ["journey_id", "cluster", "n_events_final", "back_rate",
            "n_loop_removed", "markov_logprob", "friction_flags"]
    print(scored.nsmallest(args.top, "markov_logprob")[cols].to_string(index=False))

    if args.predict_next:
        print("\nnext-action prediction (conditioned on each journey's archetype):")
        demo = scored.loc[scored.cluster != -1].head(args.top)
        for row in demo.itertuples():
            seq = row.sequence.split(" -> ")
            print(f"\n  {row.journey_id}  cluster={row.cluster}")
            print(f"    at        : {seq[-1]}")
            top = scorer.predict_next(seq, row.cluster, top_k=3)
            for cand in top.itertuples():
                print(f"    -> {cand.observed_share:>6.1%}  {cand.next_token}")
            if top.empty:
                print("    -> no observed continuation (journey ends here)")

    out = Path(args.output) if args.output else MODEL_DIR / f"{args.platform}_scored.csv"
    scored.to_csv(out, index=False)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
