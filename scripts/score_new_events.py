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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.json_to_csv import json_to_csv  # noqa: E402
from utils.clean_data import clean_data  # noqa: E402
from Rule_based.score import JourneyScorer  # noqa: E402
from Rule_based.cluster_mapping import apply_cluster_mapping  # noqa: E402

MODEL_DIR = REPO_ROOT / "output" / "clusters"
TEST_OUTPUT_DIR = REPO_ROOT / "output" / "test"
TEST_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TEST_RAW_DIR = REPO_ROOT / "data" / "test_data"


DEFAULT_RAW = {
    "android": "data/test/clean_july_events_android.csv",
    "ios": "data/test/clean_july_events_ios.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action = "store_true", help="convert JSON to CSV and exit")
    p.add_argument("--platform", required=True, choices=["android", "ios"])
    p.add_argument("--input", help="raw events CSV/JSON to score")
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

    if args.json:
        if not args.input:
            print("ERROR: --input is required for JSON conversion", file=sys.stderr)
            return 1
        input_file = Path(args.input)
        input_path = TEST_RAW_DIR / str(input_file)
        json_to_csv(input_path, TEST_RAW_DIR / f"{input_file.stem}.csv")
        input_df = pd.read_csv(TEST_RAW_DIR / f"{input_file.stem}.csv", low_memory=False)
        android_df, ios_df = clean_data(input_df)
        if args.platform == "android":
            out = TEST_RAW_DIR / f"{input_file.stem}_android.csv"
            android_df.to_csv(out, index=False)
        elif args.platform == "ios":
            out = TEST_RAW_DIR / f"{input_file.stem}_ios.csv"
            ios_df.to_csv(out, index=False)
        print(f"converted {args.input} -> {out}")

    scorer = JourneyScorer.load(MODEL_DIR / f"{args.platform}_journey_scorer.pkl")
    print(
        f"loaded scorer: {len(scorer.centroids)} archetypes, "
        f"{len(scorer.markov.vocab):,} tokens in vocabulary"
    )

    raw = android_df if args.platform == "android" else ios_df

    if args.holdout_days:
        ts = pd.to_datetime(raw["created_at"], utc=True, format="mixed")
        cutoff = ts.max() - pd.Timedelta(days=args.holdout_days)
        raw = raw.loc[ts >= cutoff]
        print(f"holdout: last {args.holdout_days} day(s) -> {len(raw):,} events since {cutoff}")

    scored = scorer.score(raw)
    if scored.empty:
        print("no journey long enough to score")
        return 0

    mapping_path = REPO_ROOT / "output" / f"{args.platform}_cluster_class_mapping.json"
    scored = apply_cluster_mapping(scored, mapping_path)

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
    cols = ["journey_id", "cluster", "class_group", "class_name", "n_events_final", "back_rate",
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

    out = TEST_OUTPUT_DIR / f"{args.output}_{args.platform}_scored.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
