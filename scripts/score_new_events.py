"""Inference for fitted journey runs: hierarchical C->B, or a single scorer.

    python scripts/score_new_events.py --platform android --input data/new_events.csv
    python scripts/score_new_events.py --platform ios --input data/events.csv --holdout-days 1
    python scripts/score_new_events.py --platform ios --input data/events.csv --predict-next

    # single-model fallback: one fitted JourneyScorer, no B rescue tier
    python scripts/score_new_events.py --platform android --input data/new_events.csv \
        --single-run output/journey_runs/<run>

Minimum raw production schema:

    session_id, platform, key, segmentation_name, client_time

Raw events are prepared once.  In hierarchical mode the strict C model is
authoritative; only its noise is transformed by B and considered for a secondary
assignment, and B must pass cluster-specific distance p95, Markov p05, and the
configured nearest vs second-nearest margin.  In single mode one fitted scorer
decides everything against its own global distance p95: a journey either matches
an archetype or is left unassigned.  Nothing is re-fitted during inference.

The semantic feature channels (`coarse`, `intent`, `operation`) are baked into
the fitted vectorizer, so they cannot be switched on or off here -- this script
only reports which ones the loaded run carries.  Changing them means refitting
with `run_journey_pipeline.py --channel-weights`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based.hierarchical_score import HierarchicalJourneyScorer  # noqa: E402
from Rule_based.production import canonicalize_frame  # noqa: E402
from Rule_based.score import JourneyScorer  # noqa: E402

TEST_OUTPUT_DIR = REPO_ROOT / "output" / "test"
TEST_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_POSTPROCESS_RUN = (
    REPO_ROOT
    / "output"
    / "journey_runs"
    / "L2_ng1-3_C-mcs100-ms5_B-mcs50-ms3_postprocessed"
)

UNKNOWN_NAME = {
    "cluster_name": "Hành trình chưa phân loại / hỗn hợp",
    "cluster_name_en": "Unclassified / mixed journeys",
    "business_family": "chưa phân loại",
    "business_family_code": "unknown",
    "naming_confidence": "not_applicable",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platform", required=True, choices=["android", "ios"])
    p.add_argument("--input", required=True, help="raw production CSV to score")
    p.add_argument(
        "--postprocess-run",
        type=Path,
        default=DEFAULT_POSTPROCESS_RUN,
        help="folder containing B/C postprocess config, thresholds, and cluster names",
    )
    p.add_argument(
        "--single-run",
        type=Path,
        help="fall back to one fitted scorer: a run folder holding "
        "<platform>_journey_scorer.pkl (no B rescue tier)",
    )
    p.add_argument(
        "--holdout-days",
        type=float,
        help="score the last N days of the training file instead (smoke test)",
    )
    p.add_argument("--output", help="where to write scored journeys")
    p.add_argument("--predict-next", action="store_true", help="show next-action predictions")
    p.add_argument("--top", type=int, default=10, help="rows to print")
    return p.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


# ------------------------------------------------------------------ loading
def load_single_scorer(run_dir: Path, platform: str) -> JourneyScorer:
    pickle_path = run_dir / f"{platform}_journey_scorer.pkl"
    if not pickle_path.exists():
        raise SystemExit(f"no fitted scorer at {pickle_path}")
    return JourneyScorer.load(pickle_path)


def load_single_names(run_dir: Path, platform: str) -> dict[int, dict[str, str]]:
    """Cluster names written by `apply_cluster_name_mapping.py`, if present.

    The mapping is keyed by `effective_cluster_key` ("C:<cluster>"); single-model
    inference only ever produces the C namespace, so the integer label is enough.
    """
    path = run_dir / f"{platform}_cluster_name_mapping.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["clusters"] if isinstance(payload, dict) else payload
    return {int(row["cluster"]): row for row in rows if row.get("namespace", "C") == "C"}


# ----------------------------------------------------------------- scoring
def score_single(
    scorer: JourneyScorer,
    raw: pd.DataFrame,
    names: dict[int, dict[str, str]],
) -> pd.DataFrame:
    """One-model inference, emitting the hierarchical output vocabulary.

    `effective_cluster_key`, `assignment_type` and the name columns are filled in
    the same shapes the hierarchical path produces, so both modes write CSVs that
    downstream consumers can read without branching.
    """
    scored = scorer.score(raw)
    if scored.empty:
        return scored

    cluster = scored["cluster"].to_numpy(dtype=int)
    assigned = cluster != -1
    scored["effective_cluster_key"] = np.where(
        assigned, [f"C:{int(value)}" for value in cluster], "UNKNOWN"
    )
    scored["assignment_type"] = np.where(assigned, "C_primary", "unassigned_novel")
    scored["effective_markov_logprob"] = scored["markov_logprob"]
    scored["effective_next_action"] = scored["next_action"]
    scored["effective_next_action_share"] = scored["next_action_share"]
    scored["behavioral_friction_flags"] = (
        scored["friction_flags"]
        .fillna("")
        .map(
            lambda value: "|".join(
                flag
                for flag in str(value).split("|")
                if flag and flag not in {"unknown_archetype", "improbable_transitions"}
            )
        )
    )
    rows = [
        names.get(int(value), UNKNOWN_NAME) if is_assigned else UNKNOWN_NAME
        for value, is_assigned in zip(cluster, assigned)
    ]
    for column, default in UNKNOWN_NAME.items():
        scored[column] = [row.get(column, default) for row in rows]
    return scored


# ---------------------------------------------------------------- reporting
def report_common(scored: pd.DataFrame, top: int) -> None:
    flags = (
        scored.loc[scored.behavioral_friction_flags.ne(""), "behavioral_friction_flags"]
        .str.split("|")
        .explode()
    )
    if not flags.empty:
        print("\nfriction signals:")
        print(flags.value_counts().to_string())

    print("\nmost anomalous journeys (lowest effective Markov log-prob):")
    cols = [
        "journey_id", "effective_cluster_key", "cluster_name", "business_family",
        "assignment_type", "n_events_final", "back_rate", "n_loop_removed",
        "effective_markov_logprob", "behavioral_friction_flags",
    ]
    print(scored.nsmallest(top, "effective_markov_logprob")[cols].to_string(index=False))


def report_hierarchical(scored: pd.DataFrame) -> None:
    n = len(scored)
    c_primary = scored.assignment_type.eq("C_primary")
    b_secondary = scored.assignment_type.eq("B_secondary_inference")
    unknown = scored.effective_cluster_key.eq("UNKNOWN")
    print(f"\nscored journeys: {n:,}")
    print(f"  C primary                 : {int(c_primary.sum()):,} ({c_primary.mean():.1%})")
    print(f"  B secondary rescue        : {int(b_secondary.sum()):,} ({b_secondary.mean():.1%})")
    print(f"  effective assigned        : {int((~unknown).sum()):,} ({(~unknown).mean():.1%})")
    print(f"  unresolved                : {int(unknown.sum()):,} ({unknown.mean():.1%})")
    print(f"  C geometric anomalies     : {int(scored.c_geometric_anomaly.sum()):,}")
    print(f"  C generative anomalies    : {int(scored.c_generative_anomaly.sum()):,}")
    print("\nassignment tiers:")
    print(scored.assignment_type.value_counts().to_string())


def report_single(scored: pd.DataFrame) -> None:
    n = len(scored)
    unknown = scored.effective_cluster_key.eq("UNKNOWN")
    print(f"\nscored journeys: {n:,}")
    print(f"  assigned                  : {int((~unknown).sum()):,} ({(~unknown).mean():.1%})")
    print(f"  unresolved                : {int(unknown.sum()):,} ({unknown.mean():.1%})")
    print(f"  geometric anomalies       : {int(scored.geometric_anomaly.sum()):,}")
    print(f"  generative anomalies      : {int(scored.generative_anomaly.sum()):,}")
    print(f"  severe (both)             : {int(scored.severe_anomaly.sum()):,}")
    print("\ntop archetypes:")
    print(scored.loc[~unknown, "cluster_name"].value_counts().head(10).to_string())


def report_predictions(scorer, scored: pd.DataFrame, top: int, single: bool) -> None:
    print("\nnext-action prediction (conditioned on the assigned archetype):")
    demo = scored.loc[scored.effective_cluster_key.ne("UNKNOWN")].head(top)
    for row in demo.itertuples():
        seq = row.sequence.split(" -> ")
        print(f"\n  {row.journey_id}  cluster={row.effective_cluster_key} name={row.cluster_name}")
        print(f"    at        : {seq[-1]}")
        if single:
            candidates = scorer.predict_next(seq, int(row.cluster), top_k=3)
        else:
            candidates = scorer.predict_next(seq, row.effective_cluster_key, top_k=3)
        for cand in candidates.itertuples():
            print(f"    -> {cand.observed_share:>6.1%}  {cand.next_token}")
        if candidates.empty:
            print("    -> no observed continuation (journey ends here)")


def main() -> int:
    args = parse_args()

    single = args.single_run is not None
    if single:
        run_dir = _resolve(args.single_run)
        scorer = load_single_scorer(run_dir, args.platform)
        names = load_single_names(run_dir, args.platform)
        base = scorer
        print(
            f"loaded single scorer: {len(scorer.centroids)} archetypes, "
            f"{len(names) or 'no'} cluster names"
        )
    else:
        postprocess_run = _resolve(args.postprocess_run)
        scorer = HierarchicalJourneyScorer.load(postprocess_run, args.platform)
        base = scorer.c_scorer
        print(
            f"loaded hierarchical scorer: C={len(scorer.c_scorer.centroids)} archetypes, "
            f"B={len(scorer.b_scorer.centroids)} fallback archetypes"
        )

    weights = getattr(base.vectorizer, "channel_weights", {})
    print(
        "feature channels: "
        + (", ".join(f"{name}={weight:g}" for name, weight in sorted(weights.items()))
           if weights else "none (single-channel representation)")
        + f"; numeric block weight={base.cfg.features.numeric_block_weight:g}"
    )

    input_path = _resolve(Path(args.input))
    source = pd.read_csv(input_path, low_memory=False)
    raw, report = canonicalize_frame(
        source, source_file=input_path.name, cfg=base.cfg.canonize
    )
    raw = raw.loc[raw.platform.eq(args.platform)].copy()
    print(f"canonicalized {len(raw):,} {args.platform} events; invalid time={report['invalid_client_time']}")

    if args.holdout_days:
        ts = pd.to_datetime(raw["event_time"], utc=True, format="mixed")
        cutoff = ts.max() - pd.Timedelta(days=args.holdout_days)
        raw = raw.loc[ts >= cutoff]
        print(f"holdout: last {args.holdout_days} day(s) -> {len(raw):,} events since {cutoff}")

    scored = score_single(scorer, raw, names) if single else scorer.score(raw)
    if scored.empty:
        print("no journey long enough to score")
        return 0

    report_single(scored) if single else report_hierarchical(scored)
    report_common(scored, args.top)
    if args.predict_next:
        report_predictions(scorer, scored, args.top, single)

    out = Path(args.output) if args.output else TEST_OUTPUT_DIR / f"{input_path.stem}_{args.platform}_scored.csv"
    out = _resolve(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
