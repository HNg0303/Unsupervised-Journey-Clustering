"""Combine fitted B/C clustering runs into a hierarchical journey catalog.

Example:
    python scripts/postprocess_cluster_runs.py \
        --b-run output/journey_runs/<ng1-3_mcs50_ms3_run> \
        --c-run output/journey_runs/<ng1-3_mcs100_ms5_run> \
        --name L2_ng1-3_C-mcs100-ms5_B-mcs50-ms3_postprocessed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cluster_postprocess import (  # noqa: E402
    PostprocessThresholds,
    align_journeys,
    apply_hierarchy,
    build_secondary_catalog,
    cluster_quantiles,
    friction_flags,
    nearest_two_centroids,
)
from src.score import JourneyScorer  # noqa: E402
from src.tokens import channels_from_exact_sequences  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-run", required=True, type=Path, help="lenient fitted run")
    parser.add_argument("--c-run", required=True, type=Path, help="strict fitted run")
    parser.add_argument("--output-root", type=Path, default=Path("output/journey_runs"))
    parser.add_argument(
        "--name",
        default="L2_ng1-3_C-mcs100-ms5_B-mcs50-ms3_postprocessed",
        help="new folder name under --output-root",
    )
    parser.add_argument(
        "--platforms", nargs="+", default=["android", "ios"], choices=["android", "ios"]
    )
    parser.add_argument("--distance-quantile", type=float, default=0.95)
    parser.add_argument("--markov-quantile", type=float, default=0.05)
    parser.add_argument("--min-distance-margin", type=float, default=0.10)
    parser.add_argument("--recurring-support", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    try:
        PostprocessThresholds(
            distance_quantile=args.distance_quantile,
            markov_quantile=args.markov_quantile,
            min_distance_margin=args.min_distance_margin,
            recurring_support=args.recurring_support,
        ).validate()
    except ValueError as exc:
        parser.error(str(exc))
    return args


def read_config(run_dir: Path) -> dict[str, object]:
    path = run_dir / "experiment_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing experiment config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_runs(b_run: Path, c_run: Path) -> tuple[dict[str, object], dict[str, object]]:
    b_config, c_config = read_config(b_run), read_config(c_run)
    if b_config.get("prepared_data_dir") != c_config.get("prepared_data_dir"):
        raise ValueError("B and C do not use the same prepared_data_dir")
    b_features = b_config.get("config", {}).get("features")
    c_features = c_config.get("config", {}).get("features")
    if b_features != c_features:
        raise ValueError("B and C must use the same fitted feature configuration")
    return b_config, c_config


def load_platform(run_dir: Path, platform: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    journeys_path = run_dir / f"{platform}_journeys.csv"
    catalog_path = run_dir / f"{platform}_cluster_catalog.json"
    if not journeys_path.exists() or not catalog_path.exists():
        raise FileNotFoundError(f"missing {platform} artifacts in {run_dir}")
    return pd.read_csv(journeys_path), pd.read_json(catalog_path)


def sequences(frame: pd.DataFrame) -> list[list[str]]:
    return [str(value).split(" -> ") if pd.notna(value) else [] for value in frame["sequence"]]


def run_platform(
    platform: str,
    b_run: Path,
    c_run: Path,
    output_dir: Path,
    thresholds: PostprocessThresholds,
    *,
    chunk_size: int,
) -> dict[str, object]:
    print(f"\n[{platform}] loading fitted B/C artifacts")
    b_journeys, b_catalog = load_platform(b_run, platform)
    c_journeys, _ = load_platform(c_run, platform)
    aligned = align_journeys(c_journeys, b_journeys)

    scorer_path = b_run / f"{platform}_journey_scorer.pkl"
    if not scorer_path.exists():
        raise FileNotFoundError(f"missing fitted B scorer: {scorer_path}")
    scorer = JourneyScorer.load(scorer_path)

    c_scorer_path = c_run / f"{platform}_journey_scorer.pkl"
    if not c_scorer_path.exists():
        raise FileNotFoundError(f"missing fitted C scorer: {c_scorer_path}")
    c_scorer = JourneyScorer.load(c_scorer_path)

    # Recreate B's fitted embedding once.  Alignment below maps it to C order.
    # Only the exact sequence was persisted, so the semantic channels are
    # re-derived from it - the enrichment stage is a pure function of the path.
    b_sequences = sequences(b_journeys)
    b_channels = channels_from_exact_sequences(b_sequences, scorer.channel_names)
    b_matrix = scorer.vectorizer.transform(b_journeys, b_sequences, b_channels)
    b_position = pd.Series(np.arange(len(b_journeys)), index=b_journeys["journey_id"])
    aligned_matrix = b_matrix[b_position.loc[aligned["journey_id"]].to_numpy(dtype=int)]

    b_labels = b_journeys["cluster"].to_numpy(dtype=int)
    known = b_labels != -1
    own_centers = np.vstack([scorer.centroids[int(label)] for label in b_labels[known]])
    own_distance = np.linalg.norm(b_matrix[known] - own_centers, axis=1)
    distance_limits = cluster_quantiles(
        b_labels[known], own_distance, thresholds.distance_quantile
    )
    known_markov = b_journeys.loc[known, "markov_logprob"].to_numpy(dtype=float)
    markov_limits = cluster_quantiles(
        b_labels[known], known_markov, thresholds.markov_quantile
    )
    threshold_payload = {
        "platform": platform,
        "source_B_run": str(b_run),
        "distance_quantile": thresholds.distance_quantile,
        "markov_quantile": thresholds.markov_quantile,
        "distance_limits": {str(key): value for key, value in distance_limits.items()},
        "markov_limits": {str(key): value for key, value in markov_limits.items()},
    }
    (output_dir / f"{platform}_B_cluster_thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2), encoding="utf-8"
    )

    c_sequences = sequences(c_journeys)
    c_channels = channels_from_exact_sequences(c_sequences, c_scorer.channel_names)
    c_matrix = c_scorer.vectorizer.transform(c_journeys, c_sequences, c_channels)
    c_labels = c_journeys["cluster"].to_numpy(dtype=int)
    c_known = c_labels != -1
    c_own_centers = np.vstack(
        [c_scorer.centroids[int(label)] for label in c_labels[c_known]]
    )
    c_own_distance = np.linalg.norm(c_matrix[c_known] - c_own_centers, axis=1)
    c_distance_limits = cluster_quantiles(
        c_labels[c_known], c_own_distance, thresholds.distance_quantile
    )
    c_threshold_payload = {
        "platform": platform,
        "source_C_run": str(c_run),
        "distance_quantile": thresholds.distance_quantile,
        "distance_limits": {str(key): value for key, value in c_distance_limits.items()},
    }
    (output_dir / f"{platform}_C_cluster_thresholds.json").write_text(
        json.dumps(c_threshold_payload, indent=2), encoding="utf-8"
    )

    n = len(aligned)
    aligned["nearest_b_cluster"] = pd.Series(pd.NA, index=aligned.index, dtype="Int64")
    aligned["second_b_cluster"] = pd.Series(pd.NA, index=aligned.index, dtype="Int64")
    for column in (
        "nearest_b_distance", "second_b_distance", "nearest_b_distance_limit",
        "nearest_b_distance_ratio", "nearest_b_distance_margin",
        "nearest_b_markov_logprob", "nearest_b_markov_limit",
    ):
        aligned[column] = np.nan

    c_noise = aligned["c_cluster"].to_numpy(dtype=int) == -1
    b_existing = c_noise & (aligned["b_cluster"].to_numpy(dtype=int) != -1)
    b_both_noise = c_noise & (aligned["b_cluster"].to_numpy(dtype=int) == -1)

    # Existing B assignments are direct fitted evidence, not soft assignments.
    if b_existing.any():
        idx = np.flatnonzero(b_existing)
        labels = aligned.loc[b_existing, "b_cluster"].to_numpy(dtype=int)
        centers = np.vstack([scorer.centroids[int(label)] for label in labels])
        distances = np.linalg.norm(aligned_matrix[idx] - centers, axis=1)
        markov = scorer.markov.score_all(
            [b_sequences[b_position.at[jid]] for jid in aligned.loc[b_existing, "journey_id"]],
            labels,
        )
        aligned.loc[b_existing, "nearest_b_cluster"] = labels
        aligned.loc[b_existing, "nearest_b_distance"] = distances
        aligned.loc[b_existing, "nearest_b_distance_limit"] = [
            distance_limits[int(label)] for label in labels
        ]
        aligned.loc[b_existing, "nearest_b_markov_logprob"] = markov
        aligned.loc[b_existing, "nearest_b_markov_limit"] = [
            markov_limits[int(label)] for label in labels
        ]

    soft_pass = np.zeros(n, dtype=bool)
    if b_both_noise.any():
        idx = np.flatnonzero(b_both_noise)
        candidate_matrix = aligned_matrix[idx]
        first, first_distance, second, second_distance = nearest_two_centroids(
            candidate_matrix, scorer.centroids, chunk_size=chunk_size
        )
        candidate_sequences = [
            b_sequences[b_position.at[jid]] for jid in aligned.loc[b_both_noise, "journey_id"]
        ]
        markov = scorer.markov.score_all(candidate_sequences, first)
        distance_limit = np.array([distance_limits[int(label)] for label in first])
        markov_limit = np.array([markov_limits[int(label)] for label in first])
        margin = (second_distance - first_distance) / np.maximum(first_distance, 1e-12)
        passed = (
            (first_distance <= distance_limit)
            & (margin >= thresholds.min_distance_margin)
            & np.isfinite(markov)
            & (markov >= markov_limit)
        )
        soft_pass[idx] = passed
        aligned.loc[b_both_noise, "nearest_b_cluster"] = first
        aligned.loc[b_both_noise, "second_b_cluster"] = second
        aligned.loc[b_both_noise, "nearest_b_distance"] = first_distance
        aligned.loc[b_both_noise, "second_b_distance"] = second_distance
        aligned.loc[b_both_noise, "nearest_b_distance_limit"] = distance_limit
        aligned.loc[b_both_noise, "nearest_b_distance_margin"] = margin
        aligned.loc[b_both_noise, "nearest_b_markov_logprob"] = markov
        aligned.loc[b_both_noise, "nearest_b_markov_limit"] = markov_limit

    aligned["nearest_b_distance_ratio"] = (
        aligned["nearest_b_distance"] / aligned["nearest_b_distance_limit"]
    )
    aligned["friction_flags"] = friction_flags(aligned, scorer.thresholds)
    aligned["contains_missing_token"] = aligned["sequence"].str.contains(
        "<missing>", regex=False, na=False
    )
    noise_support = (
        aligned.loc[c_noise, "sequence"].value_counts(dropna=False).to_dict()
    )
    aligned["exact_sequence_noise_support"] = np.where(
        c_noise, aligned["sequence"].map(noise_support).fillna(0), 0
    ).astype(int)
    enriched = apply_hierarchy(
        aligned, soft_pass=soft_pass, recurring_support=thresholds.recurring_support
    )

    secondary_catalog = build_secondary_catalog(enriched, b_catalog)
    unresolved = enriched[enriched["secondary_cluster"].isna() & (enriched["c_cluster"] == -1)]
    assignment_counts = enriched["assignment_type"].value_counts().rename_axis(
        "assignment_type"
    ).reset_index(name="journeys")
    assignment_counts["share"] = assignment_counts["journeys"] / len(enriched)

    prefix = f"{platform}_"
    enriched.to_csv(output_dir / f"{prefix}journeys_postprocessed.csv", index=False)
    unresolved.to_csv(output_dir / f"{prefix}unresolved_noise.csv", index=False)
    assignment_counts.to_csv(output_dir / f"{prefix}assignment_summary.csv", index=False)
    secondary_catalog.to_csv(output_dir / f"{prefix}secondary_cluster_catalog.csv", index=False)
    secondary_catalog.to_json(
        output_dir / f"{prefix}secondary_cluster_catalog.json", orient="records", indent=2
    )

    c_noise_count = int(c_noise.sum())
    secondary_count = int(enriched["secondary_cluster"].notna().sum())
    summary: dict[str, object] = {
        "platform": platform,
        "journeys": int(len(enriched)),
        "C_primary_journeys": int((enriched["c_cluster"] != -1).sum()),
        "C_noise_journeys": c_noise_count,
        "B_existing_secondary_journeys": int(
            (enriched["assignment_type"] == "B_existing_secondary").sum()
        ),
        "B_borderline_secondary_journeys": int(
            (enriched["assignment_type"] == "B_borderline_secondary").sum()
        ),
        "secondary_rescue_share_of_C_noise": round(
            secondary_count / max(c_noise_count, 1), 6
        ),
        "unresolved_noise_journeys": int(len(unresolved)),
        "effective_assigned_share": round(
            float((enriched["effective_cluster_key"] != "UNKNOWN").mean()), 6
        ),
        "assignment_counts": {
            str(row.assignment_type): int(row.journeys)
            for row in assignment_counts.itertuples(index=False)
        },
    }
    (output_dir / f"{prefix}postprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    b_run, c_run = args.b_run.resolve(), args.c_run.resolve()
    b_config, c_config = validate_runs(b_run, c_run)
    thresholds = PostprocessThresholds(
        distance_quantile=args.distance_quantile,
        markov_quantile=args.markov_quantile,
        min_distance_margin=args.min_distance_margin,
        recurring_support=args.recurring_support,
    )
    output_dir = (args.output_root / args.name).resolve()
    if output_dir in (b_run, c_run):
        raise ValueError("postprocess output must differ from both fitted source runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_platform(
            platform, b_run, c_run, output_dir, thresholds, chunk_size=args.chunk_size
        )
        for platform in args.platforms
    ]
    config = {
        "postprocess_kind": "C_primary_B_secondary",
        "B_run": str(b_run),
        "C_run": str(c_run),
        "B_run_slug": b_config.get("run_slug", b_run.name),
        "C_run_slug": c_config.get("run_slug", c_run.name),
        "thresholds": {
            "distance_quantile": thresholds.distance_quantile,
            "markov_quantile": thresholds.markov_quantile,
            "min_distance_margin": thresholds.min_distance_margin,
            "recurring_support": thresholds.recurring_support,
        },
        "platforms": args.platforms,
        "summaries": summaries,
    }
    (output_dir / "postprocess_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    report = [
        "# B/C hierarchical cluster post-processing",
        "",
        f"- Primary strict run (C): `{c_run.name}`",
        f"- Secondary lenient run (B): `{b_run.name}`",
        "- Original fitted labels are preserved; secondary labels are namespaced as `B:<id>`.",
        "",
        "| platform | journeys | C primary | C noise | B fitted rescue | B soft rescue | unresolved | effective assigned |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        report.append(
            "| {platform} | {journeys:,} | {C_primary_journeys:,} | {C_noise_journeys:,} | "
            "{B_existing_secondary_journeys:,} | {B_borderline_secondary_journeys:,} | "
            "{unresolved_noise_journeys:,} | {effective_assigned_share:.2%} |".format(**summary)
        )
    (output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote postprocessed run to {output_dir}")


if __name__ == "__main__":
    main()
