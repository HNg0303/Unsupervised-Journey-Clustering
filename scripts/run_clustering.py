"""Stage 5-7 for both platforms and BOTH representation routes.

    python scripts/run_clustering.py
    python scripts/run_clustering.py --ngram-sweep

Reads the postprocess artefacts (journey features + cleaned sequences), builds
two independent journey representations, clusters each, and writes the fitted
scorer used by `scripts/score_new_events.py`.

    route A  tf-idf over contiguous 1..N-grams -> SVD -> HDBSCAN
    route B  PrefixSpan pattern membership     -> SVD -> HDBSCAN

Both routes append the identical numeric/behavioural block, so any difference in
the resulting clusters is attributable to the sequence encoding alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based import cluster as CL  # noqa: E402
from Rule_based import features as F  # noqa: E402
from Rule_based.config import PipelineConfig  # noqa: E402
from Rule_based.prefixspan import mine_patterns  # noqa: E402
from Rule_based.score import JourneyScorer  # noqa: E402

IN_DIR = REPO_ROOT / "outputs" / "journeys"
OUT_DIR = REPO_ROOT / "outputs" / "clusters"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platforms", nargs="+", default=["android", "ios"])
    p.add_argument("--max-ngram", type=int, default=4)
    p.add_argument("--min-cluster-size", type=int, default=15)
    p.add_argument("--ngram-sweep", action="store_true", help="compare 1..n for n in 2,3,4")
    return p.parse_args()


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def load(platform: str) -> tuple[pd.DataFrame, list[list[str]]]:
    journeys = pd.read_csv(IN_DIR / f"{platform}_journey_features.csv")
    with (IN_DIR / f"{platform}_journey_sequences.json").open(encoding="utf-8") as fh:
        sequences = [r["tokens"] for r in json.load(fh)]
    if len(journeys) != len(sequences):
        raise ValueError(f"{platform}: {len(journeys)} journeys vs {len(sequences)} sequences")
    return journeys, sequences


def fit_route(name, vectorizer, journeys, sequences, cfg) -> dict[str, object]:
    matrix, info = vectorizer.fit_transform(journeys, sequences)
    labels, _, stats = CL.fit_hdbscan(matrix, cfg.cluster)
    bank = CL.MarkovBank(cfg.cluster.markov_smoothing).fit(sequences, labels)
    print(f"\n[{name}] representation: {json.dumps(info)}")
    print(f"[{name}] clustering    : {json.dumps(stats)}")
    return {
        "name": name,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "labels": labels,
        "bank": bank,
        "info": info,
        "stats": stats,
    }


def ngram_sweep(journeys, sequences, cfg) -> pd.DataFrame:
    """Does adding 4-grams buy anything? Evidence, not assumption."""
    rows = []
    for top_n in (2, 3, 4):
        trial = PipelineConfig()
        trial.features.ngram_range = (1, top_n)
        trial.cluster.min_cluster_size = cfg.cluster.min_cluster_size
        vec = F.JourneyVectorizer(trial.features)
        matrix, info = vec.fit_transform(journeys, sequences)
        labels, _, stats = CL.fit_hdbscan(matrix, trial.cluster)
        rows.append(
            {
                "ngram_range": f"1-{top_n}",
                "vocabulary": info["tfidf_vocabulary"],
                "svd_explained_variance": info["svd_explained_variance"],
                **{k: stats[k] for k in ("n_clusters", "noise_share", "silhouette", "davies_bouldin")},
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for platform in args.platforms:
        banner(f"{platform.upper()}")
        cfg = PipelineConfig()
        cfg.features.ngram_range = (1, args.max_ngram)
        cfg.cluster.min_cluster_size = args.min_cluster_size

        journeys, sequences = load(platform)
        print(f"journeys={len(journeys):,}  events={int(journeys.n_events_final.sum()):,}")

        if args.ngram_sweep:
            sweep = ngram_sweep(journeys, sequences, cfg)
            print("\nn-gram sweep (route A):")
            print(sweep.to_string(index=False))
            sweep.to_csv(OUT_DIR / f"{platform}_ngram_sweep.csv", index=False)

        # ---------------------------------------------------------- route A
        route_a = fit_route(
            "A tf-idf", F.JourneyVectorizer(cfg.features), journeys, sequences, cfg
        )

        # ---------------------------------------------------------- route B
        patterns = mine_patterns(sequences, cfg.prefixspan)
        vec_b = F.PatternVectorizer(cfg.features, patterns)
        route_b = fit_route("B prefixspan", vec_b, journeys, sequences, cfg)

        # journeys matching no pattern share the zero vector and would otherwise
        # be reported as route B's largest "archetype". They are unexplained,
        # not similar - relabel them as noise and re-score.
        unmatched = vec_b.match_counts(sequences) == 0
        route_b["labels"] = np.where(unmatched, -1, route_b["labels"])
        route_b["bank"] = CL.MarkovBank(cfg.cluster.markov_smoothing).fit(
            sequences, route_b["labels"]
        )
        route_b["stats"] = {
            **CL.quality(route_b["matrix"], route_b["labels"]),
            "n_clusters": int(len({l for l in route_b["labels"] if l != -1})),
            "noise_share": round(float((route_b["labels"] == -1).mean()), 4),
        }
        print(
            f"[B prefixspan] {int(unmatched.sum())} journeys matched no pattern "
            f"-> relabelled noise; {json.dumps(route_b['stats'])}"
        )

        # ------------------------------------------------------ do they agree
        ari = adjusted_rand_score(route_a["labels"], route_b["labels"])
        both = (route_a["labels"] != -1) & (route_b["labels"] != -1)
        ari_core = (
            adjusted_rand_score(route_a["labels"][both], route_b["labels"][both])
            if both.sum() > 2
            else float("nan")
        )
        print(f"\nagreement between routes: ARI={ari:.4f}  (non-noise only: {ari_core:.4f})")

        # ------------------------------------------------------------ outputs
        for route in (route_a, route_b):
            tag = "a_tfidf" if route["name"].startswith("A") else "b_prefixspan"
            catalog = CL.cluster_catalog(
                journeys, sequences, route["labels"], route["matrix"]
            )
            catalog.drop(columns=["os_mix"]).to_csv(
                OUT_DIR / f"{platform}_{tag}_catalog.csv", index=False
            )
            labelled = journeys[["journey_id", "session_id", "n_events_final"]].copy()
            labelled["cluster"] = route["labels"]
            labelled["markov_logprob"] = route["bank"].score_all(sequences, route["labels"])
            labelled["sequence"] = [" -> ".join(s) for s in sequences]
            labelled.to_csv(OUT_DIR / f"{platform}_{tag}_labels.csv", index=False)

            summary_rows.append(
                {"platform": platform, "route": route["name"], **route["stats"], "ari_vs_other": round(ari, 4)}
            )

            print(f"\n[{route['name']}] largest clusters:")
            print(
                catalog.loc[catalog.cluster != -1]
                .head(6)[["cluster", "size", "share", "median_length", "mean_back_rate", "medoid_path"]]
                .to_string(index=False, max_colwidth=90)
            )

        patterns.drop(columns=["tokens"]).to_csv(
            OUT_DIR / f"{platform}_patterns_used.csv", index=False
        )

        # the scorer ships route A (denser, and it degrades gracefully on
        # journeys that match no pattern at all)
        scorer = JourneyScorer.from_training_run(
            cfg,
            route_a["vectorizer"],
            route_a["matrix"],
            route_a["labels"],
            journeys,
            sequences,
            route_a["bank"],
        )
        scorer.save(OUT_DIR / f"{platform}_scorer.pkl")
        print(f"\nscorer -> {OUT_DIR / f'{platform}_scorer.pkl'}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "route_comparison.csv", index=False)
    banner("SUMMARY")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
