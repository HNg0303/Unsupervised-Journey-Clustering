"""Export the fitted archetype centroids from a saved scorer.

    python scripts/export_centroids.py
    python scripts/export_centroids.py --platforms ios

The centroids are what `score.py` actually assigns against: a new journey is
embedded into the same 74-d space and given the id of the nearest one. They live
inside the pickle, which makes them invisible to anything that is not Python.
This writes them out in both a human-readable and a machine-readable form.

    {platform}_centroids.csv   one row per cluster: id, name, family, size,
                               threshold, then dim_000..dim_NNN
    {platform}_centroids.npz   labels + matrix, for loading without unpickling
                               the whole scorer

NOTE: only route A (tf-idf) is exported, because only route A is fitted into the
shipped scorer. The centroid dimensions are SVD components followed by the
weighted numeric block - they are coordinates, not interpretable features. To
read what a cluster *means*, join on the catalog (medoid path, behavioural
stats) or the per-cluster n-gram lift table, not on the raw numbers here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based.cluster_names import lookup  # noqa: E402
from Rule_based.score import JourneyScorer  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "clusters"
ROUTE = "a_tfidf"  # the route the scorer ships


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platforms", nargs="+", default=["android", "ios"])
    return p.parse_args()


def export(platform: str) -> pd.DataFrame:
    scorer = JourneyScorer.load(OUT_DIR / f"{platform}_scorer.pkl")
    ids = sorted(scorer.centroids)
    matrix = np.vstack([scorer.centroids[i] for i in ids])

    # sizes come from the training catalog; the scorer does not retain them
    catalog_path = OUT_DIR / f"{platform}_{ROUTE}_catalog.csv"
    sizes: dict[int, int] = {}
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        sizes = dict(zip(catalog["cluster"].astype(int), catalog["size"].astype(int)))

    meta = pd.DataFrame(
        {
            "cluster": ids,
            "family": [lookup(platform, ROUTE, i)[0] for i in ids],
            "cluster_name": [lookup(platform, ROUTE, i)[1] for i in ids],
            "train_size": [sizes.get(i) for i in ids],
            "centroid_norm": np.round(np.linalg.norm(matrix, axis=1), 4),
        }
    )
    dims = pd.DataFrame(
        np.round(matrix, 6),
        columns=[f"dim_{j:03d}" for j in range(matrix.shape[1])],
    )
    table = pd.concat([meta, dims], axis=1)

    table.to_csv(OUT_DIR / f"{platform}_centroids.csv", index=False)
    np.savez(
        OUT_DIR / f"{platform}_centroids.npz",
        cluster_ids=np.array(ids),
        centroids=matrix,
        # the cut-off that turns "nearest centroid" into "-1, matches nothing"
        distance_p95=scorer.thresholds.distance_p95,
    )

    print(f"\n=== {platform} / {ROUTE} ===")
    print(f"archetypes={len(ids)}  dimensions={matrix.shape[1]}  "
          f"(svd={scorer.vectorizer.svd.n_components} + numeric={len(scorer.vectorizer.numeric_columns)})")
    print(f"assignment cut-off (distance_p95) = {scorer.thresholds.distance_p95:.4f}")
    print(meta.to_string(index=False))
    return table


def main() -> int:
    args = parse_args()
    for platform in args.platforms:
        export(platform)
        print(f"\n-> {OUT_DIR / f'{platform}_centroids.csv'}")
        print(f"-> {OUT_DIR / f'{platform}_centroids.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
