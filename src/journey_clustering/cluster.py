"""Stage 6 - Clustering and the anomaly-scoring companion model.

Two models are fitted, and they answer different questions:

  HDBSCAN over the journey embedding
      "What journey archetypes exist?" Density-based, so it does not force a
      k, tolerates elongated clusters, and — critically for this project —
      labels genuinely unusual journeys as noise (-1) instead of jamming them
      into the nearest centroid. That noise label is the first, cheapest
      anomaly signal you get.

  Per-cluster first-order Markov chains
      "How surprising is this journey given its archetype?" A generative model
      gives a calibrated per-journey log-likelihood, which is what you actually
      need for early error / friction detection on new data. Cluster membership
      alone cannot rank severity; likelihood can.

KMeans across a k grid is also fitted, purely as a baseline to sanity-check that
HDBSCAN's structure is not an artefact of its hyperparameters.
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .config import ClusterConfig
from .timing import timed_stage


LOGGER = logging.getLogger(__name__)


def quality(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Internal validity indices, computed over non-noise points only."""
    return _quality(matrix, labels)


def _quality(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mask = labels != -1
    uniq = np.unique(labels[mask])
    if uniq.size < 2 or mask.sum() < 3:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    sample_size = min(10_000, int(mask.sum()))
    return {
        "silhouette": round(float(silhouette_score(
            matrix[mask], labels[mask], sample_size=sample_size,
            random_state=42 if sample_size < int(mask.sum()) else None,
        )), 4),
        "davies_bouldin": round(float(davies_bouldin_score(matrix[mask], labels[mask])), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(matrix[mask], labels[mask])), 2),
    }


def fit_hdbscan(matrix: np.ndarray, cfg: ClusterConfig) -> tuple[np.ndarray, HDBSCAN, dict[str, float]]:
    model = HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        metric="euclidean",
        cluster_selection_method=cfg.cluster_selection_method,
        # The scorer and catalog compute their own representatives. Asking
        # HDBSCAN to calculate medoids adds another expensive distance pass.
        store_centers=None,
        algorithm=cfg.algorithm,
        copy=False,
    )
    with timed_stage(LOGGER, "hdbscan.fit_predict", items=len(matrix)) as timing:
        labels = model.fit_predict(matrix)
    stats = _quality(matrix, labels)
    stats["n_clusters"] = int(len({l for l in labels if l != -1}))
    stats["noise_share"] = round(float((labels == -1).mean()), 4)
    stats["fit_elapsed_seconds"] = float(timing["elapsed_seconds"])
    return labels, model, stats


def sweep_kmeans(matrix: np.ndarray, cfg: ClusterConfig) -> pd.DataFrame:
    rows = []
    for k in cfg.kmeans_k_grid:
        if k >= matrix.shape[0]:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.random_state)
        with timed_stage(LOGGER, f"kmeans.k={k}", items=len(matrix)):
            labels = km.fit_predict(matrix)
        stats = _quality(matrix, labels)
        rows.append({"k": k, "inertia": round(float(km.inertia_), 3), **stats})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Markov companion model
# --------------------------------------------------------------------------
class MarkovBank:
    """One smoothed first-order Markov chain per cluster, plus a global chain.

    `score` returns mean per-transition log-probability (length-normalised, so
    long and short journeys are comparable). Low score = the journey does not
    behave like its cluster = candidate friction / error / anomaly.

    Transitions are held as sparse observed counts, not dense log-probability
    matrices. With ~60 clusters over a ~900-token vocabulary the dense form is
    ~360 MB of mostly-smoothing-constant; the sparse form is a few MB, and the
    smoothed probability is reconstructed per lookup:

        log P(b|a) = log(count[a,b] + s) - log(rowsum[a] + s*V)

    which is exactly what the dense version stored, without materialising the
    99.9% of entries that only ever hold the prior.
    """

    def __init__(self, smoothing: float = 0.5) -> None:
        self.smoothing = smoothing
        self.vocab: list[str] = []
        self._index: dict[str, int] = {}
        self.counts: dict[int, dict[tuple[int, int], float]] = {}
        self.row_totals: dict[int, np.ndarray] = {}

    def fit(self, sequences: list[list[str]], labels: np.ndarray) -> "MarkovBank":
        self.vocab = sorted({tok for seq in sequences for tok in seq})
        self._index = {tok: i for i, tok in enumerate(self.vocab)}
        n = len(self.vocab)

        buckets: dict[int, list[list[str]]] = defaultdict(list)
        for seq, label in zip(sequences, labels):
            buckets[int(label)].append(seq)
            buckets[-999].append(seq)  # -999 = global fallback chain

        for label, seqs in buckets.items():
            counts: dict[tuple[int, int], float] = defaultdict(float)
            totals = np.zeros(n, dtype=float)
            for seq in seqs:
                for a, b in zip(seq, seq[1:]):
                    i, j = self._index[a], self._index[b]
                    counts[(i, j)] += 1.0
                    totals[i] += 1.0
            self.counts[label] = dict(counts)
            # denominator includes the smoothing mass over the full vocabulary
            self.row_totals[label] = totals + self.smoothing * n
        return self

    def log_prob(self, label: int, i: int, j: int) -> float:
        counts = self.counts.get(int(label))
        totals = self.row_totals.get(int(label))
        if counts is None or totals is None:
            counts, totals = self.counts[-999], self.row_totals[-999]
        return math.log(counts.get((i, j), 0.0) + self.smoothing) - math.log(totals[i])

    def score(self, seq: list[str], label: int) -> float:
        pairs = [
            (self._index[a], self._index[b])
            for a, b in zip(seq, seq[1:])
            if a in self._index and b in self._index
        ]
        if not pairs:
            return float("nan")
        return float(np.mean([self.log_prob(label, i, j) for i, j in pairs]))

    def score_all(self, sequences: list[list[str]], labels: np.ndarray) -> np.ndarray:
        return np.array([self.score(s, l) for s, l in zip(sequences, labels)])

    def predict_next(
        self, prefix: list[str], label: int = -999, top_k: int = 5
    ) -> list[tuple[str, float, float]]:
        """Most likely next tokens given the journey so far and its archetype.

        Only the last token conditions the prediction (first-order chain). The
        cluster label is what makes this useful: P(next | screen) differs a lot
        between "paying a bill" and "hunting for support", and the per-cluster
        chain keeps those apart instead of averaging them into a global answer.

        Returns (token, smoothed_probability, observed_share) per candidate.
        Both numbers are reported because they answer different questions and
        differ a lot here: the smoothed probability is the one the anomaly score
        actually uses, but with V ~ 700 tokens the prior mass (s*V) dominates
        the denominator and squashes every value toward zero. `observed_share`
        is count / total transitions out of this screen - the number to quote to
        a human ("of users who reached this screen, 62% went to X next").
        """
        if not prefix or prefix[-1] not in self._index:
            return []
        i = self._index[prefix[-1]]
        counts = self.counts.get(int(label))
        totals = self.row_totals.get(int(label))
        if counts is None or totals is None:
            counts, totals = self.counts[-999], self.row_totals[-999]

        observed = {j: c for (a, j), c in counts.items() if a == i}
        if not observed:
            return []
        seen_total = sum(observed.values())
        ranked = sorted(observed.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            (
                self.vocab[j],
                round((c + self.smoothing) / totals[i], 4),
                round(c / seen_total, 4),
            )
            for j, c in ranked
        ]


def cluster_catalog(
    journeys: pd.DataFrame,
    sequences: list[list[str]],
    labels: np.ndarray,
    matrix: np.ndarray,
) -> pd.DataFrame:
    """One row per cluster: size, shape, medoid journey, representative path."""
    rows = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        idx = np.flatnonzero(mask)
        sub = journeys.iloc[idx]

        centroid = matrix[idx].mean(axis=0)
        medoid_local = int(np.argmin(np.linalg.norm(matrix[idx] - centroid, axis=1)))
        medoid_global = int(idx[medoid_local])
        medoid_seq = sequences[medoid_global]

        rows.append(
            {
                "cluster": int(label),
                "label_kind": "noise" if label == -1 else "cluster",
                "size": int(mask.sum()),
                "share": round(float(mask.mean()), 4),
                "n_sessions": int(sub["session_id"].nunique()),
                "n_devices": int(sub["device_id"].nunique()) if "device_id" in sub else None,
                "os_mix": sub["os"].value_counts(normalize=True).round(3).to_dict(),
                "median_length": float(sub["n_events_final"].median()),
                "median_span_s": float(sub["span_seconds"].median()),
                "mean_action_ratio": round(float(sub["action_ratio"].mean()), 4),
                "mean_back_rate": round(float(sub["back_rate"].mean()), 4),
                "mean_revisit_ratio": round(float(sub["revisit_ratio"].mean()), 4),
                "loop_journey_share": round(float((sub["n_loop_removed"] > 0).mean()), 4),
                "top_entry_token": sub["entry_token"].mode().iat[0] if not sub["entry_token"].isna().all() else None,
                "top_exit_token": sub["exit_token"].mode().iat[0] if not sub["exit_token"].isna().all() else None,
                "medoid_journey_id": journeys.iloc[medoid_global]["journey_id"],
                "medoid_length": len(medoid_seq),
                "medoid_path": " -> ".join(medoid_seq[:14]) + (" ..." if len(medoid_seq) > 14 else ""),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def stability_check(
    matrix: np.ndarray, cfg: ClusterConfig, n_runs: int = 5, subsample: float = 0.8
) -> pd.DataFrame:
    """Do the clusters survive resampling? Reports ARI against the full fit."""
    from sklearn.metrics import adjusted_rand_score

    base_labels, _, _ = fit_hdbscan(matrix, cfg)
    rng = np.random.default_rng(cfg.random_state)
    rows = []
    for run in range(n_runs):
        idx = rng.choice(matrix.shape[0], size=int(matrix.shape[0] * subsample), replace=False)
        sub_labels, _, _ = fit_hdbscan(matrix[idx], cfg)
        rows.append(
            {
                "run": run,
                "n_clusters": int(len({l for l in sub_labels if l != -1})),
                "ari_vs_full": round(float(adjusted_rand_score(base_labels[idx], sub_labels)), 4),
                "noise_share": round(float((sub_labels == -1).mean()), 4),
            }
        )
    return pd.DataFrame(rows)
