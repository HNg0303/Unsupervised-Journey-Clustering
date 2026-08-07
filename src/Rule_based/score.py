"""Inference path: assign new journeys to known clusters and score anomaly.

This is the deliverable the clustering exists to produce. A fitted `JourneyScorer`
takes raw events for a new time window and returns, per journey:

    cluster          nearest known archetype (or -1 = does not match any)
    distance         distance to that archetype's centroid
    markov_logprob   how likely the sequence is under that archetype's chain
    anomaly_score    combined, calibrated against the training distribution
    friction_flags   rule-based signals that explain *why* it looks bad

Two independent anomaly channels are kept on purpose:
  * geometric  (far from every centroid)  -> "unusual shape of journey"
  * generative (low Markov log-prob)      -> "unusual transitions inside it"
A journey can fail either way, and the two disagree often enough to be worth
reporting separately rather than blending into one opaque number.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import canonize as C
from . import postprocess as P
from . import segment as S
from . import tokens as T
from .cluster import MarkovBank
from .config import PipelineConfig
from .features import JourneyVectorizer


@dataclass
class ScoreThresholds:
    """Percentile cut-offs learned from the training journeys."""

    distance_p95: float
    markov_p05: float
    markov_p01: float
    back_rate_p90: float
    loops_p90: float
    revisit_p90: float
    span_p95: float


class JourneyScorer:
    """Fitted pipeline: raw events in, scored journeys out."""

    def __init__(
        self,
        cfg: PipelineConfig,
        vectorizer: JourneyVectorizer,
        centroids: dict[int, np.ndarray],
        markov: MarkovBank,
        thresholds: ScoreThresholds,
    ) -> None:
        self.cfg = cfg
        self.vectorizer = vectorizer
        self.centroids = centroids
        self.markov = markov
        self.thresholds = thresholds

    # ------------------------------------------------------------------ fit
    @classmethod
    def from_training_run(
        cls,
        cfg: PipelineConfig,
        vectorizer: JourneyVectorizer,
        matrix: np.ndarray,
        labels: np.ndarray,
        journeys: pd.DataFrame,
        sequences: list[list[str]],
        markov: MarkovBank,
    ) -> "JourneyScorer":
        centroids = {
            int(label): matrix[labels == label].mean(axis=0)
            for label in set(labels.tolist())
            if label != -1
        }
        assigned, distance = _assign(matrix, centroids)
        logprob = markov.score_all(sequences, labels)
        finite = logprob[np.isfinite(logprob)]

        thresholds = ScoreThresholds(
            distance_p95=float(np.percentile(distance, 95)),
            markov_p05=float(np.percentile(finite, 5)) if finite.size else -np.inf,
            markov_p01=float(np.percentile(finite, 1)) if finite.size else -np.inf,
            back_rate_p90=float(journeys["back_rate"].quantile(0.90)),
            loops_p90=float(journeys["n_loop_removed"].quantile(0.90)),
            revisit_p90=float(journeys["revisit_ratio"].quantile(0.90)),
            span_p95=float(journeys["span_seconds"].quantile(0.95)),
        )
        return cls(cfg, vectorizer, centroids, markov, thresholds)

    # -------------------------------------------------------------- predict
    def prepare(self, raw_events: pd.DataFrame) -> tuple[pd.DataFrame, list[list[str]]]:
        """Run the identical canonize -> tokenize -> segment -> clean pipeline."""
        canon = C.canonize_events(raw_events, self.cfg.canonize)
        tok = T.build_tokens(canon, self.cfg.canonize)
        seg = S.assign_journeys(tok, self.cfg.segment)
        seg["boundary_reason"] = seg["boundary_reason"].fillna("")
        col = {"L1": "token_l1", "L2": "token_l2", "L3": "token_l3"}[self.cfg.tokens.level]
        journeys, sequences, _ = P.build_journey_sequences(seg, self.cfg.post, token_col=col)
        keep = [len(s) >= self.cfg.segment.min_journey_length for s in sequences]
        return (
            journeys.loc[keep].reset_index(drop=True),
            [s for s, k in zip(sequences, keep) if k],
        )

    def score(self, raw_events: pd.DataFrame) -> pd.DataFrame:
        journeys, sequences = self.prepare(raw_events)
        if journeys.empty:
            return journeys

        matrix = self.vectorizer.transform(journeys, sequences)
        labels, distance = _assign(matrix, self.centroids)

        # a journey too far from every centroid is not a member of any archetype
        far = distance > self.thresholds.distance_p95
        labels = np.where(far, -1, labels)

        logprob = np.array(
            [self.markov.score(s, l) for s, l in zip(sequences, labels)], dtype=float
        )

        out = journeys.copy()
        out["cluster"] = labels
        out["distance_to_centroid"] = np.round(distance, 4)
        out["markov_logprob"] = np.round(logprob, 4)
        out["geometric_anomaly"] = far
        out["generative_anomaly"] = logprob < self.thresholds.markov_p05
        out["severe_anomaly"] = far & (logprob < self.thresholds.markov_p01)

        t = self.thresholds
        flags: list[str] = []
        for row in out.itertuples():
            marks = []
            if row.back_rate > t.back_rate_p90:
                marks.append("excessive_back")
            if row.n_loop_removed > t.loops_p90:
                marks.append("navigation_loop")
            if row.revisit_ratio > t.revisit_p90:
                marks.append("screen_thrash")
            if row.span_seconds > t.span_p95:
                marks.append("slow_journey")
            if row.geometric_anomaly:
                marks.append("unknown_archetype")
            if row.generative_anomaly:
                marks.append("improbable_transitions")
            flags.append("|".join(marks))
        out["sequence"] = [" -> ".join(s) for s in sequences]
        out["friction_flags"] = flags

        # what the archetype says this user does next - the same fitted chain
        # that produced the anomaly score, read forwards instead of backwards
        nxt = [self.markov.predict_next(s, l, top_k=1) for s, l in zip(sequences, labels)]
        out["next_action"] = [p[0][0] if p else None for p in nxt]
        out["next_action_share"] = [p[0][2] if p else np.nan for p in nxt]
        return out

    def predict_next(
        self, sequence: list[str], cluster: int = -999, top_k: int = 5
    ) -> pd.DataFrame:
        """Rank the likely next tokens for a live, unfinished journey.

        `cluster` defaults to the global chain; pass the cluster from `score`
        for a sharper, archetype-conditioned answer.
        """
        return pd.DataFrame(
            self.markov.predict_next(sequence, cluster, top_k),
            columns=["next_token", "smoothed_probability", "observed_share"],
        )

    # ------------------------------------------------------------- persist
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "JourneyScorer":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def _assign(matrix: np.ndarray, centroids: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not centroids:
        return np.full(matrix.shape[0], -1), np.zeros(matrix.shape[0])
    keys = np.array(sorted(centroids))
    stack = np.vstack([centroids[k] for k in keys])
    labels = np.empty(matrix.shape[0], dtype=keys.dtype)
    minimum = np.empty(matrix.shape[0], dtype=float)
    # A full N x K x D broadcast exceeded 49 GiB on production.  Chunking has
    # identical results and bounds peak memory independently of event volume.
    chunk_size = max(256, min(4096, 20_000_000 // max(len(keys) * matrix.shape[1], 1)))
    for start in range(0, matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, matrix.shape[0])
        distance = np.linalg.norm(matrix[start:stop, None, :] - stack[None, :, :], axis=2)
        best = np.argmin(distance, axis=1)
        labels[start:stop] = keys[best]
        minimum[start:stop] = distance[np.arange(stop - start), best]
    return labels, minimum


if __name__ == "__main__":
    import sys

    from .config import PipelineConfig
    from .features import JourneyVectorizer
    from .train import fit_pipeline

    cfg = PipelineConfig()
    scorer = fit_pipeline(cfg)
    scorer.save(Path(sys.argv[1]))
