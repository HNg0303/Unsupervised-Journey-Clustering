"""Stage 5 - Journey representation.

Two channels, concatenated:

  SEQUENCE CHANNEL   TF-IDF over 1..3-grams of the cleaned token sequence,
                     reduced with truncated SVD and L2-normalised. n-grams are
                     what make order matter: HOME->PAY->CONFIRM and
                     HOME->SUPPORT->CHAT share zero bigrams even when their
                     unigram profiles overlap. <bos>/<eos> sentinels are
                     injected so entry and exit points become first-class
                     features.

  NUMERIC CHANNEL    Length, action ratio, back rate, loop count, revisit
                     ratio, dwell and gap statistics — the behaviour that the
                     cleanup stage moved out of the sequence.

The numeric block is standardised and down-weighted (default 0.35) so a handful
of scalars cannot outvote the sequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

from .config import FeatureConfig

BOS = "<bos>"
EOS = "<eos>"

NUMERIC_COLUMNS: tuple[str, ...] = (
    "n_events_final",
    "n_unique_tokens",
    "action_ratio",
    "back_rate",
    "revisit_ratio",
    "n_loop_removed",
    "n_dedup_removed",
    "span_seconds",
    "total_dwell_s",
    "median_gap_s",
)

# Tokens are opaque strings containing '/', '#', ':' and '?'. Joining them with
# a space and splitting on whitespace is the only safe analyzer.
_ANALYZER_SPLIT = r"(?u)\S+"


def sequences_to_documents(sequences: list[list[str]]) -> list[str]:
    """Whitespace-joined documents with boundary sentinels."""
    return [" ".join([BOS, *seq, EOS]) for seq in sequences]


def numeric_matrix(journeys: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The behavioural block, log1p-damped on every count/time column.

    Shared by both routes so that the only thing separating them is how the
    sequence itself is encoded.
    """
    cols = [c for c in NUMERIC_COLUMNS if c in journeys.columns]
    raw = journeys[cols].to_numpy(dtype=float)
    skewed = [
        cols.index(c)
        for c in (
            "n_events_final",
            "n_unique_tokens",
            "n_loop_removed",
            "n_dedup_removed",
            "span_seconds",
            "total_dwell_s",
            "median_gap_s",
        )
        if c in cols
    ]
    raw[:, skewed] = np.log1p(np.clip(raw[:, skewed], 0, None))
    return raw, cols


class JourneyVectorizer:
    """Fit-once / transform-many representation, so new data can be scored.

    This is what makes the pipeline usable for the stated end goal: the fitted
    object can embed an unseen journey into the same space, assign it to a
    cluster, and score it for anomaly.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self.tfidf = TfidfVectorizer(
            analyzer="word",
            token_pattern=_ANALYZER_SPLIT,
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
            max_features=cfg.max_features,
            sublinear_tf=cfg.sublinear_tf,
            lowercase=False,
        )
        self.svd: TruncatedSVD | None = None
        self.scaler = StandardScaler()
        self.numeric_columns: list[str] = []

    # -- helpers ----------------------------------------------------------
    def _numeric_matrix(self, journeys: pd.DataFrame) -> np.ndarray:
        raw, cols = numeric_matrix(journeys)
        self.numeric_columns = cols
        return raw

    # -- api --------------------------------------------------------------
    def fit_transform(
        self, journeys: pd.DataFrame, sequences: list[list[str]]
    ) -> tuple[np.ndarray, dict[str, object]]:
        docs = sequences_to_documents(sequences)
        tfidf_matrix = self.tfidf.fit_transform(docs)

        n_components = int(
            min(self.cfg.svd_components, max(2, min(tfidf_matrix.shape) - 1))
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=0)
        seq_block = normalize(self.svd.fit_transform(tfidf_matrix))

        num_block = normalize(self.scaler.fit_transform(self._numeric_matrix(journeys)))
        matrix = np.hstack([seq_block, num_block * self.cfg.numeric_block_weight])

        info = {
            "n_journeys": int(matrix.shape[0]),
            "tfidf_vocabulary": int(len(self.tfidf.vocabulary_)),
            "svd_components": n_components,
            "svd_explained_variance": round(float(self.svd.explained_variance_ratio_.sum()), 4),
            "numeric_features": len(self.numeric_columns),
            "final_dimension": int(matrix.shape[1]),
        }
        return matrix, info

    def transform(self, journeys: pd.DataFrame, sequences: list[list[str]]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("call fit_transform before transform")
        seq_block = normalize(self.svd.transform(self.tfidf.transform(sequences_to_documents(sequences))))
        num_block = normalize(self.scaler.transform(self._numeric_matrix(journeys)))
        return np.hstack([seq_block, num_block * self.cfg.numeric_block_weight])


class PatternVectorizer:
    """Route B: the same journey encoded by which frequent patterns it contains.

    Deliberately exposes the identical `fit_transform` / `transform` API as
    `JourneyVectorizer`, so `score.py` and every downstream consumer can be
    handed either route without knowing which one it got.

    The binary pattern matrix is reduced with SVD and the numeric block is
    appended exactly as in route A. The *only* difference between the routes is
    the sequence encoding: contiguous n-gram counts (A) versus non-contiguous
    subsequence membership (B).
    """

    def __init__(self, cfg: FeatureConfig, patterns: pd.DataFrame) -> None:
        self.cfg = cfg
        # tokens column holds the pattern as a list; keep it, it is the model
        self.patterns = patterns.reset_index(drop=True)
        self.svd: TruncatedSVD | None = None
        self.scaler = StandardScaler()
        self.numeric_columns: list[str] = []

    def _pattern_matrix(self, sequences: list[list[str]]) -> np.ndarray:
        from .prefixspan import contains

        matrix = np.zeros((len(sequences), len(self.patterns)), dtype=float)
        for j, tokens in enumerate(self.patterns["tokens"]):
            for i, seq in enumerate(sequences):
                if contains(tokens, seq):
                    matrix[i, j] = 1.0
        return matrix

    def _numeric_matrix(self, journeys: pd.DataFrame) -> np.ndarray:
        raw, cols = numeric_matrix(journeys)
        self.numeric_columns = cols
        return raw

    def match_counts(self, sequences: list[list[str]]) -> np.ndarray:
        """How many patterns each journey contains.

        A journey with zero matches has no route-B evidence at all: its
        sequence block is the zero vector, so every such journey lands on the
        same point and HDBSCAN reports them as one large, meaningless cluster.
        Callers should label those as noise rather than as an archetype.
        """
        return self._pattern_matrix(sequences).sum(axis=1).astype(int)

    def fit_transform(
        self, journeys: pd.DataFrame, sequences: list[list[str]]
    ) -> tuple[np.ndarray, dict[str, object]]:
        binary = self._pattern_matrix(sequences)
        n_components = int(min(self.cfg.svd_components, max(2, min(binary.shape) - 1)))
        self.svd = TruncatedSVD(n_components=n_components, random_state=0)
        seq_block = normalize(self.svd.fit_transform(binary))

        num_block = normalize(self.scaler.fit_transform(self._numeric_matrix(journeys)))
        matrix = np.hstack([seq_block, num_block * self.cfg.numeric_block_weight])

        info = {
            "n_journeys": int(matrix.shape[0]),
            "n_patterns": int(len(self.patterns)),
            "svd_components": n_components,
            "svd_explained_variance": round(float(self.svd.explained_variance_ratio_.sum()), 4),
            "mean_patterns_per_journey": round(float(binary.sum(axis=1).mean()), 2),
            "journeys_matching_none": int((binary.sum(axis=1) == 0).sum()),
            "numeric_features": len(self.numeric_columns),
            "final_dimension": int(matrix.shape[1]),
        }
        return matrix, info

    def transform(self, journeys: pd.DataFrame, sequences: list[list[str]]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("call fit_transform before transform")
        seq_block = normalize(self.svd.transform(self._pattern_matrix(sequences)))
        num_block = normalize(self.scaler.transform(self._numeric_matrix(journeys)))
        return np.hstack([seq_block, num_block * self.cfg.numeric_block_weight])


def top_ngrams_per_group(
    vectorizer: JourneyVectorizer,
    sequences: list[list[str]],
    labels: np.ndarray,
    top_k: int = 8,
) -> pd.DataFrame:
    """class-based TF-IDF: which n-grams distinguish each cluster.

    Computes term frequency within each cluster, divides by the corpus-wide
    frequency, and reports the highest-lift n-grams. This is the interpretation
    layer — a cluster is only useful if you can name it.
    """
    docs = sequences_to_documents(sequences)
    counts = vectorizer.tfidf.transform(docs)
    vocab = np.array(vectorizer.tfidf.get_feature_names_out())

    corpus_mass = np.asarray(counts.sum(axis=0)).ravel() + 1e-9
    rows = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        cluster_mass = np.asarray(counts[mask].sum(axis=0)).ravel()
        if cluster_mass.sum() == 0:
            continue
        lift = (cluster_mass / cluster_mass.sum()) / (corpus_mass / corpus_mass.sum())
        # require the n-gram to actually be present in the cluster
        lift[cluster_mass <= 0] = 0.0
        order = np.argsort(-lift)[:top_k]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "cluster": int(label),
                    "rank": rank,
                    "ngram": vocab[idx],
                    "lift": round(float(lift[idx]), 3),
                    "cluster_mass": round(float(cluster_mass[idx]), 4),
                }
            )
    return pd.DataFrame(rows)
