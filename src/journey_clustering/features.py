"""Stage 6 - Journey representation.

Several channels, each vectorised on its own and then concatenated:

  PRIMARY SEQUENCE   TF-IDF over 1..3-grams of the cleaned token sequence,
                     reduced with truncated SVD and L2-normalised. n-grams are
                     what make order matter: HOME->PAY->CONFIRM and
                     HOME->SUPPORT->CHAT share zero bigrams even when their
                     unigram profiles overlap. <bos>/<eos> sentinels are
                     injected so entry and exit points become first-class
                     features.

  SEMANTIC CHANNELS  The same journey re-read at the resolutions the semantic
                     enrichment stage produces - `coarse` (family/module),
                     `intent` (family/module/object/operation) and `operation`
                     (stage:operation). Each gets its own TF-IDF + SVD block.
                     They are what let two journeys that never share an exact
                     token still land near each other: an iOS modem restart and
                     an Android modem restart differ in every exact token and
                     agree in every coarse one.

  NUMERIC CHANNEL    Length, action ratio, back rate, loop count, revisit
                     ratio, dwell and gap statistics — the behaviour that the
                     cleanup stage moved out of the sequence.

Every block is L2-normalised *before* weighting, so a channel's influence is
its weight and nothing else - not its vocabulary size, not its SVD rank. The
numeric block is standardised and down-weighted (default 0.35) so a handful of
scalars cannot outvote the sequence.

Channels are optional. `FeatureConfig.channel_weights = {}` reproduces the
single-channel representation exactly, and a vectorizer unpickled from a run
that predates the channels keeps working: `transform` reads its extra encoders
through `getattr`, and an old object simply has none.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
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
    "median_gap_s",
    "p90_gap_s",
    "max_gap_s",
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
    # `to_numpy` hands back a read-only view when the selection is already one
    # homogeneous float block, and the log1p damping below writes in place.
    raw = np.array(journeys[cols].to_numpy(dtype=float), dtype=float, copy=True)
    skewed = [
        cols.index(c)
        for c in (
            "n_events_final",
            "n_unique_tokens",
            "n_loop_removed",
            "n_dedup_removed",
            "span_seconds",
            "median_gap_s",
            "p90_gap_s",
            "max_gap_s",
        )
        if c in cols
    ]
    raw[:, skewed] = np.log1p(np.clip(raw[:, skewed], 0, None))
    return raw, cols


class _SequenceEncoder:
    """One TF-IDF + SVD block. Fit once, transform many, L2-normalised out."""

    def __init__(self, cfg: FeatureConfig) -> None:
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
        self.svd_components = int(cfg.svd_components)

    def fit_transform(self, sequences: list[list[str]]) -> np.ndarray:
        counts = self.tfidf.fit_transform(sequences_to_documents(sequences))
        n_components = int(min(self.svd_components, max(2, min(counts.shape) - 1)))
        self.svd = TruncatedSVD(n_components=n_components, random_state=0)
        return normalize(self.svd.fit_transform(counts))

    def transform(self, sequences: list[list[str]]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("call fit_transform before transform")
        return normalize(self.svd.transform(self.tfidf.transform(sequences_to_documents(sequences))))

    @property
    def info(self) -> dict[str, object]:
        assert self.svd is not None
        return {
            "vocabulary": int(len(self.tfidf.vocabulary_)),
            "svd_components": int(self.svd.n_components),
            "svd_explained_variance": round(float(self.svd.explained_variance_ratio_.sum()), 4),
        }


class JourneyVectorizer:
    """Fit-once / transform-many representation, so new data can be scored.

    This is what makes the pipeline usable for the stated end goal: the fitted
    object can embed an unseen journey into the same space, assign it to a
    cluster, and score it for anomaly.

    `sequences` is always the primary channel. `channels` carries the extra
    semantic resolutions of the *same* journeys, index-aligned; the channel
    names must match `cfg.channel_weights`, and both `fit_transform` and
    `transform` must be given the same set.
    """

    def __init__(
        self, cfg: FeatureConfig, *, channel_weights: Mapping[str, float] | None = None
    ) -> None:
        self.cfg = cfg
        weights = cfg.channel_weights if channel_weights is None else channel_weights
        self.channel_weights: dict[str, float] = {
            name: float(weight) for name, weight in sorted(weights.items()) if weight > 0
        }
        # `tfidf` / `svd` stay on the object under their historical names: the
        # ONNX exporter and `top_ngrams_per_group` read them, and so do
        # scorers pickled before this module grew channels.
        self.primary = _SequenceEncoder(cfg)
        self.channels: dict[str, _SequenceEncoder] = {
            name: _SequenceEncoder(cfg) for name in self.channel_weights
        }
        self.scaler = StandardScaler()
        self.numeric_columns: list[str] = []
        self.global_pca: PCA | None = None

    # -- backward-compatible accessors ------------------------------------
    @property
    def tfidf(self) -> TfidfVectorizer:
        return self.primary.tfidf

    @property
    def svd(self) -> TruncatedSVD | None:
        return self.primary.svd

    def __setstate__(self, state: dict[str, object]) -> None:
        """Load pickles written before the primary encoder was factored out.

        Those objects stored `tfidf` and `svd` directly on the instance, which
        the properties above now shadow. Rebuilding a `_SequenceEncoder` around
        them restores a working single-channel vectorizer.
        """
        if "primary" not in state:
            encoder = _SequenceEncoder.__new__(_SequenceEncoder)
            encoder.tfidf = state.pop("tfidf")  # type: ignore[assignment]
            encoder.svd = state.pop("svd")  # type: ignore[assignment]
            encoder.svd_components = int(getattr(encoder.svd, "n_components", 0))
            state["primary"] = encoder
            state.setdefault("channels", {})
            state.setdefault("channel_weights", {})
        state.setdefault("global_pca", None)
        self.__dict__.update(state)

    # -- helpers ----------------------------------------------------------
    def _numeric_matrix(self, journeys: pd.DataFrame) -> np.ndarray:
        raw, cols = numeric_matrix(journeys)
        self.numeric_columns = cols
        return raw

    def _active_channels(self) -> dict[str, _SequenceEncoder]:
        """Channels this instance actually has; empty for legacy pickles."""
        return getattr(self, "channels", {})

    def _check_channels(self, channels: Mapping[str, list[list[str]]] | None) -> None:
        expected = set(self._active_channels())
        given = set(channels or {})
        if expected != given:
            raise ValueError(
                "channel mismatch: vectorizer was built for "
                f"{sorted(expected)} but was given {sorted(given)}"
            )

    def _global_projection(self, matrix: np.ndarray, *, fit: bool) -> np.ndarray:
        """Optionally reduce the already weighted global feature matrix."""
        components = getattr(self.cfg, "global_pca_components", None)
        if fit:
            self.global_pca = None
            if components is not None:
                components = int(components)
                if components < 1:
                    raise ValueError("global_pca_components must be positive or None")
                n_components = min(components, matrix.shape[0] - 1, matrix.shape[1])
                if n_components < 1:
                    raise ValueError("global PCA requires at least two rows")
                self.global_pca = PCA(
                    n_components=n_components,
                    whiten=bool(getattr(self.cfg, "global_pca_whiten", False)),
                    svd_solver="randomized",
                    random_state=0,
                )
                matrix = self.global_pca.fit_transform(matrix)
        elif self.global_pca is not None:
            matrix = self.global_pca.transform(matrix)
        return normalize(matrix)

    def _blocks(
        self,
        journeys: pd.DataFrame,
        sequences: list[list[str]],
        channels: Mapping[str, list[list[str]]] | None,
        *,
        fit: bool,
    ) -> tuple[list[np.ndarray], dict[str, object]]:
        self._check_channels(channels)
        primary = self.primary
        blocks = [primary.fit_transform(sequences) if fit else primary.transform(sequences)]
        info: dict[str, object] = {"primary": primary.info} if fit else {}

        for name, encoder in sorted(self._active_channels().items()):
            assert channels is not None  # guaranteed by _check_channels
            channel_sequences = channels[name]
            if len(channel_sequences) != len(sequences):
                raise ValueError(f"channel {name!r} has {len(channel_sequences)} rows, expected {len(sequences)}")
            block = encoder.fit_transform(channel_sequences) if fit else encoder.transform(channel_sequences)
            blocks.append(block * self.channel_weights[name])
            if fit:
                info[name] = {**encoder.info, "weight": self.channel_weights[name]}

        numeric = self._numeric_matrix(journeys)
        scaled = self.scaler.fit_transform(numeric) if fit else self.scaler.transform(numeric)
        blocks.append(normalize(scaled) * self.cfg.numeric_block_weight)
        return blocks, info

    # -- api --------------------------------------------------------------
    def fit_transform(
        self,
        journeys: pd.DataFrame,
        sequences: list[list[str]],
        channels: Mapping[str, list[list[str]]] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        blocks, channel_info = self._blocks(journeys, sequences, channels, fit=True)
        matrix = self._global_projection(np.hstack(blocks), fit=True)
        info = {
            "n_journeys": int(matrix.shape[0]),
            "tfidf_vocabulary": int(len(self.tfidf.vocabulary_)),
            "svd_components": int(channel_info["primary"]["svd_components"]),  # type: ignore[index]
            "svd_explained_variance": channel_info["primary"]["svd_explained_variance"],  # type: ignore[index]
            "channels": channel_info,
            "numeric_features": len(self.numeric_columns),
            "final_dimension": int(matrix.shape[1]),
        }
        if self.global_pca is not None:
            info["global_pca_components"] = int(self.global_pca.n_components_)
            info["global_pca_explained_variance"] = round(
                float(self.global_pca.explained_variance_ratio_.sum()), 4
            )
        else:
            info["global_pca_components"] = None
        return matrix, info

    def transform(
        self,
        journeys: pd.DataFrame,
        sequences: list[list[str]],
        channels: Mapping[str, list[list[str]]] | None = None,
    ) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("call fit_transform before transform")
        blocks, _ = self._blocks(journeys, sequences, channels, fit=False)
        return self._global_projection(np.hstack(blocks), fit=False)


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
