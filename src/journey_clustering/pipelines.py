"""Aggregate preparation and model-fitting pipelines.

This is the orchestration boundary for the reusable package.  It composes the
smaller preprocessing, segmentation, feature, clustering, and scoring modules
into the two operations used by application code:

``prepare_event_partition``
    Convert one session-safe raw-event partition into journey-level data.

``fit_global_journey_model``
    Fit and persist one platform model from prepared journey partitions.

Storage mechanics live in :mod:`journey_clustering.storage`; individual model
stages remain in their focused modules.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .storage import parquet_dataset, safe_partition_id
from .timing import timed_stage

LOGGER = logging.getLogger(__name__)

__all__ = ["prepare_event_partition", "fit_global_journey_model"]


@dataclass
class PreparedPartition:
    """Prepared journey rows and aligned sequence channels for one partition."""

    partition_id: str
    platform: str
    journeys: pd.DataFrame
    sequences: list[list[str]]
    channels: dict[str, list[list[str]]]
    events: pd.DataFrame
    reports: dict[str, pd.DataFrame]


def _qualified_ids(frame: pd.DataFrame, partition_id: str) -> dict[str, str]:
    prefix = safe_partition_id(partition_id)
    return {
        str(value): f"{prefix}::{value}"
        for value in frame["journey_id"].drop_duplicates().tolist()
    }


def prepare_event_partition(
    raw: pd.DataFrame,
    *,
    partition_id: str,
    platform: str,
    cfg: PipelineConfig,
    write_events: bool = False,
) -> PreparedPartition:
    """Run the existing stages 1-5 on one complete-session parquet partition.

    Rare-token folding is intentionally deferred until the global training
    population is assembled.  Folding independently per partition would make
    the vocabulary depend on the storage layout.  Inference does not need
    folding because its fitted TF-IDF vectorizer naturally ignores unseen
    terms.
    """
    # Keep storage-only callers (for example raw CSV -> Parquet partitioning)
    # independent from the optional model stack.  The heavy pipeline modules
    # are loaded only when a journey partition is actually prepared.
    from . import canonize as C
    from . import postprocess as P
    from . import segment as S
    from . import semantics as SEM
    from . import tokens as T

    if {"_source_file", "_source_row_number"}.issubset(raw.columns):
        # `canonicalize_frame` creates deterministic tie-breaker positions from
        # input order. Re-establish source-file/row order first so equal event
        # timestamps retain the all-at-once source ordering.
        raw = raw.sort_values(
            ["_source_file", "_source_row_number"], kind="mergesort"
        ).reset_index(drop=True)
    canon = C.canonize_events(raw, cfg.canonize)
    enriched = SEM.annotate_semantics(canon)
    tok = T.build_tokens(enriched, cfg.canonize)

    sweep_events = tok if len(tok) <= 200_000 else tok.iloc[:200_000].copy()
    reports: dict[str, pd.DataFrame] = {
        "idle_gap_sweep": S.sweep_idle_gap(sweep_events, cfg.segment)
    }
    seg = S.assign_journeys(tok, cfg.segment)
    seg = S.refine_with_entropy(seg, cfg.segment)
    reports["segmentation"] = S.segmentation_report(seg, cfg.segment)

    token_col = T.level_column(cfg.tokens.level)
    channel_names = tuple(sorted(cfg.features.channel_weights))
    backoff_col = T.level_column(cfg.tokens.backoff_level)
    extra_cols = tuple(
        column
        for column in dict.fromkeys(
            (*(T.channel_column(name) for name in channel_names), backoff_col)
        )
        if column != token_col
    )
    journeys, sequences, extras = P.build_journey_sequences(
        seg, cfg.post, token_col=token_col, extra_token_cols=extra_cols
    )
    reports["postprocess"] = P.postprocess_report(journeys)

    channels = {
        name: extras.get(T.channel_column(name), sequences)
        for name in channel_names
    }
    backoff_sequences = extras.get(backoff_col, sequences)
    id_map = _qualified_ids(journeys, partition_id)
    seg["journey_id"] = seg["journey_id"].map(id_map)
    journeys["journey_id"] = journeys["journey_id"].map(id_map)

    # A partition is complete for a session, so this local ID is now globally
    # unique after adding the partition prefix.  Keep all journeys on disk;
    # the current pipeline's minimum-length rule is applied at model fitting.
    journeys["platform"] = str(platform).lower()
    journeys["source_partition"] = str(partition_id)
    journeys["model_eligible"] = [
        len(sequence) >= cfg.segment.min_journey_length for sequence in sequences
    ]
    journeys["sequence_tokens"] = sequences
    journeys["backoff_sequence"] = backoff_sequences
    journeys["sequence"] = [" -> ".join(sequence) for sequence in sequences]
    for name, rows in channels.items():
        journeys[f"channel_{name}"] = rows

    events = seg if write_events else pd.DataFrame()
    return PreparedPartition(
        partition_id=str(partition_id),
        platform=str(platform).lower(),
        journeys=journeys,
        sequences=sequences,
        channels=channels,
        events=events,
        reports=reports,
    )


def sequence_value(value: object) -> list[str]:
    """Normalise a parquet list cell or a legacy display string."""
    if value is None or value is pd.NA:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text.split(" -> ") if text else []


def extract_journey_payload(
    journeys: pd.DataFrame, cfg: PipelineConfig
) -> tuple[pd.DataFrame, list[list[str]], dict[str, list[list[str]]]]:
    """Read journey rows plus aligned primary/semantic sequence channels."""
    from . import tokens as T

    frame = journeys.reset_index(drop=True).copy()
    sequence_column = "sequence_tokens" if "sequence_tokens" in frame else "sequence"
    sequences = [sequence_value(value) for value in frame[sequence_column]]
    channels = {
        name: [
            sequence_value(value)
            for value in frame.get(
                f"channel_{name}", pd.Series(sequences, index=frame.index)
            )
        ]
        for name in sorted(cfg.features.channel_weights)
    }
    return frame, sequences, channels


def read_journey_partitions(
    paths: Iterable[Path],
    *,
    platform: str,
    max_journeys: int | None = None,
    seed: int = 42,
    sampling_strategy: str = "customer_stratified",
) -> pd.DataFrame:
    """Load eligible journeys with an optional deterministic size cap.

    ``customer_stratified`` guarantees at least one journey for every distinct
    customer (missing customer IDs form one anonymous stratum), then allocates
    the remaining capacity proportionally to each customer's journey count.
    This preserves customer coverage and the journeys-per-customer distribution
    much better than a global reservoir.  It requires two bounded-memory passes
    over the parquet files.

    ``journey_reservoir`` retains the legacy uniform journey-level sampler.
    """
    strategies = {"customer_stratified", "journey_reservoir"}
    if sampling_strategy not in strategies:
        raise ValueError(
            f"sampling_strategy must be one of {sorted(strategies)}, got "
            f"{sampling_strategy!r}"
        )
    if max_journeys is not None and max_journeys < 1:
        raise ValueError("max_journeys must be positive")

    paths = list(paths)
    if max_journeys is not None and sampling_strategy == "customer_stratified":
        return _read_customer_stratified_journeys(
            paths, platform=platform, max_journeys=max_journeys, seed=seed
        )

    rng = np.random.default_rng(seed)
    candidates: list[pd.DataFrame] = []
    reservoir: list[dict[str, object]] = []
    seen = 0
    total_paths = len(paths)
    for path_number, path in enumerate(paths, start=1):
        frame = pd.read_parquet(path)
        if "platform" in frame:
            frame = frame.loc[frame["platform"].astype(str).str.lower().eq(platform.lower())]
        if "model_eligible" in frame:
            eligible = frame.loc[frame["model_eligible"].astype(bool)].copy()
        else:
            eligible = frame.loc[
                frame["n_events_final"].astype(int) >= 4
            ].copy()
        if eligible.empty:
            continue
        LOGGER.info(
            "%s: read journey partition %d/%d: %s eligible=%d%s",
            platform,
            path_number,
            total_paths,
            path.name,
            len(eligible),
            f" reservoir_seen={seen:,}" if max_journeys is not None else "",
        )
        if max_journeys is None:
            candidates.append(eligible)
            continue
        for row in eligible.to_dict(orient="records"):
            seen += 1
            if len(reservoir) < max_journeys:
                reservoir.append(row)
            else:
                position = int(rng.integers(0, seen))
                if position < max_journeys:
                    reservoir[position] = row
    if max_journeys is None:
        if not candidates:
            return pd.DataFrame()
        return pd.concat(candidates, ignore_index=True)
    return pd.DataFrame(reservoir)


def _eligible_platform_journeys(path: Path, platform: str) -> pd.DataFrame:
    """Read the eligible rows for one platform from a journey partition."""
    frame = pd.read_parquet(path)
    if "platform" in frame:
        frame = frame.loc[
            frame["platform"].astype(str).str.lower().eq(platform.lower())
        ]
    if "model_eligible" in frame:
        return frame.loc[frame["model_eligible"].astype(bool)].copy()
    return frame.loc[frame["n_events_final"].astype(int) >= 4].copy()


def _customer_strata(frame: pd.DataFrame) -> pd.Series:
    """Return stable customer strata, treating all missing IDs as anonymous."""
    if "customer_id" not in frame:
        raise ValueError(
            "customer_stratified sampling requires a customer_id column; "
            "use sampling_strategy='journey_reservoir' for legacy data"
        )
    customer = frame["customer_id"].astype("string").str.strip()
    missing = customer.isna() | customer.eq("")
    return customer.mask(missing, "__anonymous__").astype(str)


def _proportional_customer_quotas(
    counts: dict[str, int], max_journeys: int
) -> dict[str, int]:
    """Allocate an exact cap proportionally, with one row per customer."""
    if not counts:
        return {}
    if max_journeys < len(counts):
        raise ValueError(
            "max_journeys cannot preserve all customers: "
            f"cap={max_journeys:,}, customer_strata={len(counts):,}. "
            "Increase the cap or use sampling_strategy='journey_reservoir'."
        )
    total = sum(counts.values())
    if max_journeys >= total:
        return counts.copy()

    quotas = {customer: 1 for customer in counts}
    remaining = max_journeys - len(counts)
    extra_capacity = {customer: count - 1 for customer, count in counts.items()}
    capacity_total = sum(extra_capacity.values())
    exact = {
        customer: remaining * capacity / capacity_total
        for customer, capacity in extra_capacity.items()
    }
    for customer, value in exact.items():
        quotas[customer] += int(np.floor(value))
    unassigned = max_journeys - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda customer: (
            -(exact[customer] - np.floor(exact[customer])),
            customer,
        ),
    )
    for customer in ranked:
        if unassigned == 0:
            break
        if quotas[customer] < counts[customer]:
            quotas[customer] += 1
            unassigned -= 1
    return quotas


def _read_customer_stratified_journeys(
    paths: list[Path], *, platform: str, max_journeys: int, seed: int
) -> pd.DataFrame:
    """Two-pass, bounded-memory customer-stratified reservoir sampling."""
    counts: dict[str, int] = {}
    for path in paths:
        eligible = _eligible_platform_journeys(path, platform)
        if eligible.empty:
            continue
        for customer, count in _customer_strata(eligible).value_counts().items():
            counts[customer] = counts.get(customer, 0) + int(count)

    quotas = _proportional_customer_quotas(counts, max_journeys)
    if not quotas:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    seen = {customer: 0 for customer in quotas}
    samples: dict[str, list[dict[str, object]]] = {
        customer: [] for customer in quotas
    }
    for path_number, path in enumerate(paths, start=1):
        eligible = _eligible_platform_journeys(path, platform)
        if eligible.empty:
            continue
        LOGGER.info(
            "%s: customer-stratified pass 2, partition %d: %s eligible=%d",
            platform,
            path_number,
            path.name,
            len(eligible),
        )
        strata = _customer_strata(eligible)
        for customer, row in zip(strata, eligible.to_dict(orient="records")):
            seen[customer] += 1
            reservoir = samples[customer]
            quota = quotas[customer]
            if len(reservoir) < quota:
                reservoir.append(row)
            else:
                position = int(rng.integers(0, seen[customer]))
                if position < quota:
                    reservoir[position] = row

    rows = [row for customer in sorted(samples) for row in samples[customer]]
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def fit_global_journey_model(
    journeys: pd.DataFrame,
    *,
    platform: str,
    cfg: PipelineConfig,
    output_dir: Path,
    holdout: pd.DataFrame | None = None,
    training_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fit the unchanged current model logic on a journey-level dataset."""
    from . import cluster as CL
    from . import features as F
    from . import tokens as T
    from .score import JourneyScorer

    if journeys.empty:
        raise ValueError(f"{platform}: no eligible journeys available for training")
    started = time.perf_counter()
    LOGGER.info("%s: preparing %d journeys for model fitting", platform, len(journeys))
    timings: dict[str, dict[str, object]] = {}
    with timed_stage(LOGGER, f"{platform}.prepare_and_fold", items=len(journeys)) as timing:
        train, sequences, channels = extract_journey_payload(journeys, cfg)
        backoff = [sequence_value(value) for value in train["backoff_sequence"]]
        sequences, fold_stats = T.fold_rare_tokens(sequences, backoff, cfg.tokens)
    timings["prepare_and_fold"] = timing
    LOGGER.info(
        "%s: rare-token folding complete; vocabulary_before=%s vocabulary_after=%s",
        platform,
        fold_stats.iloc[0].get("vocabulary_before", "unknown")
        if not fold_stats.empty
        else "unknown",
        fold_stats.iloc[0].get("vocabulary_after", "unknown")
        if not fold_stats.empty
        else "unknown",
    )

    vectorizer = F.JourneyVectorizer(cfg.features)
    with timed_stage(LOGGER, f"{platform}.features.fit_transform", items=len(train)) as timing:
        matrix, feature_info = vectorizer.fit_transform(train, sequences, channels)
    timings["features.fit_transform"] = timing
    LOGGER.info(
        "%s: feature matrix fitted: rows=%d columns=%d",
        platform,
        matrix.shape[0],
        matrix.shape[1],
    )
    labels, model, hstats = CL.fit_hdbscan(matrix, cfg.cluster)
    LOGGER.info(
        "%s: HDBSCAN fitted: clusters=%s noise=%.1f%%",
        platform,
        hstats.get("n_clusters", 0),
        hstats.get("noise_share", 0) * 100,
    )
    sweep_matrix = matrix
    if len(matrix) > 50_000:
        rng = np.random.default_rng(cfg.cluster.random_state)
        indexes = np.sort(rng.choice(len(matrix), 50_000, replace=False))
        sweep_matrix = matrix[indexes]
    with timed_stage(LOGGER, f"{platform}.kmeans_sweep", items=len(sweep_matrix)) as timing:
        kmeans_report = CL.sweep_kmeans(sweep_matrix, cfg.cluster)
    timings["kmeans_sweep"] = timing
    reports = {"rare_folding": fold_stats, "kmeans_sweep": kmeans_report}

    with timed_stage(LOGGER, f"{platform}.catalog_and_ngrams", items=len(train)) as timing:
        catalog = CL.cluster_catalog(train, sequences, labels, matrix)
        ngrams = F.top_ngrams_per_group(vectorizer, sequences, labels)
    timings["catalog_and_ngrams"] = timing
    train["cluster"] = labels
    with timed_stage(LOGGER, f"{platform}.markov", items=len(sequences)) as timing:
        bank = CL.MarkovBank(cfg.cluster.markov_smoothing).fit(sequences, labels)
        train["markov_logprob"] = bank.score_all(sequences, labels)
        train["markov_logprob_global"] = bank.score_all(
            sequences, np.full(len(sequences), -999)
        )
    timings["markov"] = timing
    valid = train["markov_logprob"].notna()
    threshold = float(train.loc[valid, "markov_logprob"].quantile(0.05)) if valid.any() else -np.inf
    train["anomaly_flag"] = (train["markov_logprob"] < threshold) | (labels == -1)

    with timed_stage(LOGGER, f"{platform}.scorer_and_artifacts", items=len(train)) as timing:
        scorer = JourneyScorer.from_training_run(
            cfg, vectorizer, matrix, labels, train, sequences, bank
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        scorer.save(output_dir / f"{platform}_journey_scorer.pkl")
    timings["scorer_and_artifacts"] = timing
    LOGGER.info("%s: scorer saved to %s", platform, output_dir / f"{platform}_journey_scorer.pkl")

    train["sequence"] = [" -> ".join(sequence) for sequence in sequences]
    train.to_csv(output_dir / f"{platform}_journeys.csv", index=False)
    catalog.to_json(output_dir / f"{platform}_cluster_catalog.json", orient="records", indent=2)
    ngrams.to_csv(output_dir / f"{platform}_cluster_ngrams.csv", index=False)
    fold_stats.to_csv(output_dir / f"{platform}_report_rare_folding.csv", index=False)
    reports["kmeans_sweep"].to_csv(output_dir / f"{platform}_report_kmeans_sweep.csv", index=False)

    if holdout is not None and not holdout.empty:
        with timed_stage(LOGGER, f"{platform}.holdout_scoring", items=len(holdout)) as timing:
            holdout_frame, holdout_sequences, holdout_channels = extract_journey_payload(holdout, cfg)
            scored = scorer.score_prepared(
                holdout_frame, holdout_sequences, channels=holdout_channels
            )
        timings["holdout_scoring"] = timing
        scored.to_csv(output_dir / f"{platform}_scored_holdout.csv", index=False)
        LOGGER.info(
            "%s: holdout scored: journeys=%d known_cluster_rate=%.1f%%",
            platform,
            len(scored),
            (scored["cluster"].ne(-1).mean() * 100) if len(scored) else 0,
        )
    else:
        scored = pd.DataFrame()

    config_payload = {
        "platform": platform,
        "config": cfg.to_dict(),
        "feature_info": feature_info,
        "hdbscan": hstats,
        "timings": timings,
        "training_manifest": training_manifest or {},
    }
    (output_dir / f"{platform}_run_config.json").write_text(
        json.dumps(config_payload, indent=2, default=_json_default), encoding="utf-8"
    )
    (output_dir / f"{platform}_training_timings.json").write_text(
        json.dumps(timings, indent=2, default=_json_default), encoding="utf-8"
    )
    LOGGER.info(
        "%s: model artifacts written to %s; fit_elapsed=%.1fs",
        platform,
        output_dir,
        time.perf_counter() - started,
    )
    return {
        "scorer": scorer,
        "model": model,
        "journeys": train,
        "holdout_scored": scored,
        "catalog": catalog,
        "reports": reports,
        "feature_info": feature_info,
        "hdbscan": hstats,
        "timings": timings,
    }
