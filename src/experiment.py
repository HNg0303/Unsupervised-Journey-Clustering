"""Experiment naming, cache keys, and leakage-safe train/test splitting."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pandas as pd

from .config import PipelineConfig


def _slug_number(value: int | float) -> str:
    """Render a number as a filesystem-safe, stable slug component."""
    return format(value, "g").replace("-", "m").replace(".", "p")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "data"


def build_run_slug(cfg: PipelineConfig, *, test_size: float) -> str:
    """Return the deterministic directory name for one experiment."""
    f = cfg.features
    c = cfg.cluster
    s = cfg.segment
    t = cfg.tokens
    p = cfg.post
    # Compact on purpose: the slug is a directory name and Windows still caps
    # a path at 260 characters. `coarse=0.45, intent=0.30` -> `c45i30`.
    channels = "".join(
        f"{name[0]}{round(weight * 100):02.0f}"
        for name, weight in sorted(f.channel_weights.items())
    )
    return "_".join(
        [
            t.level,
            f"ch-{channels or 'none'}",
            f"ng{f.ngram_range[0]}-{f.ngram_range[1]}",
            f"svd{f.svd_components}",
            f"fdf{f.min_df}",
            f"mf{f.max_features}",
            f"nw{_slug_number(f.numeric_block_weight)}",
            f"mcs{c.min_cluster_size}",
            f"ms{c.min_samples}",
            f"sel-{c.cluster_selection_method}",
            f"gap{_slug_number(s.idle_gap_seconds)}",
            f"jmin{s.min_journey_length}",
            f"tdf{t.min_journey_df}",
            f"ent{int(s.use_entropy_boundaries)}",
            f"chr{int(p.drop_chrome)}",
            f"boot{int(p.drop_boot)}",
            f"test{_slug_number(test_size)}",
        ]
    )


def build_prepared_data_slug(
    cfg: PipelineConfig, *, input_path: Path, test_size: float
) -> str:
    """Key the canonical-data cache only by settings that can change it."""
    resolved = str(input_path.resolve())
    source_hash = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8]
    s = cfg.segment
    return "_".join(
        [
            f"{_safe_name(input_path.name)}-{source_hash}",
            "time-session-split",
            f"test{_slug_number(test_size)}",
            f"gap{_slug_number(s.idle_gap_seconds)}",
            f"max{s.max_journey_length}",
            f"root{s.root_return_min_events}",
            f"auth{int(s.cut_on_auth_change)}",
        ]
    )


def split_sessions_chronologically(
    frame: pd.DataFrame, *, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Hold out the latest complete sessions, never individual event rows."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    required = {"session_id", "event_time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cannot split sessions; missing columns: {sorted(missing)}")

    session_starts = (
        frame.groupby("session_id", sort=False, as_index=False)["event_time"]
        .min()
        .rename(columns={"event_time": "session_start"})
    )
    if len(session_starts) < 2:
        raise ValueError("at least two sessions are required for a train/test split")
    session_starts["session_sort_key"] = session_starts["session_id"].astype(str)
    session_starts = session_starts.sort_values(
        ["session_start", "session_sort_key"], kind="mergesort"
    )
    n_test = min(max(1, math.ceil(len(session_starts) * test_size)), len(session_starts) - 1)
    test_ids = set(session_starts.iloc[-n_test:]["session_id"])
    test_mask = frame["session_id"].isin(test_ids)
    train = frame.loc[~test_mask].copy()
    test = frame.loc[test_mask].copy()

    train_ids = set(train["session_id"])
    actual_test_ids = set(test["session_id"])
    overlap = train_ids & actual_test_ids
    if overlap:
        raise RuntimeError(f"session leakage detected after split: {len(overlap)} sessions")
    report = {
        "strategy": "chronological_complete_session",
        "requested_test_size": test_size,
        "train_events": int(len(train)),
        "test_events": int(len(test)),
        "train_sessions": int(len(train_ids)),
        "test_sessions": int(len(actual_test_ids)),
        "session_overlap": 0,
        "test_event_share": round(float(len(test) / len(frame)), 6),
        "test_session_share": round(float(len(actual_test_ids) / len(session_starts)), 6),
        "test_session_start_min": session_starts.iloc[-n_test]["session_start"],
    }
    return train, test, report
