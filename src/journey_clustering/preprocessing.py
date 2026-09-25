"""Production clickstream schema adapter.

The production contract is deliberately small: client_time is the event clock,
and the categorical event is exactly ``key + segmentation_name``.  This module
is shared by training and batch inference; the Android implementation mirrors
the same rules and is checked with golden fixtures.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .canonize import canonize_name
from .config import ROOT_SCREENS, CanonizeConfig, SegmentConfig

MISSING = "<missing>"
TOKEN_SEPARATOR = "@"
CANONICAL_COLUMNS = (
    "record_id", "device_id", "customer_id", "session_id", "platform",
    "event_time", "event_type", "segment_name", "screen_id", "event_token",
    "screen_context", "gap_prev_seconds", "gap_next_seconds", "source_file", "source_row_number",
)

AUTH_ACTION_MARKERS: tuple[str, ...] = (
    "continue_login", "login_with", "click_login", "log_out", "logout", "sign_out", "signin_success",
)

PRODUCTION_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "platform": ("segmentation.segment", "segmentation_segment",),
    "segmentation_name": ("segmentation.name",),
    "client_time": ("timestamp",),
    "screen_id": ("segmentation.screen_id", "segmentation_screen_id", ),
    "_id": ("_id.$oid",),
}


def normalize_production_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Accept both flattened exports and the canonical production headers."""

    rename: dict[str, str] = {}
    for target, aliases in PRODUCTION_COLUMN_ALIASES.items():
        if target in raw.columns:
            continue
        source = next((alias for alias in aliases if alias in raw.columns), None)
        if source is not None:
            rename[source] = target
    return raw.rename(columns=rename) if rename else raw


def parse_client_time(values: pd.Series) -> pd.Series:
    """Parse ISO-8601 and epoch-millisecond values into UTC datetimes."""
    raw = values.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    epoch = numeric.notna()
    if epoch.any():
        result.loc[epoch] = pd.to_datetime(numeric.loc[epoch], unit="ms", utc=True, errors="coerce")
    text = ~epoch & raw.notna() & raw.ne("")
    if text.any():
        result.loc[text] = pd.to_datetime(raw.loc[text], utc=True, format="mixed", errors="coerce")
    return result


def normalize_platform(values: pd.Series) -> pd.Series:
    out = values.astype("string").str.strip().str.lower()
    return out.replace({"": "unknown", "nan": "unknown", "none": "unknown", "<na>": "unknown"}).fillna("unknown")


def normalize_event_type(values: pd.Series) -> pd.Series:
    out = values.astype("string").str.strip().str.lower()
    out = out.replace({"[cly]_view": "view", "[cly]_action": "action"})
    return out.replace({"": "unknown", "nan": "unknown", "none": "unknown", "<na>": "unknown"}).fillna("unknown")


def _customer_id(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text or pd.NA


def _stable_id(row: pd.Series) -> str:
    material = "\x1f".join(str(row.get(c, "")) for c in (
        "platform", "session_id", "event_time", "event_type", "segment_name",
        "source_file", "source_row_number",
    ))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def canonicalize_frame(
    raw: pd.DataFrame,
    *,
    source_file: str = "in_memory",
    source_row_offset: int = 0,
    cfg: CanonizeConfig | None = None,
    segment_cfg: SegmentConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map one production-shaped frame to the canonical event contract."""
    cfg = cfg or CanonizeConfig()
    segment_cfg = segment_cfg or SegmentConfig()
    raw = normalize_production_columns(raw)
    required = {"session_id", "platform", "key", "segmentation_name", "client_time"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"production input is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(index=raw.index)
    out["source_file"] = str(source_file)
    out["source_row_number"] = np.arange(source_row_offset, source_row_offset + len(raw), dtype=np.int64)
    out["device_id"] = raw.get("device_id", pd.Series("", index=raw.index)).astype("string").fillna("")
    out["customer_id"] = raw.get("customer_id", pd.Series(pd.NA, index=raw.index)).map(_customer_id).astype("string")
    out["session_id"] = raw["session_id"].astype("string").str.strip()
    out["platform"] = normalize_platform(raw["platform"])
    out["event_time"] = parse_client_time(raw["client_time"])
    out["event_type"] = normalize_event_type(raw["key"])
    out["segment_name"] = [canonize_name(value, cfg=cfg, missing=MISSING) for value in raw["segmentation_name"]]
    screen_ids = raw.get("screen_id", pd.Series(pd.NA, index=raw.index)).astype("string")
    out["screen_id"] = screen_ids
    out["event_token"] = ""
    out["screen_context"] = MISSING

    source_id = raw["_id"].astype("string") if "_id" in raw else pd.Series(pd.NA, index=raw.index, dtype="string")
    out["record_id"] = source_id.where(source_id.notna() & source_id.ne(""), pd.NA)
    missing_id = out["record_id"].isna()
    if missing_id.any():
        out.loc[missing_id, "record_id"] = out.loc[missing_id].apply(_stable_id, axis=1)

    invalid_time = int(out["event_time"].isna().sum())
    invalid_session = int((out["session_id"].isna() | out["session_id"].eq("")).sum())
    out = out.loc[out["event_time"].notna() & out["session_id"].notna() & out["session_id"].ne("")].copy()
    out = _apply_screen_context(out, cfg=cfg, segment_cfg=segment_cfg)
    grouped = out.groupby(["platform", "session_id"], sort=False)["event_time"]
    out["gap_prev_seconds"] = grouped.diff().dt.total_seconds()
    out["gap_next_seconds"] = (grouped.shift(-1) - out["event_time"]).dt.total_seconds()
    report = {
        "rows_input": int(len(raw)),
        "rows_output": int(len(out)),
        "invalid_client_time": invalid_time,
        "invalid_session_id": invalid_session,
        "exact_duplicates": int(out.duplicated(subset=["platform", "session_id", "event_time", "event_type", "segment_name"]).sum()),
        "negative_gap": int((out["gap_prev_seconds"] < 0).sum()),
    }
    return out[list(CANONICAL_COLUMNS)], report


def _is_auth_action(event_type: str, target: str) -> bool:
    if event_type != "action":
        return False
    low = target.lower()
    return any(marker in low for marker in AUTH_ACTION_MARKERS)


def _apply_screen_context(
    frame: pd.DataFrame,
    *,
    cfg: CanonizeConfig,
    segment_cfg: SegmentConfig,
) -> pd.DataFrame:
    """Attach screen-aware tokens after sorting the complete event stream."""
    out = frame.sort_values(
        ["platform", "session_id", "event_time", "source_file", "source_row_number"],
        kind="mergesort",
    ).reset_index(drop=True).copy()
    screen_contexts: list[str] = []
    event_tokens: list[str] = []
    previous_stream: tuple[str, str] | None = None
    previous_time: pd.Timestamp | None = None
    previous_type: str | None = None
    previous_segment_name: str | None = None
    current_context = MISSING
    since_cut = 0

    for row in out.itertuples(index=False):
        stream = (str(row.platform), str(row.session_id))
        same_stream = previous_stream == stream
        gap = (row.event_time - previous_time).total_seconds() if same_stream and previous_time is not None else None
        boundary = None
        if same_stream:
            if gap is not None and gap > segment_cfg.idle_gap_seconds:
                boundary = "idle_gap"
            elif since_cut >= segment_cfg.max_journey_length:
                boundary = "length_cap"
            elif (
                row.event_type == "view"
                and row.segment_name in ROOT_SCREENS
                and previous_segment_name not in ROOT_SCREENS
                and since_cut >= segment_cfg.root_return_min_events
            ):
                boundary = "root_return"
            elif (
                segment_cfg.cut_on_auth_change
                and _is_auth_action(row.event_type, row.segment_name)
                and not _is_auth_action(previous_type or "", previous_segment_name or MISSING)
            ):
                boundary = "auth_change"

        if not same_stream or boundary is not None:
            current_context = MISSING
            since_cut = 0

        explicit_screen = canonize_name(row.screen_id, cfg=cfg, missing=MISSING)
        if explicit_screen != MISSING:
            screen = explicit_screen
        elif row.event_type == "view":
            screen = row.segment_name
        else:
            screen = current_context

        screen_contexts.append(screen)
        event_tokens.append(
            f"{row.event_type}{TOKEN_SEPARATOR}{screen}"
            if row.event_type == "view"
            else f"{row.event_type}{TOKEN_SEPARATOR}{screen}#{row.segment_name}"
        )
        if row.event_type == "view":
            current_context = screen

        previous_stream = stream
        previous_time = row.event_time
        previous_type = row.event_type
        previous_segment_name = row.segment_name
        since_cut += 1

    out["screen_context"] = screen_contexts
    out["event_token"] = event_tokens
    return out


def read_production_folder(
    folder: Path,
    *,
    chunksize: int = 250_000,
    cfg: CanonizeConfig | None = None,
    segment_cfg: SegmentConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every raw CSV below folder and return canonical events + file report."""
    frames: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    cfg = cfg or CanonizeConfig()
    segment_cfg = segment_cfg or SegmentConfig()
    for path in sorted(Path(folder).rglob("*.csv")):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        offset = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            canonical, report = canonicalize_frame(
                chunk,
                source_file=str(path.relative_to(folder)),
                source_row_offset=offset,
                cfg=cfg,
                segment_cfg=segment_cfg,
            )
            frames.append(canonical)
            reports.append({"source_file": str(path.relative_to(folder)), "chunk_start": offset, **report})
            offset += len(chunk)
    if not frames:
        raise ValueError(f"no production CSV files found below {folder}")
    combined = pd.concat(frames, ignore_index=True)
    combined = _apply_screen_context(combined, cfg=cfg, segment_cfg=segment_cfg)
    combined = combined.sort_values(
        ["platform", "session_id", "event_time", "source_file", "source_row_number"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = combined.groupby(["platform", "session_id"], sort=False)["event_time"]
    combined["gap_prev_seconds"] = grouped.diff().dt.total_seconds()
    combined["gap_next_seconds"] = (grouped.shift(-1) - combined["event_time"]).dt.total_seconds()
    return combined, pd.DataFrame(reports)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read production CSV files and report canonical events.")
    parser.add_argument("folder", type=Path, help="Folder containing production CSV files.")
    parser.add_argument("--chunksize", type=int, default=250_000, help="Number of rows to read at a time.")
    args = parser.parse_args()

    events, report = read_production_folder(args.folder, chunksize=args.chunksize)
    print(f"Read {len(events)} canonical events from {len(report)} source files.")
    print(report)
