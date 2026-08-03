"""Reusable helpers for exact event exploration and tokenization.

The functions in this module intentionally preserve source values. Any parsing,
session splitting, or token formatting is written to derived columns only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: list[str] = [
    "record_id",
    "device_id",
    "customer_id",
    "session_id",
    "event_type",
    "timestamp",
    "created_at",
    "segment_name",
    "screen_name",
    "duration",
]

TOKEN_SOURCE_COLUMNS: list[str] = ["event_type", "segment_name", "screen_name"]
SOURCE_ID_COLUMNS: list[str] = ["record_id", "device_id", "customer_id", "session_id"]
MISSING_SENTINEL = "<MISSING>"
RANDOM_SEED = 42
INACTIVITY_THRESHOLDS_MINUTES: list[int] = [15, 30, 60]
SELECTED_INACTIVITY_THRESHOLD_MINUTES = 30
VIEW_EVENT_TYPES: tuple[str, ...] = ("View",)
ACTION_EVENT_TYPES: tuple[str, ...] = ("Action",)


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by looking for the input CSV."""
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "final_clean_events.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find data/final_clean_events.csv from the current path.")


def input_csv_path(repo_root: Path) -> Path:
    """Return the repository-relative input CSV path."""
    return repo_root / "data" / "final_clean_events.csv"


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return size, mtime, and SHA-256 for a file without modifying it."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": hasher.hexdigest()}


def read_events(path: Path) -> pd.DataFrame:
    """Read source events while preserving source fields as string-backed values."""
    df = pd.read_csv(path, dtype="string", keep_default_na=True)
    df.insert(0, "source_row_number", np.arange(len(df), dtype=np.int64))
    return df


def inferred_dtypes(path: Path) -> pd.Series:
    """Infer CSV data types using pandas defaults for documentation only."""
    inferred = pd.read_csv(path, low_memory=False)
    return inferred.dtypes.astype(str)


def validate_required_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Raise a clear error if any required columns are absent."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")


def validate_sentinel_absent(df: pd.DataFrame, sentinel: str = MISSING_SENTINEL) -> None:
    """Ensure the missing-value sentinel cannot collide with a real source value."""
    collisions: dict[str, int] = {}
    for column in TOKEN_SOURCE_COLUMNS:
        collisions[column] = int((df[column].eq(sentinel)).fillna(False).sum())
    collisions = {column: count for column, count in collisions.items() if count}
    if collisions:
        raise ValueError(f"Token missing sentinel collides with real source values: {collisions}")


def mask_identifier(value: Any, prefix: int = 6, suffix: int = 4) -> str:
    """Shorten identifiers for display while preserving traceability in memory."""
    if pd.isna(value):
        return MISSING_SENTINEL
    text = str(value)
    if len(text) <= prefix + suffix + 3:
        return text
    return f"{text[:prefix]}...{text[-suffix:]}"


def masked_sample(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return a small sample with long identifiers shortened for display."""
    sample = df.head(n).copy()
    for column in SOURCE_ID_COLUMNS:
        if column in sample.columns:
            sample[column] = sample[column].map(mask_identifier)
    return sample


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values by column."""
    missing_count = df.isna().sum()
    return pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": (missing_count / len(df) * 100).round(4),
        }
    ).sort_values(["missing_count", "missing_percentage"], ascending=False)


def availability_by_event_type(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate source-field availability grouped by exact event_type."""
    grouped = df.groupby("event_type", dropna=False)
    rows = grouped.size().rename("rows")
    result = rows.to_frame()
    for column in ["segment_name", "screen_name", "duration"]:
        result[f"{column}_available"] = grouped[column].apply(lambda series: int(series.notna().sum()))
        result[f"{column}_missing"] = grouped[column].apply(lambda series: int(series.isna().sum()))
        result[f"{column}_available_pct"] = (result[f"{column}_available"] / result["rows"] * 100).round(4)
    return result.reset_index()


def value_counts_with_pct(series: pd.Series, top_n: int | None = None) -> pd.DataFrame:
    """Return exact value counts and percentages without altering source values."""
    counts = series.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    table = counts.rename("count").to_frame()
    table["percentage"] = (table["count"] / len(series) * 100).round(4)
    table = table.reset_index()
    table = table.rename(columns={table.columns[0]: series.name or "value"})
    return table


def exact_tuple_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count exact (event_type, segment_name, screen_name) combinations."""
    counts = (
        df.groupby(TOKEN_SOURCE_COLUMNS, dropna=False)
        .size()
        .rename("event_frequency")
        .reset_index()
        .sort_values("event_frequency", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    counts["percentage"] = (counts["event_frequency"] / len(df) * 100).round(4)
    return counts


def component_by_event_type(df: pd.DataFrame, component_column: str, top_n: int = 10) -> pd.DataFrame:
    """Return top component values within each exact event_type."""
    counts = (
        df.groupby(["event_type", component_column], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["event_type", "count", component_column], ascending=[True, False, True], kind="mergesort")
    )
    return counts.groupby("event_type", dropna=False).head(top_n).reset_index(drop=True)


def _parse_epoch(numeric: pd.Series, unit: str) -> pd.Series:
    """Parse numeric epoch timestamps with invalid values coerced to NaT."""
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def determine_timestamp_unit(timestamp_numeric: pd.Series, created_at_datetime: pd.Series) -> tuple[str, pd.DataFrame]:
    """Choose the timestamp unit from evidence, preferring agreement with created_at."""
    rows: list[dict[str, Any]] = []
    created_valid = created_at_datetime.notna()
    for unit in ["s", "ms", "us", "ns"]:
        parsed = _parse_epoch(timestamp_numeric, unit)
        both_valid = parsed.notna() & created_valid
        median_abs_diff = np.nan
        if both_valid.any():
            median_abs_diff = float((parsed[both_valid] - created_at_datetime[both_valid]).abs().dt.total_seconds().median())
        rows.append(
            {
                "candidate_unit": unit,
                "valid_count": int(parsed.notna().sum()),
                "min_parsed": parsed.min(),
                "max_parsed": parsed.max(),
                "median_abs_diff_vs_created_at_seconds": median_abs_diff,
            }
        )
    evidence = pd.DataFrame(rows)
    ranked = evidence.assign(
        diff_rank=evidence["median_abs_diff_vs_created_at_seconds"].fillna(np.inf),
        valid_rank=-evidence["valid_count"],
    ).sort_values(["diff_rank", "valid_rank", "candidate_unit"], kind="mergesort")
    selected = str(ranked.iloc[0]["candidate_unit"])
    return selected, evidence


def add_time_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add derived datetime columns without overwriting source timestamp fields."""
    result = df.copy()
    result["timestamp_numeric"] = pd.to_numeric(result["timestamp"], errors="coerce")
    result["created_at_datetime"] = pd.to_datetime(result["created_at"], utc=True, errors="coerce")
    selected_unit, evidence = determine_timestamp_unit(result["timestamp_numeric"], result["created_at_datetime"])
    result["timestamp_datetime"] = _parse_epoch(result["timestamp_numeric"], selected_unit)
    result["timestamp_disagreement_seconds"] = (
        result["timestamp_datetime"] - result["created_at_datetime"]
    ).abs().dt.total_seconds()

    timestamp_valid = int(result["timestamp_datetime"].notna().sum())
    created_valid = int(result["created_at_datetime"].notna().sum())
    timestamp_unique = int(result["timestamp_datetime"].nunique(dropna=True))
    created_unique = int(result["created_at_datetime"].nunique(dropna=True))
    if timestamp_valid >= created_valid and timestamp_unique >= created_unique:
        primary = "timestamp_datetime"
        secondary = "created_at_datetime"
        reason = "timestamp has at least as many valid values and at least as many distinct event times as created_at"
    else:
        primary = "created_at_datetime"
        secondary = "timestamp_datetime"
        reason = "created_at has stronger validity or distinctness evidence than timestamp"

    result["primary_event_time"] = result[primary]
    result["secondary_event_time"] = result[secondary]
    metadata = {
        "selected_timestamp_unit": selected_unit,
        "timestamp_unit_evidence": evidence,
        "primary_event_time_column": primary,
        "secondary_event_time_column": secondary,
        "primary_event_time_reason": reason,
        "timestamp_valid_count": timestamp_valid,
        "created_at_valid_count": created_valid,
        "timestamp_unique_count": timestamp_unique,
        "created_at_unique_count": created_unique,
    }
    return result, metadata


def add_duration_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived duration fields without capping, filling, or deleting source values."""
    result = df.copy()
    result["duration_numeric"] = pd.to_numeric(result["duration"], errors="coerce")
    result["duration_missing"] = result["duration_numeric"].isna()
    result["duration_zero"] = result["duration_numeric"].eq(0)
    result["duration_negative"] = result["duration_numeric"].lt(0)
    result["duration_positive"] = result["duration_numeric"].gt(0)
    valid_for_log = result["duration_numeric"].where(result["duration_numeric"].ge(0))
    result["log1p_duration"] = np.log1p(valid_for_log)
    positive = result.loc[result["duration_numeric"].gt(0), "duration_numeric"]
    outlier_threshold = positive.quantile(0.99) if len(positive) else np.nan
    result["duration_outlier"] = result["duration_numeric"].gt(outlier_threshold) if pd.notna(outlier_threshold) else False
    return result


def sort_events(df: pd.DataFrame) -> pd.DataFrame:
    """Sort events deterministically within original sessions."""
    sort_columns = ["session_id", "primary_event_time", "secondary_event_time", "record_id", "source_row_number"]
    return df.sort_values(sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)


def original_order_timestamp_issues(df: pd.DataFrame) -> dict[str, Any]:
    """Find invalid or decreasing event times before deterministic sorting."""
    working = df.sort_values(["session_id", "source_row_number"], kind="mergesort", na_position="last").copy()
    previous_time = working.groupby("session_id", dropna=False)["primary_event_time"].shift()
    delta = (working["primary_event_time"] - previous_time).dt.total_seconds()
    decreasing = delta.lt(0).fillna(False)
    invalid_current = working["primary_event_time"].isna()
    invalid_pair = (~previous_time.isna()) & invalid_current
    return {
        "invalid_primary_event_time_rows": int(invalid_current.sum()),
        "invalid_or_missing_pair_rows": int(invalid_pair.sum()),
        "decreasing_timestamp_rows_before_sort": int(decreasing.sum()),
        "sample_decreasing_rows": working.loc[decreasing, ["record_id", "session_id", "primary_event_time", "source_row_number"]].head(10),
    }


def session_group_key(series: pd.Series) -> pd.Series:
    """Return a derived group key that makes missing session IDs explicit."""
    return series.astype("object").where(series.notna(), MISSING_SENTINEL)


def add_original_session_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add original-session event indexes and gap fields."""
    result = df.copy()
    group_key = session_group_key(result["session_id"])
    result["event_index_in_original_session"] = result.groupby(group_key, dropna=False).cumcount().astype("int64")
    previous_time = result.groupby(group_key, dropna=False)["primary_event_time"].shift()
    next_time = result.groupby(group_key, dropna=False)["primary_event_time"].shift(-1)
    result["time_gap_from_previous_seconds"] = (result["primary_event_time"] - previous_time).dt.total_seconds()
    result["time_gap_to_next_seconds"] = (next_time - result["primary_event_time"]).dt.total_seconds()
    return result


def original_session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize original logging sessions."""
    working = add_original_session_context(df)
    group_key = session_group_key(working["session_id"])
    summary = (
        working.groupby(group_key, dropna=False)
        .agg(
            original_session_id=("session_id", "first"),
            event_count=("record_id", "size"),
            session_start=("primary_event_time", "min"),
            session_end=("primary_event_time", "max"),
            non_null_device_count=("device_id", lambda s: int(s.dropna().nunique())),
            non_null_customer_count=("customer_id", lambda s: int(s.dropna().nunique())),
            max_inter_event_gap_seconds=("time_gap_from_previous_seconds", "max"),
        )
        .reset_index(drop=True)
    )
    summary["session_span_seconds"] = (summary["session_end"] - summary["session_start"]).dt.total_seconds()
    return summary


def threshold_comparison(df: pd.DataFrame, thresholds: Iterable[int] = INACTIVITY_THRESHOLDS_MINUTES) -> pd.DataFrame:
    """Compare inactivity split counts for candidate thresholds."""
    working = add_original_session_context(df)
    gaps = working["time_gap_from_previous_seconds"].dropna()
    rows: list[dict[str, Any]] = []
    original_sessions = int(session_group_key(working["session_id"]).nunique(dropna=False))
    for threshold in thresholds:
        split_count = int(gaps.gt(threshold * 60).sum())
        rows.append(
            {
                "threshold_minutes": threshold,
                "gaps_exceeding_threshold": split_count,
                "percentage_of_observed_gaps": round(split_count / len(gaps) * 100, 4) if len(gaps) else 0.0,
                "resulting_clean_sessions_if_gap_only": original_sessions + split_count,
            }
        )
    return pd.DataFrame(rows)


def construct_clean_sessions(
    df: pd.DataFrame,
    threshold_minutes: int = SELECTED_INACTIVITY_THRESHOLD_MINUTES,
) -> pd.DataFrame:
    """Construct derived clean sessions using configurable split conditions."""
    result = add_original_session_context(df)
    group_key = session_group_key(result["session_id"])
    previous_device = result.groupby(group_key, dropna=False)["device_id"].shift()
    previous_customer = result.groupby(group_key, dropna=False)["customer_id"].shift()
    previous_time = result.groupby(group_key, dropna=False)["primary_event_time"].shift()
    first_in_original = result["event_index_in_original_session"].eq(0)

    inactivity = result["time_gap_from_previous_seconds"].gt(threshold_minutes * 60).fillna(False)
    device_change = previous_device.notna() & result["device_id"].notna() & previous_device.ne(result["device_id"])
    customer_change = previous_customer.notna() & result["customer_id"].notna() & previous_customer.ne(result["customer_id"])
    invalid_or_discontinuous_time = (~first_in_original) & (
        result["primary_event_time"].isna()
        | previous_time.isna()
        | result["time_gap_from_previous_seconds"].lt(0).fillna(False)
    )

    reasons: list[str] = []
    for is_first, is_inactive, dev_changed, cust_changed, bad_time in zip(
        first_in_original, inactivity, device_change, customer_change, invalid_or_discontinuous_time
    ):
        row_reasons: list[str] = []
        if bool(is_first):
            row_reasons.append("original_session_start")
        if bool(is_inactive):
            row_reasons.append(f"inactivity>{threshold_minutes}m")
        if bool(dev_changed):
            row_reasons.append("known_device_changed")
        if bool(cust_changed):
            row_reasons.append("known_customer_changed")
        if bool(bad_time):
            row_reasons.append("invalid_or_discontinuous_time")
        reasons.append(";".join(row_reasons) if row_reasons else "")

    result["session_split_flag"] = (~first_in_original) & (inactivity | device_change | customer_change | invalid_or_discontinuous_time)
    result["session_split_reason"] = reasons
    result["clean_session_index"] = result.groupby(group_key, dropna=False)["session_split_flag"].cumsum().astype("int64")
    session_label = group_key.astype(str)
    result["clean_session_id"] = session_label + "::clean_" + result["clean_session_index"].astype(str)
    result["event_index_in_clean_session"] = (
        result.groupby("clean_session_id", dropna=False).cumcount().astype("int64")
    )
    return result


def _token_value(value: Any, sentinel: str = MISSING_SENTINEL) -> str:
    """Convert one token field to its serialized representation."""
    if pd.isna(value):
        return sentinel
    return str(value)


def serialize_exact_token_values(
    event_type: Any,
    segment_name: Any,
    screen_name: Any,
    sentinel: str = MISSING_SENTINEL,
) -> str:
    """Serialize an exact event tuple as a reversible JSON array."""
    return json.dumps(
        [_token_value(event_type, sentinel), _token_value(segment_name, sentinel), _token_value(screen_name, sentinel)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def add_exact_tokens(df: pd.DataFrame, sentinel: str = MISSING_SENTINEL) -> pd.DataFrame:
    """Add reversible exact-token strings from source event tuple fields."""
    validate_sentinel_absent(df, sentinel)
    result = df.copy()
    result["exact_token"] = [
        serialize_exact_token_values(event_type, segment_name, screen_name, sentinel)
        for event_type, segment_name, screen_name in zip(
            result["event_type"], result["segment_name"], result["screen_name"]
        )
    ]
    return result


def build_token_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a deterministic token dictionary from exact serialized tuples."""
    if "exact_token" not in df.columns:
        df = add_exact_tokens(df)

    unique_tokens = (
        df[["exact_token", *TOKEN_SOURCE_COLUMNS]]
        .drop_duplicates("exact_token")
        .sort_values("exact_token", kind="mergesort")
        .reset_index(drop=True)
    )
    unique_tokens.insert(0, "token_id", np.arange(1, len(unique_tokens) + 1, dtype=np.int64))

    total_frequency = df.groupby("exact_token", dropna=False).size().rename("total_event_frequency")
    original_session_frequency = (
        df.groupby("exact_token", dropna=False)["session_id"].nunique(dropna=True).rename("original_session_frequency")
    )
    clean_session_frequency = None
    if "clean_session_id" in df.columns:
        clean_session_frequency = (
            df.groupby("exact_token", dropna=False)["clean_session_id"].nunique(dropna=True).rename("clean_session_frequency")
        )
    first_last = df.groupby("exact_token", dropna=False)["primary_event_time"].agg(
        first_observed_timestamp="min", last_observed_timestamp="max"
    )

    dictionary = unique_tokens.merge(total_frequency, on="exact_token", how="left")
    dictionary = dictionary.merge(original_session_frequency, on="exact_token", how="left")
    if clean_session_frequency is not None:
        dictionary = dictionary.merge(clean_session_frequency, on="exact_token", how="left")
    dictionary = dictionary.merge(first_last, on="exact_token", how="left")
    return dictionary.sort_values("token_id", kind="mergesort").reset_index(drop=True)


def add_token_ids(df: pd.DataFrame, token_dictionary: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic token IDs and neighbor/context fields."""
    result = df.merge(token_dictionary[["token_id", "exact_token"]], on="exact_token", how="left", validate="many_to_one")
    group_key = session_group_key(result["session_id"])
    result["previous_token_id"] = result.groupby(group_key, dropna=False)["token_id"].shift()
    result["next_token_id"] = result.groupby(group_key, dropna=False)["token_id"].shift(-1)
    result["is_consecutive_exact_repeat"] = result["token_id"].eq(result["previous_token_id"]).fillna(False)

    previous_view_token: list[Any] = []
    seconds_since_previous_view: list[float] = []
    for _, group in result.groupby(group_key, dropna=False, sort=False):
        last_view_token: Any = pd.NA
        last_view_time: Any = pd.NaT
        for _, row in group.iterrows():
            previous_view_token.append(last_view_token)
            if pd.notna(last_view_time) and pd.notna(row["primary_event_time"]):
                seconds_since_previous_view.append(float((row["primary_event_time"] - last_view_time).total_seconds()))
            else:
                seconds_since_previous_view.append(np.nan)
            if row["event_type"] in VIEW_EVENT_TYPES:
                last_view_token = row["token_id"]
                last_view_time = row["primary_event_time"]
    result["previous_view_token_id"] = previous_view_token
    result["seconds_since_previous_view"] = seconds_since_previous_view
    result["segment_name_missing"] = result["segment_name"].isna()
    result["screen_name_missing"] = result["screen_name"].isna()
    result["event_type_missing"] = result["event_type"].isna()
    return result


def compress_token_sequence(tokens: list[int]) -> list[dict[str, int]]:
    """Run-length encode consecutive exact token repeats."""
    compressed: list[dict[str, int]] = []
    for token in tokens:
        token = int(token)
        if compressed and compressed[-1]["token_id"] == token:
            compressed[-1]["repeat_count"] += 1
        else:
            compressed.append({"token_id": token, "repeat_count": 1})
    return compressed


def build_session_sequences(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Construct raw and compressed token sequences with per-session metrics."""
    rows: list[dict[str, Any]] = []
    for session_id, group in df.groupby(group_column, dropna=False, sort=False):
        token_sequence = [int(token) for token in group["token_id"].tolist()]
        compressed = compress_token_sequence(token_sequence)
        event_count = len(token_sequence)
        compressed_count = len(compressed)
        repeat_count = int(group["is_consecutive_exact_repeat"].sum())
        start = group["primary_event_time"].min()
        end = group["primary_event_time"].max()
        duration_seconds = (end - start).total_seconds() if pd.notna(start) and pd.notna(end) else np.nan
        rows.append(
            {
                group_column: session_id,
                "raw_token_sequence": token_sequence,
                "compressed_token_sequence": compressed,
                "raw_event_count": event_count,
                "compressed_event_count": compressed_count,
                "unique_exact_token_count": int(group["token_id"].nunique()),
                "view_count": int(group["event_type"].isin(VIEW_EVENT_TYPES).sum()),
                "action_count": int(group["event_type"].isin(ACTION_EVENT_TYPES).sum()),
                "missing_segment_count": int(group["segment_name"].isna().sum()),
                "missing_screen_count": int(group["screen_name"].isna().sum()),
                "session_start": start,
                "session_end": end,
                "session_duration_seconds": duration_seconds,
                "maximum_inter_event_gap_seconds": float(group["time_gap_from_previous_seconds"].max())
                if group["time_gap_from_previous_seconds"].notna().any()
                else np.nan,
                "exact_repeat_count": repeat_count,
                "compression_ratio_compressed_to_raw": round(compressed_count / event_count, 6) if event_count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def readable_sequence(
    token_sequence: list[int],
    token_dictionary: pd.DataFrame,
    max_items: int = 20,
) -> list[str]:
    """Map token IDs to readable exact-token strings, truncating long sequences."""
    lookup = token_dictionary.set_index("token_id")["exact_token"].to_dict()
    visible = [lookup.get(int(token), f"<UNKNOWN:{token}>") for token in token_sequence[:max_items]]
    if len(token_sequence) > max_items:
        visible.append(f"... ({len(token_sequence) - max_items} more)")
    return visible


def representative_sequences(
    sequences: pd.DataFrame,
    tokenized_events: pd.DataFrame,
    group_column: str,
    token_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Select representative short, median, long, repeated, and missing-field sessions."""
    selected: list[tuple[str, Any]] = []
    if sequences.empty:
        return pd.DataFrame()

    sorted_by_length = sequences.sort_values("raw_event_count", kind="mergesort")
    selected.append(("short_session", sorted_by_length.iloc[0][group_column]))
    median_length = sequences["raw_event_count"].median()
    median_idx = (sequences["raw_event_count"] - median_length).abs().sort_values(kind="mergesort").index[0]
    selected.append(("median_length_session", sequences.loc[median_idx, group_column]))
    selected.append(("long_session", sorted_by_length.iloc[-1][group_column]))

    repeated = sequences.loc[sequences["exact_repeat_count"].gt(0)]
    if not repeated.empty:
        selected.append(("session_with_repeated_exact_tokens", repeated.sort_values("exact_repeat_count", ascending=False).iloc[0][group_column]))

    missing_fields = sequences.loc[sequences["missing_segment_count"].gt(0) | sequences["missing_screen_count"].gt(0)]
    if not missing_fields.empty:
        selected.append(("session_with_missing_token_fields", missing_fields.iloc[0][group_column]))

    if group_column == "clean_session_id" and "session_split_reason" in tokenized_events.columns:
        split_rows = tokenized_events.loc[tokenized_events["session_split_reason"].astype("string").str.contains("inactivity", na=False)]
        if not split_rows.empty:
            selected.append(("session_split_by_inactivity", split_rows.iloc[0][group_column]))

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for label, session_id in selected:
        key = (label, str(session_id))
        if key in seen:
            continue
        seen.add(key)
        row = sequences.loc[sequences[group_column].astype(str).eq(str(session_id))].iloc[0]
        records.append(
            {
                "example_type": label,
                group_column: session_id,
                "raw_event_count": row["raw_event_count"],
                "compressed_event_count": row["compressed_event_count"],
                "compression_ratio_compressed_to_raw": row["compression_ratio_compressed_to_raw"],
                "raw_token_sequence_preview": row["raw_token_sequence"][:20],
                "readable_sequence_preview": readable_sequence(row["raw_token_sequence"], token_dictionary, max_items=20),
            }
        )
    return pd.DataFrame(records)


def dataframe_fingerprint(df: pd.DataFrame, columns: list[str]) -> str:
    """Create a stable fingerprint for selected DataFrame columns."""
    payload = df[columns].astype("string").fillna(MISSING_SENTINEL).to_json(orient="records", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_tokenization(
    source_df: pd.DataFrame,
    tokenized_events: pd.DataFrame,
    token_dictionary: pd.DataFrame,
    original_sequences: pd.DataFrame,
    clean_sequences: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run assertions required for exact tokenization and session sequence validation."""
    results: dict[str, Any] = {}
    results["source_rows"] = len(source_df)
    results["tokenized_rows"] = len(tokenized_events)
    assert len(source_df) == len(tokenized_events), "No input rows should disappear during tokenization."
    assert tokenized_events["record_id"].notna().all(), "record_id must remain traceable for every row."
    assert tokenized_events["token_id"].notna().all(), "Every row must receive exactly one token_id."
    assert not token_dictionary["token_id"].duplicated().any(), "Every token_id must map to one tuple."
    assert not token_dictionary["exact_token"].duplicated().any(), "Every exact tuple must map to one token_id."

    for exact_token in token_dictionary["exact_token"]:
        parsed = json.loads(exact_token)
        assert isinstance(parsed, list) and len(parsed) == 3, f"Token is not a three-field JSON array: {exact_token}"

    restored = tokenized_events.sort_values("source_row_number", kind="mergesort").reset_index(drop=True)
    source_without_index = source_df.drop(columns=["source_row_number"]).reset_index(drop=True)
    restored_source = restored[source_without_index.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        restored_source,
        source_without_index,
        check_dtype=False,
        check_names=True,
    )

    expected_sorted = sort_events(tokenized_events).reset_index(drop=True)
    assert tokenized_events["source_row_number"].tolist() == expected_sorted["source_row_number"].tolist(), (
        "Events are not in deterministic order."
    )

    original_counts = tokenized_events.groupby(session_group_key(tokenized_events["session_id"]), dropna=False).size()
    sequence_counts = original_sequences.set_index("session_id")["raw_event_count"]
    sequence_counts.index = sequence_counts.index.astype(str)
    original_counts.index = original_counts.index.astype(str)
    assert original_counts.sort_index().equals(sequence_counts.sort_index()), "Original session counts do not reconcile."
    assert (original_sequences["raw_event_count"] == original_sequences["raw_token_sequence"].map(len)).all()
    assert (original_sequences["compressed_event_count"] <= original_sequences["raw_event_count"]).all()

    if clean_sequences is not None:
        clean_counts = tokenized_events.groupby("clean_session_id", dropna=False).size()
        clean_sequence_counts = clean_sequences.set_index("clean_session_id")["raw_event_count"]
        assert clean_counts.sort_index().equals(clean_sequence_counts.sort_index()), "Clean session counts do not reconcile."
        assert (clean_sequences["raw_event_count"] == clean_sequences["raw_token_sequence"].map(len)).all()
        assert (clean_sequences["compressed_event_count"] <= clean_sequences["raw_event_count"]).all()

    regenerated = build_token_dictionary(tokenized_events.drop(columns=["token_id"], errors="ignore"))
    pd.testing.assert_frame_equal(
        regenerated[["token_id", "exact_token"]],
        token_dictionary[["token_id", "exact_token"]],
        check_dtype=False,
    )

    results["token_vocabulary_size"] = int(len(token_dictionary))
    results["original_sequence_count"] = int(len(original_sequences))
    results["clean_sequence_count"] = int(len(clean_sequences)) if clean_sequences is not None else None
    results["validation_status"] = "passed"
    return results


def quantile_table(series: pd.Series, quantiles: Iterable[float]) -> pd.DataFrame:
    """Return named quantiles for a numeric series."""
    clean = series.dropna()
    values = clean.quantile(list(quantiles)) if len(clean) else pd.Series(dtype="float64")
    return pd.DataFrame({"quantile": values.index, "value": values.values})


def safe_json(value: Any) -> str:
    """Serialize values for readable notebook output."""
    return json.dumps(value, ensure_ascii=False, default=str, indent=2)


def json_ready(value: Any) -> Any:
    """Convert pandas/numpy objects into JSON-serializable Python values."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as JSON Lines, preserving nested list/dict sequence fields."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in df.to_dict(orient="records"):
            handle.write(json.dumps(json_ready(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def save_token_outputs(
    token_dictionary: pd.DataFrame,
    original_sequences: pd.DataFrame,
    clean_sequences: pd.DataFrame | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Save token dictionary and tokenized session sequences to output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    token_dictionary_csv = output_dir / "token_dictionary.csv"
    token_dictionary_jsonl = output_dir / "token_dictionary.jsonl"
    original_sequences_jsonl = output_dir / "original_session_sequences.jsonl"
    manifest_path = output_dir / "tokenized_output_manifest.json"

    token_dictionary.to_csv(token_dictionary_csv, index=False)
    write_jsonl(token_dictionary, token_dictionary_jsonl)
    write_jsonl(original_sequences, original_sequences_jsonl)

    files: list[dict[str, Any]] = [
        {"name": token_dictionary_csv.name, "path": str(token_dictionary_csv), "rows": int(len(token_dictionary))},
        {"name": token_dictionary_jsonl.name, "path": str(token_dictionary_jsonl), "rows": int(len(token_dictionary))},
        {"name": original_sequences_jsonl.name, "path": str(original_sequences_jsonl), "rows": int(len(original_sequences))},
    ]

    if clean_sequences is not None:
        clean_sequences_jsonl = output_dir / "clean_session_sequences.jsonl"
        write_jsonl(clean_sequences, clean_sequences_jsonl)
        files.append({"name": clean_sequences_jsonl.name, "path": str(clean_sequences_jsonl), "rows": int(len(clean_sequences))})

    manifest = {
        "output_directory": str(output_dir),
        "files": files,
        "token_vocabulary_size": int(len(token_dictionary)),
        "original_session_sequence_count": int(len(original_sequences)),
        "clean_session_sequence_count": int(len(clean_sequences)) if clean_sequences is not None else None,
        "format_notes": {
            "token_dictionary_csv": "Flat CSV for review and joins; original missing values are blank in source tuple columns.",
            "jsonl_files": "One JSON object per line; raw and compressed sequence fields remain nested arrays.",
        },
    }
    manifest_path.write_text(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["files"].append({"name": manifest_path.name, "path": str(manifest_path), "rows": 1})
    return manifest


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a compact Markdown table without optional third-party packages."""
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
    display = display.astype("string").fillna(MISSING_SENTINEL)
    display = display.replace({"\n": " "}, regex=True)
    headers = [str(column) for column in display.columns]

    def clean_cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(clean_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(clean_cell(row[column]) for column in display.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)
