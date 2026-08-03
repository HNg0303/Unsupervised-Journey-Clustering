"""Build exact-event exploration notebooks and Markdown documentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd

REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from utils.exact_event_analysis import (
    ACTION_EVENT_TYPES,
    INACTIVITY_THRESHOLDS_MINUTES,
    MISSING_SENTINEL,
    REQUIRED_COLUMNS,
    SELECTED_INACTIVITY_THRESHOLD_MINUTES,
    VIEW_EVENT_TYPES,
    add_duration_fields,
    add_exact_tokens,
    add_time_fields,
    add_token_ids,
    availability_by_event_type,
    build_session_sequences,
    build_token_dictionary,
    component_by_event_type,
    construct_clean_sessions,
    exact_tuple_counts,
    file_fingerprint,
    find_repo_root,
    inferred_dtypes,
    input_csv_path,
    markdown_table,
    masked_sample,
    missing_summary,
    original_order_timestamp_issues,
    original_session_summary,
    quantile_table,
    read_events,
    representative_sequences,
    safe_json,
    save_token_outputs,
    sort_events,
    threshold_comparison,
    validate_required_columns,
    validate_tokenization,
    value_counts_with_pct,
)


def code_cell(source: str) -> dict:
    """Create a code cell."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dedent(source).strip() + "\n"}


def markdown_cell(source: str) -> dict:
    """Create a Markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def notebook(cells: list[dict]) -> dict:
    """Create a minimal executable notebook document."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_IMPORTS = r"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

WORKING_DIR = Path.cwd()
REPO_ROOT = WORKING_DIR if (WORKING_DIR / "data" / "final_clean_events.csv").exists() else WORKING_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.exact_event_analysis import (
    ACTION_EVENT_TYPES,
    INACTIVITY_THRESHOLDS_MINUTES,
    MISSING_SENTINEL,
    REQUIRED_COLUMNS,
    SELECTED_INACTIVITY_THRESHOLD_MINUTES,
    VIEW_EVENT_TYPES,
    add_duration_fields,
    add_exact_tokens,
    add_time_fields,
    add_token_ids,
    availability_by_event_type,
    build_session_sequences,
    build_token_dictionary,
    component_by_event_type,
    construct_clean_sessions,
    exact_tuple_counts,
    file_fingerprint,
    inferred_dtypes,
    input_csv_path,
    markdown_table,
    masked_sample,
    missing_summary,
    original_order_timestamp_issues,
    original_session_summary,
    quantile_table,
    read_events,
    representative_sequences,
    safe_json,
    save_token_outputs,
    sort_events,
    threshold_comparison,
    validate_required_columns,
    validate_tokenization,
    value_counts_with_pct,
)

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 160)
np.random.seed(42)

INPUT_CSV = input_csv_path(REPO_ROOT)
print(f"Repository root: {REPO_ROOT}")
print(f"Input CSV: {INPUT_CSV.relative_to(REPO_ROOT)}")
"""


def data_exploration_notebook() -> dict:
    """Build Part 1 notebook."""
    cells = [
        markdown_cell(
            """
            # Part 1 - Data Exploration

            This notebook reads `data/final_clean_events.csv` directly and preserves source values exactly.
            Source columns are not lowercased, merged, simplified, imputed, or overwritten. Derived parsing
            columns are added separately for analysis.
            """
        ),
        markdown_cell(
            """
            ## Configuration and imports

            Important thresholds and constants are defined in the reusable helper module. The fixed random
            seed is set here even though this notebook is deterministic, so later sampling remains repeatable.
            """
        ),
        code_cell(COMMON_IMPORTS),
        markdown_cell(
            """
            ## Load source data and validate required columns

            The CSV is loaded with string-backed source columns to avoid changing identifiers or component
            labels. Numeric and datetime interpretations are derived later.
            """
        ),
        code_cell(
            """
            before_fingerprint = file_fingerprint(INPUT_CSV)
            events = read_events(INPUT_CSV)
            print(f"Rows after load: {len(events):,}")
            print(f"Columns after load: {events.shape[1]:,}")
            validate_required_columns(events)
            print("Required-column validation: passed")
            print("Required columns:", REQUIRED_COLUMNS)
            """
        ),
        markdown_cell(
            """
            ## Dataset structure

            The following tables document row and column counts, inferred data types, memory usage, sample
            records, duplicate identifiers, and exact component cardinalities.
            """
        ),
        code_cell(
            """
            inferred = inferred_dtypes(INPUT_CSV).rename("inferred_dtype").reset_index()
            inferred.columns = ["column", "inferred_dtype"]
            overview = {
                "rows": len(events),
                "columns": events.shape[1],
                "memory_usage_mb_preserved_load": round(events.memory_usage(deep=True).sum() / 1024**2, 3),
                "duplicate_record_id_rows": int(events["record_id"].duplicated(keep=False).sum()),
                "duplicate_record_id_excess": int(events["record_id"].duplicated().sum()),
                "fully_duplicated_rows": int(events.drop(columns=["source_row_number"]).duplicated().sum()),
                "devices": int(events["device_id"].nunique(dropna=True)),
                "customers": int(events["customer_id"].nunique(dropna=True)),
                "original_sessions": int(events["session_id"].nunique(dropna=True)),
                "event_types": int(events["event_type"].nunique(dropna=True)),
                "distinct_segment_names": int(events["segment_name"].nunique(dropna=True)),
                "distinct_screen_names": int(events["screen_name"].nunique(dropna=True)),
                "exact_event_tuples": int(events.drop_duplicates(["event_type", "segment_name", "screen_name"]).shape[0]),
            }
            print(safe_json(overview))
            print("\\nColumn names:")
            print(events.drop(columns=["source_row_number"]).columns.tolist())
            print("\\nInferred dtypes when pandas reads the CSV normally:")
            print(markdown_table(inferred, max_rows=50))
            print("\\nMasked sample:")
            print(markdown_table(masked_sample(events.drop(columns=["source_row_number"]), n=8), max_rows=8))
            """
        ),
        markdown_cell(
            """
            ## Missing-value exploration

            Missing values are counted exactly as loaded. Some missingness may be structural, but this notebook
            does not assign business meaning to it without source-owner confirmation.
            """
        ),
        code_cell(
            """
            miss = missing_summary(events.drop(columns=["source_row_number"]))
            by_event_availability = availability_by_event_type(events)
            both_missing = int(events["segment_name"].isna().mul(events["screen_name"].isna()).sum())
            segment_only = int(events["segment_name"].notna().mul(events["screen_name"].isna()).sum())
            screen_only = int(events["segment_name"].isna().mul(events["screen_name"].notna()).sum())
            print("Rows before missing-value analysis:", len(events))
            print("\\nMissing summary:")
            miss_display = miss.reset_index()
            miss_display.columns = ["column", "missing_count", "missing_percentage"]
            print(markdown_table(miss_display, max_rows=50))
            print("\\nAvailability by event_type:")
            print(markdown_table(by_event_availability, max_rows=50))
            print("\\nSegment/screen missingness combinations:")
            print(safe_json({
                "both_segment_name_and_screen_name_missing": both_missing,
                "segment_present_screen_missing": segment_only,
                "segment_missing_screen_present": screen_only,
            }))
            """
        ),
        markdown_cell(
            """
            ## Event and component distributions

            Values are counted exactly. No platform prefixes, suffixes, paths, URLs, spellings, or semantic
            variants are merged.
            """
        ),
        code_cell(
            """
            event_counts = value_counts_with_pct(events["event_type"])
            top_segments = value_counts_with_pct(events["segment_name"], top_n=25)
            top_screens = value_counts_with_pct(events["screen_name"], top_n=25)
            top_segment_by_event = component_by_event_type(events, "segment_name", top_n=10)
            top_screen_by_event = component_by_event_type(events, "screen_name", top_n=10)
            exact_counts = exact_tuple_counts(events)
            singleton_count = int(exact_counts["event_frequency"].eq(1).sum())
            singleton_pct = singleton_count / len(exact_counts) * 100
            long_tail = {
                "exact_token_vocabulary_size": len(exact_counts),
                "singleton_exact_token_combinations": singleton_count,
                "singleton_percentage_of_vocabulary": round(singleton_pct, 4),
                "top_10_combinations_event_share_pct": round(exact_counts.head(10)["event_frequency"].sum() / len(events) * 100, 4),
                "top_100_combinations_event_share_pct": round(exact_counts.head(100)["event_frequency"].sum() / len(events) * 100, 4),
            }
            print("Rows before distribution analysis:", len(events))
            print("\\nEvent-type counts:")
            print(markdown_table(event_counts, max_rows=25))
            print("\\nTop segment names overall:")
            print(markdown_table(top_segments, max_rows=25))
            print("\\nTop segment names by event_type:")
            print(markdown_table(top_segment_by_event, max_rows=30))
            print("\\nTop screen names overall:")
            print(markdown_table(top_screens, max_rows=25))
            print("\\nTop screen names by event_type:")
            print(markdown_table(top_screen_by_event, max_rows=30))
            print("\\nMost frequent exact tuples:")
            print(markdown_table(exact_counts.head(25), max_rows=25))
            print("\\nExact-token frequency distribution and long-tail summary:")
            print(safe_json(long_tail))
            print("\\nFrequency quantiles:")
            print(markdown_table(quantile_table(exact_counts["event_frequency"], [0, .25, .5, .75, .9, .95, .99, 1]), max_rows=20))
            """
        ),
        markdown_cell(
            """
            ## Timestamp analysis

            `timestamp` and `created_at` are parsed into derived columns. The epoch unit for `timestamp` is
            selected from evidence by comparing candidate units against `created_at`.
            """
        ),
        code_cell(
            """
            events_with_time, time_metadata = add_time_fields(events)
            print(f"Rows after timestamp parsing: {len(events_with_time):,}")
            print("\\nTimestamp unit evidence:")
            print(markdown_table(time_metadata["timestamp_unit_evidence"], max_rows=10))
            print("\\nPrimary ordering decision:")
            print(safe_json({
                "selected_timestamp_unit": time_metadata["selected_timestamp_unit"],
                "primary_event_time_column": time_metadata["primary_event_time_column"],
                "secondary_event_time_column": time_metadata["secondary_event_time_column"],
                "reason": time_metadata["primary_event_time_reason"],
                "deterministic_order": ["session_id", "primary_event_time", "secondary_event_time", "record_id", "source_row_number"],
            }))
            timestamp_summary = {
                "timestamp_datetime_min": str(events_with_time["timestamp_datetime"].min()),
                "timestamp_datetime_max": str(events_with_time["timestamp_datetime"].max()),
                "created_at_datetime_min": str(events_with_time["created_at_datetime"].min()),
                "created_at_datetime_max": str(events_with_time["created_at_datetime"].max()),
                "missing_timestamp_count": int(events_with_time["timestamp"].isna().sum()),
                "invalid_timestamp_count": int(events_with_time["timestamp_datetime"].isna().sum() - events_with_time["timestamp"].isna().sum()),
                "missing_created_at_count": int(events_with_time["created_at"].isna().sum()),
                "invalid_created_at_count": int(events_with_time["created_at_datetime"].isna().sum() - events_with_time["created_at"].isna().sum()),
            }
            print("\\nTimestamp summary:")
            print(safe_json(timestamp_summary))
            disagreement = events_with_time["timestamp_disagreement_seconds"].dropna()
            large_threshold = disagreement.quantile(.99) if len(disagreement) else np.nan
            large_disagreement = events_with_time.loc[
                events_with_time["timestamp_disagreement_seconds"].gt(large_threshold),
                ["record_id", "session_id", "timestamp", "created_at", "timestamp_datetime", "created_at_datetime", "timestamp_disagreement_seconds"],
            ].sort_values("timestamp_disagreement_seconds", ascending=False).head(10)
            print("\\nTimestamp disagreement quantiles:")
            print(markdown_table(quantile_table(disagreement, [0, .25, .5, .75, .9, .95, .99, 1]), max_rows=20))
            print("\\nEvents with unusually large timestamp disagreement:")
            print(markdown_table(large_disagreement, max_rows=10))
            """
        ),
        markdown_cell(
            """
            ## Duration analysis

            The source `duration` column is preserved. Numeric, log, missingness, zero, negative, positive, and
            outlier flags are derived separately. Duration is not included in categorical event tokens.
            """
        ),
        code_cell(
            """
            events_with_duration = add_duration_fields(events_with_time)
            print(f"Rows after duration derivation: {len(events_with_duration):,}")
            duration_numeric = events_with_duration["duration_numeric"]
            duration_stats = {
                "missing_duration_count": int(duration_numeric.isna().sum()),
                "zero_duration_count": int(duration_numeric.eq(0).sum()),
                "negative_duration_count": int(duration_numeric.lt(0).sum()),
                "positive_duration_count": int(duration_numeric.gt(0).sum()),
                "minimum": float(duration_numeric.min()) if duration_numeric.notna().any() else None,
                "maximum": float(duration_numeric.max()) if duration_numeric.notna().any() else None,
                "mean": float(duration_numeric.mean()) if duration_numeric.notna().any() else None,
                "median": float(duration_numeric.median()) if duration_numeric.notna().any() else None,
            }
            duration_by_event = events_with_duration.groupby("event_type", dropna=False).agg(
                rows=("record_id", "size"),
                duration_available=("duration_numeric", lambda s: int(s.notna().sum())),
                duration_missing=("duration_numeric", lambda s: int(s.isna().sum())),
                duration_zero=("duration_zero", "sum"),
                duration_positive=("duration_positive", "sum"),
            ).reset_index()
            extremes = events_with_duration.sort_values("duration_numeric", ascending=False).head(10)[
                ["record_id", "session_id", "event_type", "segment_name", "screen_name", "duration", "duration_numeric"]
            ]
            print("\\nDuration summary:")
            print(safe_json(duration_stats))
            print("\\nDuration percentiles:")
            print(markdown_table(quantile_table(duration_numeric, [0, .25, .5, .75, .9, .95, .99, 1]), max_rows=20))
            print("\\nDuration availability by event_type:")
            print(markdown_table(duration_by_event, max_rows=20))
            print("\\nExtreme-duration records:")
            print(markdown_table(extremes, max_rows=10))
            print("\\nlog1p(duration) percentiles for non-negative durations:")
            print(markdown_table(quantile_table(events_with_duration["log1p_duration"], [0, .25, .5, .75, .9, .95, .99, 1]), max_rows=20))
            """
        ),
        markdown_cell(
            """
            ## Part 1 summary

            This notebook stops at exploration. It does not perform semantic canonicalization or journey
            algorithm work.
            """
        ),
        code_cell(
            """
            after_fingerprint = file_fingerprint(INPUT_CSV)
            assert before_fingerprint == after_fingerprint, "Input CSV changed during exploration."
            print("Input file unchanged: passed")
            print(safe_json({
                "dataset_size": len(events),
                "columns": events.shape[1],
                "event_types_observed": events["event_type"].dropna().unique().tolist(),
                "exact_tuple_count": int(events.drop_duplicates(["event_type", "segment_name", "screen_name"]).shape[0]),
                "input_file_validation": "unchanged",
            }))
            """
        ),
    ]
    return notebook(cells)


def session_notebook() -> dict:
    """Build Part 2 notebook."""
    cells = [
        markdown_cell(
            """
            # Part 2 - Original-Session Exploration

            This notebook treats the existing `session_id` as the original logging-session identifier. Any
            derived clean-session boundary is stored separately and never overwrites `session_id`.
            """
        ),
        markdown_cell("## Configuration and imports"),
        code_cell(COMMON_IMPORTS),
        markdown_cell("## Load, validate, parse time, and deterministically sort events"),
        code_cell(
            """
            before_fingerprint = file_fingerprint(INPUT_CSV)
            events = read_events(INPUT_CSV)
            print(f"Rows after load: {len(events):,}")
            validate_required_columns(events)
            events, time_metadata = add_time_fields(events)
            events = add_duration_fields(events)
            print(f"Rows after derived time/duration fields: {len(events):,}")
            sorted_events = sort_events(events)
            sorted_events = construct_clean_sessions(sorted_events, SELECTED_INACTIVITY_THRESHOLD_MINUTES)
            print(f"Rows after deterministic sorting and clean-session derivation: {len(sorted_events):,}")
            print(safe_json({
                "selected_inactivity_threshold_minutes": SELECTED_INACTIVITY_THRESHOLD_MINUTES,
                "thresholds_compared_minutes": INACTIVITY_THRESHOLDS_MINUTES,
                "primary_event_time_column": time_metadata["primary_event_time_column"],
            }))
            """
        ),
        markdown_cell(
            """
            ## Original logging-session statistics

            Session spans and inter-event gaps are calculated after deterministic ordering by `session_id`,
            primary event time, secondary event time, `record_id`, and source row number.
            """
        ),
        code_cell(
            """
            sessions = original_session_summary(sorted_events)
            print(f"Original sessions: {len(sessions):,}")
            print("\\nEvent-count distribution:")
            print(markdown_table(quantile_table(sessions["event_count"], [0, .25, .5, .75, .9, .95, .99, 1]), max_rows=20))
            print("\\nSession-span distribution in seconds:")
            print(markdown_table(quantile_table(sessions["session_span_seconds"], [0, .25, .5, .75, .9, .95, .99, 1]), max_rows=20))
            print("\\nInter-event gap distribution in seconds:")
            print(markdown_table(quantile_table(sorted_events["time_gap_from_previous_seconds"], [0, .25, .5, .75, .9, .95, .99, 1]), max_rows=20))
            print("\\nSessions with exceptionally large event counts:")
            print(markdown_table(sessions.sort_values("event_count", ascending=False).head(10), max_rows=10))
            print("\\nSessions with exceptionally long spans:")
            print(markdown_table(sessions.sort_values("session_span_seconds", ascending=False).head(10), max_rows=10))
            """
        ),
        markdown_cell(
            """
            ## Identifier and timestamp integrity checks

            Known-to-known device/customer changes are examined. Changes between a known value and a missing
            value are not treated as split conditions.
            """
        ),
        code_cell(
            """
            issues = original_order_timestamp_issues(events)
            multi_device = sessions.loc[sessions["non_null_device_count"].gt(1)].sort_values("non_null_device_count", ascending=False)
            multi_customer = sessions.loc[sessions["non_null_customer_count"].gt(1)].sort_values("non_null_customer_count", ascending=False)
            gap_counts = {
                f"gaps_exceeding_{threshold}_minutes": int(sorted_events["time_gap_from_previous_seconds"].gt(threshold * 60).sum())
                for threshold in [15, 30, 60]
            }
            print("Timestamp issues before sorting:")
            printable_issues = {k: v for k, v in issues.items() if not isinstance(v, pd.DataFrame)}
            print(safe_json(printable_issues))
            print("\\nSample decreasing rows before sorting:")
            print(markdown_table(issues["sample_decreasing_rows"], max_rows=10))
            print("\\nSessions containing more than one non-null device:")
            print(markdown_table(multi_device.head(10), max_rows=10))
            print("\\nSessions containing more than one non-null customer:")
            print(markdown_table(multi_customer.head(10), max_rows=10))
            print("\\nInter-event gap counts:")
            print(safe_json(gap_counts))
            """
        ),
        markdown_cell(
            """
            ## Inactivity-threshold comparison and clean-session construction

            The 15-, 30-, and 60-minute thresholds are compared as modeling thresholds. The selected
            30-minute threshold is configurable and should not be read as a confirmed business rule.
            """
        ),
        code_cell(
            """
            threshold_table = threshold_comparison(sorted_events, INACTIVITY_THRESHOLDS_MINUTES)
            clean_session_count = sorted_events["clean_session_id"].nunique(dropna=True)
            split_reasons = sorted_events.loc[sorted_events["session_split_flag"], "session_split_reason"].value_counts().reset_index()
            split_reasons.columns = ["session_split_reason", "count"]
            print("\\nThreshold comparison:")
            print(markdown_table(threshold_table, max_rows=10))
            print("\\nClean-session summary:")
            print(safe_json({
                "original_sessions": int(sessions.shape[0]),
                "clean_sessions": int(clean_session_count),
                "selected_threshold_minutes": SELECTED_INACTIVITY_THRESHOLD_MINUTES,
                "split_rows": int(sorted_events["session_split_flag"].sum()),
            }))
            print("\\nSplit reasons:")
            print(markdown_table(split_reasons, max_rows=20))
            print("\\nClean-session event-count distribution:")
            clean_session_events = sorted_events.groupby("clean_session_id", dropna=False).size()
            print(markdown_table(quantile_table(clean_session_events, [0, .25, .5, .75, .9, .95, .99, 1]), max_rows=20))
            """
        ),
        markdown_cell("## Part 2 summary"),
        code_cell(
            """
            after_fingerprint = file_fingerprint(INPUT_CSV)
            assert before_fingerprint == after_fingerprint, "Input CSV changed during session exploration."
            print("Input file unchanged: passed")
            print(safe_json({
                "original_sessions": int(sessions.shape[0]),
                "clean_sessions": int(sorted_events["clean_session_id"].nunique(dropna=True)),
                "selected_inactivity_threshold_minutes": SELECTED_INACTIVITY_THRESHOLD_MINUTES,
                "gap_threshold_counts": gap_counts,
                "timestamp_validation_status": "completed",
            }))
            """
        ),
    ]
    return notebook(cells)


def tokenization_notebook() -> dict:
    """Build Parts 3, 4, and 6 notebook."""
    cells = [
        markdown_cell(
            """
            # Part 3 and 4 - Exact Tokenization and Tokenized Session Sequences

            ## Exact Tokenization and the Decision Not to Semantically Canonicalize

            The source owner defines every distinct `segment_name` and `screen_name` string as a genuinely
            different component or segment. Semantic canonicalization would therefore destroy source-defined
            information. Values such as `Home`, `HomeVC`, `android/Home`, suffixes, controller names, paths,
            URLs, punctuation, and casing are preserved exactly.

            Each event token is the exact tuple `(event_type, segment_name, screen_name)`. Duration and
            identifiers are not part of the categorical token.
            """
        ),
        markdown_cell("## Configuration and imports"),
        code_cell(COMMON_IMPORTS),
        markdown_cell(
            """
            ## Load, validate, parse, sort, and derive clean sessions

            Clean sessions are derived separately as `clean_session_id`. The original `session_id` remains
            unchanged and traceable.
            """
        ),
        code_cell(
            """
            before_fingerprint = file_fingerprint(INPUT_CSV)
            source_events = read_events(INPUT_CSV)
            print(f"Rows after source load: {len(source_events):,}")
            validate_required_columns(source_events)
            events, time_metadata = add_time_fields(source_events)
            print(f"Rows after timestamp parsing: {len(events):,}")
            events = add_duration_fields(events)
            print(f"Rows after duration derivation: {len(events):,}")
            ordered_events = sort_events(events)
            print(f"Rows after deterministic ordering: {len(ordered_events):,}")
            ordered_events = construct_clean_sessions(ordered_events, SELECTED_INACTIVITY_THRESHOLD_MINUTES)
            print(f"Rows after clean-session derivation: {len(ordered_events):,}")
            print(safe_json({
                "selected_timestamp_unit": time_metadata["selected_timestamp_unit"],
                "primary_event_time_column": time_metadata["primary_event_time_column"],
                "selected_inactivity_threshold_minutes": SELECTED_INACTIVITY_THRESHOLD_MINUTES,
            }))
            """
        ),
        markdown_cell(
            """
            ## Token serialization

            Tokens are serialized as JSON arrays to avoid unsafe delimiter concatenation. For derived token
            serialization only, missing fields are represented by `<MISSING>`. Original source columns remain
            missing and are not filled.
            """
        ),
        code_cell(
            """
            tokenized_base = add_exact_tokens(ordered_events, MISSING_SENTINEL)
            print(f"Rows after exact-token serialization: {len(tokenized_base):,}")
            print("Example serialized tokens:")
            print(markdown_table(tokenized_base[["event_type", "segment_name", "screen_name", "exact_token"]].head(10), max_rows=10))
            """
        ),
        markdown_cell(
            """
            ## Deterministic token dictionary

            Token IDs are assigned by sorting the exact JSON token strings and numbering them from 1. Python's
            runtime hash is not used.
            """
        ),
        code_cell(
            """
            token_dictionary = build_token_dictionary(tokenized_base)
            tokenized_events = add_token_ids(tokenized_base, token_dictionary)
            print(f"Rows after token ID assignment: {len(tokenized_events):,}")
            print(f"Token vocabulary size: {len(token_dictionary):,}")
            print("\\nToken dictionary preview:")
            print(markdown_table(token_dictionary.head(20), max_rows=20))
            print("\\nMost frequent tokens:")
            print(markdown_table(token_dictionary.sort_values("total_event_frequency", ascending=False).head(20), max_rows=20))
            """
        ),
        markdown_cell(
            """
            ## Tokenized event table

            The tokenized event table retains every source column and adds derived context fields, including
            previous/next token IDs, exact-repeat flags, previous-view context, gap fields, duration-derived
            fields, and missingness flags. Previous-view context does not alter the exact source token.
            """
        ),
        code_cell(
            """
            tokenized_preview_columns = [
                "record_id", "session_id", "clean_session_id", "event_index_in_original_session",
                "event_index_in_clean_session", "event_type", "segment_name", "screen_name",
                "token_id", "exact_token", "previous_token_id", "next_token_id",
                "time_gap_from_previous_seconds", "time_gap_to_next_seconds",
                "is_consecutive_exact_repeat", "previous_view_token_id", "seconds_since_previous_view",
                "duration_numeric", "log1p_duration", "duration_missing", "duration_zero",
                "duration_outlier", "segment_name_missing", "screen_name_missing",
            ]
            print(markdown_table(tokenized_events[tokenized_preview_columns].head(20), max_rows=20))
            """
        ),
        markdown_cell(
            """
            ## Tokenized session sequences

            Ordered raw token sequences and run-length compressed sequences are created for both original
            sessions and derived clean sessions.
            """
        ),
        code_cell(
            """
            original_sequences = build_session_sequences(tokenized_events, "session_id")
            clean_sequences = build_session_sequences(tokenized_events, "clean_session_id")
            print(f"Original-session sequences: {len(original_sequences):,}")
            print(f"Clean-session sequences: {len(clean_sequences):,}")
            print("\\nOriginal sequence metrics:")
            print(markdown_table(original_sequences[[
                "session_id", "raw_event_count", "compressed_event_count", "unique_exact_token_count",
                "view_count", "action_count", "missing_segment_count", "missing_screen_count",
                "session_duration_seconds", "maximum_inter_event_gap_seconds",
                "exact_repeat_count", "compression_ratio_compressed_to_raw",
            ]].head(20), max_rows=20))
            print("\\nCompression statistics:")
            print(markdown_table(quantile_table(original_sequences["compression_ratio_compressed_to_raw"], [0, .25, .5, .75, .9, .95, .99, 1]), max_rows=20))
            """
        ),
        markdown_cell(
            """
            ## Representative readable sequences

            Representative examples are displayed with token IDs and readable exact JSON tokens. Long previews
            are truncated only for display.
            """
        ),
        code_cell(
            """
            original_examples = representative_sequences(original_sequences, tokenized_events, "session_id", token_dictionary)
            clean_examples = representative_sequences(clean_sequences, tokenized_events, "clean_session_id", token_dictionary)
            print("Original-session examples:")
            print(markdown_table(original_examples, max_rows=10))
            print("\\nClean-session examples:")
            print(markdown_table(clean_examples, max_rows=10))
            """
        ),
        markdown_cell(
            """
            ## Validation

            These assertions confirm exact-value preservation, one-to-one token mapping, reversible
            serialization, deterministic ordering, count reconciliation, sequence validity, and repeated-run
            token dictionary stability.
            """
        ),
        code_cell(
            """
            validation_results = validate_tokenization(
                source_events,
                tokenized_events,
                token_dictionary,
                original_sequences,
                clean_sequences,
            )
            after_fingerprint = file_fingerprint(INPUT_CSV)
            assert before_fingerprint == after_fingerprint, "Input CSV was modified."
            validation_results["input_file_status"] = "unchanged"
            print("Validation results:")
            print(safe_json(validation_results))
            """
        ),
        markdown_cell(
            """
            ## Save tokenized outputs

            After validation passes, save the deterministic token dictionary and tokenized session sequences
            into repo-relative output files. Sequence files are JSON Lines so raw token arrays and compressed
            repeat objects are preserved without delimiter ambiguity.
            """
        ),
        code_cell(
            """
            OUTPUT_DIR = REPO_ROOT / "outputs"
            output_manifest = save_token_outputs(
                token_dictionary=token_dictionary,
                original_sequences=original_sequences,
                clean_sequences=clean_sequences,
                output_dir=OUTPUT_DIR,
            )
            print("Saved tokenized outputs:")
            print(markdown_table(pd.DataFrame(output_manifest["files"]), max_rows=10))
            """
        ),
        markdown_cell(
            """
            ## Final summary

            This notebook stops after exact tokenization, duplicate-aware sequence construction, and validation.
            It does not implement journey-candidate discovery, PrefixSpan, embeddings, clustering, journey
            labeling, or cluster evaluation.
            """
        ),
        code_cell(
            """
            duplicate_stats = {
                "duplicate_record_id_rows": int(source_events["record_id"].duplicated(keep=False).sum()),
                "duplicate_record_id_excess": int(source_events["record_id"].duplicated().sum()),
                "fully_duplicated_rows": int(source_events.drop(columns=["source_row_number"]).duplicated().sum()),
            }
            compression_stats = {
                "mean_compression_ratio_compressed_to_raw": float(original_sequences["compression_ratio_compressed_to_raw"].mean()),
                "median_compression_ratio_compressed_to_raw": float(original_sequences["compression_ratio_compressed_to_raw"].median()),
                "sessions_with_repeated_exact_tokens": int(original_sequences["exact_repeat_count"].gt(0).sum()),
            }
            final_summary = {
                "dataset_size": int(len(source_events)),
                "token_vocabulary_size": int(len(token_dictionary)),
                "original_sessions": int(original_sequences.shape[0]),
                "clean_sessions": int(clean_sequences.shape[0]),
                "selected_inactivity_threshold_minutes": SELECTED_INACTIVITY_THRESHOLD_MINUTES,
                "duplicate_event_statistics": duplicate_stats,
                "compression_statistics": compression_stats,
                "validation_status": validation_results["validation_status"],
                "saved_output_files": [file_info["name"] for file_info in output_manifest["files"]],
                "important_unresolved_questions": [
                    "Whether the 30-minute inactivity threshold should be changed by business policy.",
                    "Whether timestamp and created_at disagreement has source-system meaning.",
                    "Whether duration units are formally documented by the source owner.",
                ],
            }
            print(safe_json(final_summary))
            """
        ),
    ]
    return notebook(cells)


def write_notebook(path: Path, nb: dict) -> None:
    """Write a notebook JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_for_docs(repo_root: Path) -> dict:
    """Compute the shared metrics used in Markdown documentation."""
    input_path = input_csv_path(repo_root)
    fingerprint_before = file_fingerprint(input_path)
    source = read_events(input_path)
    validate_required_columns(source)
    inferred = inferred_dtypes(input_path).rename("inferred_dtype").reset_index()
    inferred.columns = ["column", "inferred_dtype"]
    events, time_metadata = add_time_fields(source)
    events = add_duration_fields(events)
    ordered = sort_events(events)
    clean = construct_clean_sessions(ordered, SELECTED_INACTIVITY_THRESHOLD_MINUTES)
    tokenized_base = add_exact_tokens(clean)
    token_dictionary = build_token_dictionary(tokenized_base)
    tokenized_events = add_token_ids(tokenized_base, token_dictionary)
    original_sequences = build_session_sequences(tokenized_events, "session_id")
    clean_sequences = build_session_sequences(tokenized_events, "clean_session_id")
    validation = validate_tokenization(source, tokenized_events, token_dictionary, original_sequences, clean_sequences)
    output_manifest = save_token_outputs(
        token_dictionary=token_dictionary,
        original_sequences=original_sequences,
        clean_sequences=clean_sequences,
        output_dir=repo_root / "outputs",
    )
    fingerprint_after = file_fingerprint(input_path)
    if fingerprint_before != fingerprint_after:
        raise RuntimeError("Input CSV changed while building docs.")

    exact_counts = exact_tuple_counts(source)
    sessions = original_session_summary(clean)
    threshold_table = threshold_comparison(clean, INACTIVITY_THRESHOLDS_MINUTES)
    duration_numeric = clean["duration_numeric"]
    timestamp_disagreement = clean["timestamp_disagreement_seconds"].dropna()
    issues = original_order_timestamp_issues(events)
    split_reasons = clean.loc[clean["session_split_flag"], "session_split_reason"].value_counts().reset_index()
    split_reasons.columns = ["session_split_reason", "count"]

    return {
        "source": source,
        "inferred": inferred,
        "events": events,
        "clean": clean,
        "token_dictionary": token_dictionary,
        "tokenized_events": tokenized_events,
        "original_sequences": original_sequences,
        "clean_sequences": clean_sequences,
        "validation": validation,
        "output_manifest": output_manifest,
        "time_metadata": time_metadata,
        "exact_counts": exact_counts,
        "sessions": sessions,
        "threshold_table": threshold_table,
        "duration_numeric": duration_numeric,
        "timestamp_disagreement": timestamp_disagreement,
        "issues": issues,
        "split_reasons": split_reasons,
        "fingerprint": fingerprint_after,
    }


def paragraph_table(title: str, df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a titled Markdown table block."""
    return f"\n\n**{title}**\n\n{markdown_table(df, max_rows=max_rows)}"


def build_combined_doc(summary: dict) -> str:
    """Build the required combined documentation file."""
    source = summary["source"]
    clean = summary["clean"]
    token_dictionary = summary["token_dictionary"]
    original_sequences = summary["original_sequences"]
    clean_sequences = summary["clean_sequences"]
    exact_counts = summary["exact_counts"]
    sessions = summary["sessions"]
    time_metadata = summary["time_metadata"]
    duration_numeric = summary["duration_numeric"]
    timestamp_disagreement = summary["timestamp_disagreement"]
    validation = summary["validation"]

    duplicate_stats = pd.DataFrame(
        [
            {"metric": "duplicate_record_id_rows", "value": int(source["record_id"].duplicated(keep=False).sum())},
            {"metric": "duplicate_record_id_excess", "value": int(source["record_id"].duplicated().sum())},
            {"metric": "fully_duplicated_rows", "value": int(source.drop(columns=["source_row_number"]).duplicated().sum())},
        ]
    )
    overview = pd.DataFrame(
        [
            {"metric": "rows", "value": len(source)},
            {"metric": "columns", "value": source.shape[1] - 1},
            {"metric": "devices", "value": int(source["device_id"].nunique(dropna=True))},
            {"metric": "customers", "value": int(source["customer_id"].nunique(dropna=True))},
            {"metric": "original_sessions", "value": int(source["session_id"].nunique(dropna=True))},
            {"metric": "clean_sessions", "value": int(clean["clean_session_id"].nunique(dropna=True))},
            {"metric": "event_types", "value": int(source["event_type"].nunique(dropna=True))},
            {"metric": "distinct_segment_names", "value": int(source["segment_name"].nunique(dropna=True))},
            {"metric": "distinct_screen_names", "value": int(source["screen_name"].nunique(dropna=True))},
            {"metric": "exact_event_tuples", "value": int(source.drop_duplicates(["event_type", "segment_name", "screen_name"]).shape[0])},
            {"metric": "token_vocabulary_size", "value": len(token_dictionary)},
        ]
    )
    duration_stats = pd.DataFrame(
        [
            {"metric": "missing_duration_count", "value": int(duration_numeric.isna().sum())},
            {"metric": "zero_duration_count", "value": int(duration_numeric.eq(0).sum())},
            {"metric": "negative_duration_count", "value": int(duration_numeric.lt(0).sum())},
            {"metric": "positive_duration_count", "value": int(duration_numeric.gt(0).sum())},
            {"metric": "minimum", "value": duration_numeric.min()},
            {"metric": "maximum", "value": duration_numeric.max()},
            {"metric": "mean", "value": duration_numeric.mean()},
            {"metric": "median", "value": duration_numeric.median()},
        ]
    )
    timestamp_stats = pd.DataFrame(
        [
            {"metric": "selected_timestamp_unit", "value": time_metadata["selected_timestamp_unit"]},
            {"metric": "primary_event_time_column", "value": time_metadata["primary_event_time_column"]},
            {"metric": "secondary_event_time_column", "value": time_metadata["secondary_event_time_column"]},
            {"metric": "timestamp_min", "value": clean["timestamp_datetime"].min()},
            {"metric": "timestamp_max", "value": clean["timestamp_datetime"].max()},
            {"metric": "created_at_min", "value": clean["created_at_datetime"].min()},
            {"metric": "created_at_max", "value": clean["created_at_datetime"].max()},
            {"metric": "timestamp_invalid_count", "value": int(clean["timestamp_datetime"].isna().sum() - clean["timestamp"].isna().sum())},
            {"metric": "created_at_invalid_count", "value": int(clean["created_at_datetime"].isna().sum() - clean["created_at"].isna().sum())},
            {"metric": "timestamp_disagreement_median_seconds", "value": timestamp_disagreement.median() if len(timestamp_disagreement) else None},
            {"metric": "timestamp_disagreement_99th_seconds", "value": timestamp_disagreement.quantile(.99) if len(timestamp_disagreement) else None},
        ]
    )
    compression_stats = pd.DataFrame(
        [
            {"metric": "mean_compression_ratio_compressed_to_raw", "value": original_sequences["compression_ratio_compressed_to_raw"].mean()},
            {"metric": "median_compression_ratio_compressed_to_raw", "value": original_sequences["compression_ratio_compressed_to_raw"].median()},
            {"metric": "sessions_with_repeated_exact_tokens", "value": int(original_sequences["exact_repeat_count"].gt(0).sum())},
        ]
    )
    output_files = pd.DataFrame(summary["output_manifest"]["files"])

    doc = f"""# Data Exploration and Exact Tokenization

This document was generated from `data/final_clean_events.csv`. The input CSV was not modified.

## 1. Dataset overview

Confirmed source fact: the source owner confirmed that every distinct `segment_name` and `screen_name` is a different component or segment.

Observed data evidence is summarized below. Source values are preserved exactly; no case folding, suffix stripping, URL/path simplification, fuzzy matching, or semantic merging was performed.
{paragraph_table("Dataset overview", overview, 30)}
{paragraph_table("Masked sample", masked_sample(source.drop(columns=["source_row_number"]), n=8), 8)}

## 2. Data-type findings

The notebooks load source columns as string-backed values for preservation, then create derived numeric and datetime columns. The table below shows pandas' default inferred data types when the CSV is read normally for documentation.
{paragraph_table("Inferred data types", summary["inferred"], 50)}

## 3. Missing-value analysis

Missing values are preserved in source columns. Missingness may be structural, but the notebooks do not claim its meaning without source-owner confirmation.
{paragraph_table("Missing summary", missing_summary(source.drop(columns=["source_row_number"])).reset_index().rename(columns={"index": "column"}), 50)}
{paragraph_table("Availability by event_type", availability_by_event_type(source), 20)}

Rows where both `segment_name` and `screen_name` are missing: {int(source["segment_name"].isna().mul(source["screen_name"].isna()).sum())}.
Rows where `segment_name` is present and `screen_name` is missing: {int(source["segment_name"].notna().mul(source["screen_name"].isna()).sum())}.
Rows where `segment_name` is missing and `screen_name` is present: {int(source["segment_name"].isna().mul(source["screen_name"].notna()).sum())}.

## 4. Timestamp findings

The `timestamp` unit is determined from evidence rather than assumption. Candidate epoch units are compared to parsed `created_at`, and `{time_metadata["selected_timestamp_unit"]}` is selected.
{paragraph_table("Timestamp unit evidence", time_metadata["timestamp_unit_evidence"], 10)}
{paragraph_table("Timestamp summary", timestamp_stats, 30)}

Observed ordering choice: `{time_metadata["primary_event_time_column"]}` is the primary event-ordering field because {time_metadata["primary_event_time_reason"]}. The deterministic ordering is `session_id`, primary event time, secondary event time, `record_id`, then source row number.

## 5. Duration findings

Duration is not included in categorical tokens. Derived fields include `duration_numeric`, `log1p_duration`, `duration_missing`, `duration_zero`, and `duration_outlier`. Extreme values are not capped, replaced, or deleted.
{paragraph_table("Duration summary", duration_stats, 30)}
{paragraph_table("Duration percentiles", quantile_table(duration_numeric, [0, .25, .5, .75, .9, .95, .99, 1]), 20)}

## 6. Original-session findings

`session_id` is treated as the original logging-session identifier and is never overwritten.
{paragraph_table("Session event-count quantiles", quantile_table(sessions["event_count"], [0, .25, .5, .75, .9, .95, .99, 1]), 20)}
{paragraph_table("Session-span quantiles in seconds", quantile_table(sessions["session_span_seconds"], [0, .25, .5, .75, .9, .95, .99, 1]), 20)}
{paragraph_table("Original-order timestamp issues", pd.DataFrame([{k: v for k, v in summary["issues"].items() if not isinstance(v, pd.DataFrame)}]), 5)}

Sessions with more than one non-null device: {int(sessions["non_null_device_count"].gt(1).sum())}.
Sessions with more than one non-null customer: {int(sessions["non_null_customer_count"].gt(1).sum())}.

## 7. Inactivity-threshold comparison

Configurable thresholds of 15, 30, and 60 minutes are compared. The selected 30-minute threshold is a modeling assumption for this notebook, not a confirmed business fact.
{paragraph_table("Threshold comparison", summary["threshold_table"], 10)}
{paragraph_table("Clean-session split reasons", summary["split_reasons"], 20)}

## 8. Duplicate-event findings
{paragraph_table("Duplicate statistics", duplicate_stats, 10)}

## 9. Exact token definition

Each event is represented as the exact tuple `(event_type, segment_name, screen_name)`. For example, `["View","HomeVC",null]`, `["View","android/Home",null]`, `["Action","btn_back","HomeVC"]`, and `["Action","btn_back","android/Home"]` remain four different tokens if observed.

## 10. Why semantic canonicalization is intentionally not performed

Confirmed source fact: distinct segment and screen strings are source-defined distinct components. Therefore, semantic canonicalization would remove information that the source owner says is meaningful. This work performs schema validation, derived timestamp parsing, deterministic exact token ID assignment, and derived missing-value representation only.

## 11. Missing-value handling inside tokens

Original missing source values remain missing. In derived serialized token strings only, missing token fields are represented explicitly as `{MISSING_SENTINEL}` inside a JSON array. The sentinel is validated not to collide with observed source values.

## 12. Deterministic token-ID strategy

The exact JSON token strings are sorted deterministically, then assigned integer IDs from 1 to {len(token_dictionary)}. Python runtime hashes are not used.
{paragraph_table("Token dictionary preview", token_dictionary.head(20), 20)}

## 13. Raw versus compressed session sequences

Raw sequences retain every event token. Compressed sequences use run-length encoding for consecutive exact-token repeats and never exceed the raw length.
{paragraph_table("Original-sequence compression statistics", compression_stats, 10)}
{paragraph_table("Representative original sequences", representative_sequences(original_sequences, tokenized_events=summary["tokenized_events"], group_column="session_id", token_dictionary=token_dictionary), 10)}

Saved output files are written under `outputs/`. The token dictionary is available as CSV and JSON Lines; session sequences are JSON Lines to preserve nested arrays and compressed repeat objects.
{paragraph_table("Saved tokenized output files", output_files, 10)}

## 14. Validation results

Validation passed for row preservation, record traceability, one-to-one token mapping, reversible serialization, unchanged source columns, deterministic ordering, session count reconciliation, sequence lengths, compressed sequence lengths, repeated dictionary determinism, and unchanged input file fingerprint.
{paragraph_table("Validation summary", pd.DataFrame([validation]), 10)}

## 15. Limitations and unresolved questions

Unresolved questions:

- Whether the 30-minute inactivity threshold should be changed by business policy.
- Whether timestamp and `created_at` disagreement has source-system meaning.
- Whether duration units and extreme duration semantics are formally documented by the source owner.
- Whether missing `segment_name` or `screen_name` values are structural for specific event types.

## 16. Recommended next step toward journey-candidate discovery

The next step should be journey-candidate discovery using the validated exact-token session sequences. That next phase can compare candidate mining approaches, but this deliverable intentionally stops before PrefixSpan, embeddings, clustering, journey labeling, and cluster evaluation.

## Classification of statements

- Confirmed source facts: distinct `segment_name` and `screen_name` values are distinct components or segments.
- Observed data evidence: all counts, distributions, timestamp comparisons, duration summaries, duplicate findings, and token frequencies above.
- Modeling assumptions: derived clean-session split conditions and the selected 30-minute inactivity threshold.
- Configurable thresholds: 15, 30, and 60 minute inactivity thresholds, with 30 minutes selected for derived clean sessions.
- Derived fields: parsed datetime columns, numeric duration fields, missingness flags, clean-session fields, exact token strings, token IDs, neighbor token context, and sequence metrics.
"""
    return doc


def build_section_docs(summary: dict) -> dict[str, str]:
    """Build smaller docs for the major sections."""
    combined = build_combined_doc(summary)
    data_exploration_text = combined.split("## 6. Original-session findings")[0].strip() + "\n"
    return {
        "1_Data_Exploration.md": data_exploration_text,
        "2_Session_Based_EDA.md": f"""# Session-Based EDA

This document summarizes original-session and derived clean-session exploration. `session_id` is preserved, and `clean_session_id` is derived separately.
{paragraph_table("Session event-count quantiles", quantile_table(summary["sessions"]["event_count"], [0, .25, .5, .75, .9, .95, .99, 1]), 20)}
{paragraph_table("Session-span quantiles in seconds", quantile_table(summary["sessions"]["session_span_seconds"], [0, .25, .5, .75, .9, .95, .99, 1]), 20)}
{paragraph_table("Threshold comparison", summary["threshold_table"], 10)}
{paragraph_table("Clean-session split reasons", summary["split_reasons"], 20)}
""",
        "3_Exact_Tokenization_and_Sequences.md": f"""# Exact Tokenization and Session Sequences

Semantic canonicalization is intentionally not performed. Each exact `(event_type, segment_name, screen_name)` tuple is serialized as a JSON array and mapped to one deterministic token ID.
{paragraph_table("Token dictionary preview", summary["token_dictionary"].head(20), 20)}
{paragraph_table("Saved tokenized output files", pd.DataFrame(summary["output_manifest"]["files"]), 10)}
{paragraph_table("Validation summary", pd.DataFrame([summary["validation"]]), 10)}
{paragraph_table("Representative original sequences", representative_sequences(summary["original_sequences"], summary["tokenized_events"], "session_id", summary["token_dictionary"]), 10)}
""",
    }


def main() -> None:
    """Generate notebooks and docs."""
    repo_root = find_repo_root()
    notebooks_dir = repo_root / "notebooks"
    docs_dir = repo_root / "docs"
    scripts_dir = repo_root / "scripts"
    notebooks_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)

    write_notebook(notebooks_dir / "1_data_exploration.ipynb", data_exploration_notebook())
    write_notebook(notebooks_dir / "2_session_based_eda.ipynb", session_notebook())
    write_notebook(notebooks_dir / "3_Tokenization.ipynb", tokenization_notebook())

    summary = summarize_for_docs(repo_root)
    (docs_dir / "Data_Exploration_and_Tokenization.md").write_text(build_combined_doc(summary), encoding="utf-8")
    for filename, text in build_section_docs(summary).items():
        (docs_dir / filename).write_text(text, encoding="utf-8")

    print("Generated notebooks and documentation.")


if __name__ == "__main__":
    main()
