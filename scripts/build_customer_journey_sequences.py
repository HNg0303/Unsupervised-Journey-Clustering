#!/usr/bin/env python3
"""Build chronological customer/session journey sequences from named score exports.

The named inspection CSVs contain one row per *journey*, not one row per event.  This
script therefore orders journeys by ``start_ts`` inside ``(customer_id, platform,
session_id)`` and exports both compact sequence tables and a bounded detail table for
visualisation.

The source files in the current bundle are multi-GB CSVs.  DuckDB is intentionally used
for CSV scanning, sorting, grouping and spilling to disk.  The detail export is limited
to the top sessions by journey count by default; pass ``--top-sessions 0`` to export all
sessions (which may be large).

Example:
    python scripts/build_customer_journey_sequences.py \
      --android output/scores/pca48_ngrams12_500/android/model_version=latest/platform=android/inspection/android_all_named.csv \
      --ios output/scores/pca48_ngrams12_500/ios/model_version=latest/platform=ios/inspection/ios_all_named.csv \
      --output-dir output/scores/pca48_ngrams12_500/customer_journey_sequences \
      --top-sessions 200

Outputs:
    summary_total.csv              Whole-bundle headline statistics.
    summary_platform.csv           Headline statistics by platform.
    summary_level_counts.csv       Counts by business family/submodule/cluster.
    customer_summary.csv            One row per customer.
    session_sequences.csv           One row per customer/platform/session.
    transition_counts.csv          Previous -> next journey transitions by level.
    journey_path_detail.csv        Ordered rows for the selected top sessions.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


REQUIRED_COLUMNS = {
    "journey_id",
    "session_id",
    "customer_id",
    "start_ts",
    "end_ts",
    "sequence",
}

LEVELS = (
    ("business_family", "business_family"),
    ("business_submodule", "business_submodule"),
    ("cluster", "cluster_name"),
    ("business_family_code", "business_family_code"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android", type=Path, required=True, help="named Android journey CSV")
    parser.add_argument("--ios", type=Path, required=True, help="named iOS journey CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--top-sessions",
        type=int,
        default=200,
        help="number of highest-volume sessions to include in journey_path_detail.csv; 0 = all",
    )
    parser.add_argument(
        "--top-customers",
        type=int,
        default=0,
        help="optional customer cap before selecting top sessions; 0 = all customers",
    )
    parser.add_argument("--memory-limit", default="4GB", help="DuckDB memory limit; overflow spills to disk")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="DuckDB worker threads; 1 reduces peak memory for multi-GB CSV sorts",
    )
    parser.add_argument("--keep-duckdb", action="store_true", help="keep the temporary DuckDB file for inspection")
    return parser.parse_args()


def read_header(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    return {name.strip() for name in header}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def clean_expr(columns: set[str], column: str, fallback: str = "__unknown__") -> str:
    """Return a DuckDB expression that normalises empty/missing labels."""
    if column not in columns:
        return sql_string(fallback)
    quoted = sql_identifier(column)
    return f"coalesce(nullif(trim(cast({quoted} as varchar)), ''), {sql_string(fallback)})"


def source_view_sql(name: str, path: Path, platform: str) -> str:
    # all_varchar avoids inference differences between Android and iOS exports.
    return f"""
        CREATE OR REPLACE TEMP VIEW {name} AS
        SELECT *, {sql_string(platform)} AS source_platform
        FROM read_csv_auto(
            {sql_string(str(path.resolve()))},
            header=true,
            all_varchar=true,
            encoding='utf-8'
        )
    """


def select_source(name: str, columns: set[str]) -> str:
    def raw(column: str, fallback: str = "") -> str:
        if column in columns:
            return f"cast({sql_identifier(column)} as varchar)"
        return sql_string(fallback)

    # ``platform`` is deliberately taken from the input flag, not from a potentially
    # stale column copied from an earlier scoring step.
    fields = [
        "source_platform AS platform",
        f"{clean_expr(columns, 'journey_id', '')} AS journey_id",
        f"{clean_expr(columns, 'session_id', '')} AS session_id",
        f"{clean_expr(columns, 'customer_id', '__anonymous__')} AS customer_id",
        f"{raw('device_id')} AS device_id",
        f"{raw('start_ts')} AS start_ts",
        f"{raw('end_ts')} AS end_ts",
        f"try_cast(nullif(trim({raw('start_ts')}), '') AS TIMESTAMPTZ) AS start_ts_value",
        f"try_cast(nullif(trim({raw('end_ts')}), '') AS TIMESTAMPTZ) AS end_ts_value",
        f"{clean_expr(columns, 'sequence', '')} AS sequence",
        f"{clean_expr(columns, 'cluster', '')} AS cluster",
        f"{clean_expr(columns, 'cluster_id', '')} AS cluster_id",
        f"{clean_expr(columns, 'cluster_name', '')} AS cluster_name",
        f"{clean_expr(columns, 'cluster_name_en', '')} AS cluster_name_en",
        f"{clean_expr(columns, 'business_family', '__unknown__')} AS business_family",
        f"{clean_expr(columns, 'business_family_code', '__unknown__')} AS business_family_code",
        f"{clean_expr(columns, 'business_submodule', '__unknown__')} AS business_submodule",
        f"{clean_expr(columns, 'business_detail', '')} AS business_detail",
        f"{raw('n_events_raw')} AS n_events_raw",
        f"{raw('n_events_final')} AS n_events_final",
        f"{raw('severe_anomaly')} AS severe_anomaly",
        f"{raw('friction_flags')} AS friction_flags",
        f"{raw('next_action')} AS next_action",
    ]
    return "SELECT " + ",\n               ".join(fields) + f"\n        FROM {name}"


def copy_query(connection, query: str, output_path: Path) -> None:
    """Write a query as UTF-8 CSV, replacing only this named generated output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    escaped = str(output_path.resolve()).replace("'", "''")
    connection.execute(
        f"COPY ({query}) TO '{escaped}' "
        "(FORMAT CSV, HEADER, DELIMITER ',')"
    )


def csv_value(value: object) -> object:
    """Keep DuckDB timestamps readable when writing with Python's csv module."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_session_sequences_streaming(connection, output_path: Path) -> None:
    """Write one session sequence at a time to avoid a global string_agg allocation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    query = """
        SELECT customer_id, platform, session_id, session_key,
               journey_id, session_journey_index,
               start_ts_value, end_ts_value,
               cluster_name, cluster_id,
               business_family, business_submodule, business_family_code
        FROM ordered
        ORDER BY customer_id, platform, session_id, session_journey_index
    """
    cursor = connection.execute(query)
    columns = [description[0] for description in cursor.description]
    output_fields = [
        "customer_id",
        "platform",
        "session_id",
        "session_key",
        "journey_count",
        "first_start_ts",
        "last_end_ts",
        "journey_id_sequence",
        "cluster_sequence",
        "business_family_sequence",
        "business_submodule_sequence",
        "business_family_code_sequence",
    ]

    def flush(state: dict[str, object], writer: csv.DictWriter) -> None:
        if not state:
            return
        writer.writerow(
            {
                "customer_id": state["customer_id"],
                "platform": state["platform"],
                "session_id": state["session_id"],
                "session_key": state["session_key"],
                "journey_count": state["journey_count"],
                "first_start_ts": csv_value(state["first_start_ts"]),
                "last_end_ts": csv_value(state["last_end_ts"]),
                "journey_id_sequence": " -> ".join(state["journey_ids"]),
                "cluster_sequence": " -> ".join(state["clusters"]),
                "business_family_sequence": " -> ".join(state["families"]),
                "business_submodule_sequence": " -> ".join(state["submodules"]),
                "business_family_code_sequence": " -> ".join(state["family_codes"]),
            }
        )

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        state: dict[str, object] = {}
        row_count = 0
        while True:
            batch = cursor.fetchmany(10_000)
            if not batch:
                break
            for values in batch:
                row = dict(zip(columns, values))
                key = (row["customer_id"], row["platform"], row["session_id"])
                current_key = state.get("key")
                if current_key is not None and key != current_key:
                    flush(state, writer)
                    state = {}
                if not state:
                    state = {
                        "key": key,
                        "customer_id": row["customer_id"],
                        "platform": row["platform"],
                        "session_id": row["session_id"],
                        "session_key": row["session_key"],
                        "journey_count": 0,
                        "first_start_ts": row["start_ts_value"],
                        "last_end_ts": row["end_ts_value"],
                        "journey_ids": [],
                        "clusters": [],
                        "families": [],
                        "submodules": [],
                        "family_codes": [],
                    }
                state["journey_count"] += 1
                state["last_end_ts"] = row["end_ts_value"]
                state["journey_ids"].append(str(row["journey_id"]))
                cluster = row["cluster_name"]
                if cluster in (None, "", "__unknown__"):
                    cluster = row["cluster_id"]
                state["clusters"].append(str(cluster or "__unknown__"))
                state["families"].append(str(row["business_family"] or "__unknown__"))
                state["submodules"].append(str(row["business_submodule"] or "__unknown__"))
                state["family_codes"].append(str(row["business_family_code"] or "__unknown__"))
                row_count += 1
        flush(state, writer)
    print(f"  session_sequences.csv: streamed {row_count:,} journey rows")


def build(args: argparse.Namespace) -> None:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on the local environment
        raise SystemExit(
            "DuckDB is required for the multi-GB named CSVs. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    if args.top_sessions < 0 or args.top_customers < 0:
        raise SystemExit("--top-sessions and --top-customers must be >= 0")

    android_columns = read_header(args.android)
    ios_columns = read_header(args.ios)
    for path, columns in ((args.android, android_columns), (args.ios, ios_columns)):
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise SystemExit(f"{path} missing required columns: {', '.join(sorted(missing))}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / ".customer_journey_sequences.duckdb"
    if db_path.exists():
        db_path.unlink()
    temp_dir = output_dir / ".duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(db_path))
    try:
        # DuckDB settings are kept local to this generated database.  The source CSVs
        # remain untouched and sorts can spill into the output directory.
        connection.execute(f"SET memory_limit={sql_string(args.memory_limit)}")
        connection.execute(f"SET threads={max(1, int(args.threads))}")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET temp_directory={sql_string(str(temp_dir.resolve()))}")
        connection.execute(source_view_sql("android_source", args.android, "android"))
        connection.execute(source_view_sql("ios_source", args.ios, "ios"))

        print("Materializing ordered journey table (disk-backed)...")
        union_sql = f"""
            CREATE OR REPLACE TEMP VIEW normalised AS
            {select_source('android_source', android_columns)}
            UNION ALL
            {select_source('ios_source', ios_columns)}
        """
        connection.execute(union_sql)
        connection.execute(
            """
            CREATE OR REPLACE TABLE ordered AS
            WITH ordered_base AS (
                SELECT
                    *,
                    customer_id || '::' || platform || '::' || session_id AS session_key,
                    row_number() OVER (
                        PARTITION BY customer_id, platform, session_id
                        ORDER BY start_ts_value NULLS LAST, end_ts_value NULLS LAST, journey_id
                    ) AS session_journey_index
                FROM normalised
                WHERE nullif(trim(session_id), '') IS NOT NULL
            )
            SELECT
                *,
                lag(journey_id) OVER w AS previous_journey_id,
                lead(journey_id) OVER w AS next_journey_id,
                lag(business_family) OVER w AS previous_business_family,
                lead(business_family) OVER w AS next_business_family,
                lag(business_submodule) OVER w AS previous_business_submodule,
                lead(business_submodule) OVER w AS next_business_submodule,
                lag(cluster_name) OVER w AS previous_cluster,
                lead(cluster_name) OVER w AS next_cluster
            FROM ordered_base
            WINDOW w AS (
                PARTITION BY customer_id, platform, session_id
                ORDER BY session_journey_index
            )
            """
        )

        print("Writing whole-bundle summaries...")
        summary_total = """
            SELECT metric, value FROM (
                SELECT 'journeys' AS metric, cast(count(*) AS varchar) AS value FROM ordered
                UNION ALL SELECT 'customers_including_anonymous', cast(count(DISTINCT customer_id) AS varchar) FROM ordered
                UNION ALL SELECT 'customers_excluding_anonymous', cast(count(DISTINCT CASE WHEN customer_id <> '__anonymous__' THEN customer_id END) AS varchar) FROM ordered
                UNION ALL SELECT 'anonymous_journeys', cast(count(*) FILTER (WHERE customer_id = '__anonymous__') AS varchar) FROM ordered
                UNION ALL SELECT 'session_ids', cast(count(DISTINCT session_id) AS varchar) FROM ordered
                UNION ALL SELECT 'platform_session_keys', cast(count(DISTINCT session_key) AS varchar) FROM ordered
                UNION ALL SELECT 'platforms', cast(count(DISTINCT platform) AS varchar) FROM ordered
                UNION ALL SELECT 'first_start_ts', cast(min(start_ts_value) AS varchar) FROM ordered
                UNION ALL SELECT 'last_start_ts', cast(max(start_ts_value) AS varchar) FROM ordered
                UNION ALL SELECT 'journeys_with_valid_start_ts', cast(count(*) FILTER (WHERE start_ts_value IS NOT NULL) AS varchar) FROM ordered
                UNION ALL SELECT 'business_families', cast(count(DISTINCT business_family) AS varchar) FROM ordered
                UNION ALL SELECT 'business_submodules', cast(count(DISTINCT business_submodule) AS varchar) FROM ordered
                UNION ALL SELECT 'clusters', cast(count(DISTINCT cluster_id) AS varchar) FROM ordered
                UNION ALL SELECT 'sessions_with_multiple_journeys', cast(count(*) FILTER (WHERE journey_count > 1) AS varchar) FROM (
                    SELECT session_key, count(*) AS journey_count FROM ordered GROUP BY session_key
                )
                UNION ALL SELECT 'avg_journeys_per_session', cast(round(avg(journey_count), 4) AS varchar) FROM (
                    SELECT session_key, count(*) AS journey_count FROM ordered GROUP BY session_key
                )
                UNION ALL SELECT 'avg_journeys_per_customer', cast(round(avg(journey_count), 4) AS varchar) FROM (
                    SELECT customer_id, count(*) AS journey_count FROM ordered GROUP BY customer_id
                )
            ) AS metrics ORDER BY metric
        """
        copy_query(connection, summary_total, output_dir / "summary_total.csv")
        copy_query(
            connection,
            """
            SELECT platform, count(*) AS journeys,
                   count(DISTINCT CASE WHEN customer_id <> '__anonymous__' THEN customer_id END) AS unique_customers,
                   count(DISTINCT session_id) AS unique_session_ids,
                   count(DISTINCT session_key) AS platform_session_keys,
                   min(start_ts_value) AS first_start_ts,
                   max(start_ts_value) AS last_start_ts,
                   round(avg(try_cast(nullif(trim(n_events_final), '') AS DOUBLE)), 4) AS avg_events_final
            FROM ordered
            GROUP BY platform
            ORDER BY platform
            """,
            output_dir / "summary_platform.csv",
        )

        level_queries = []
        for level, column in LEVELS:
            level_queries.append(
                f"""
                SELECT {sql_string(level)} AS level, {column} AS value,
                       count(*) AS journey_count,
                       count(DISTINCT customer_id) AS unique_customers,
                       count(DISTINCT session_key) AS unique_sessions
                FROM ordered
                GROUP BY {column}
                """
            )
        copy_query(
            connection,
            " UNION ALL ".join(level_queries) + " ORDER BY level, journey_count DESC, value",
            output_dir / "summary_level_counts.csv",
        )
        copy_query(
            connection,
            """
            SELECT customer_id,
                   count(*) AS journey_count,
                   count(DISTINCT session_key) AS session_count,
                   count(DISTINCT platform) AS platform_count,
                   min(start_ts_value) AS first_journey_ts,
                   max(end_ts_value) AS last_journey_ts,
                   count(DISTINCT business_family) AS business_family_count,
                   count(DISTINCT business_submodule) AS business_submodule_count,
                   count(DISTINCT cluster_id) AS cluster_count
            FROM ordered
            GROUP BY customer_id
            ORDER BY journey_count DESC, customer_id
            """,
            output_dir / "customer_summary.csv",
        )

        write_session_sequences_streaming(connection, output_dir / "session_sequences.csv")

        transition_parts = []
        for level, current, following in (
            ("business_family", "business_family", "next_business_family"),
            ("business_submodule", "business_submodule", "next_business_submodule"),
            ("cluster", "cluster_name", "next_cluster"),
        ):
            transition_parts.append(
                f"""
                SELECT {sql_string(level)} AS level,
                       {current} AS from_value,
                       {following} AS to_value,
                       count(*) AS transition_count,
                       count(DISTINCT session_key) AS unique_sessions,
                       count(DISTINCT customer_id) AS unique_customers,
                       count(DISTINCT platform) AS platforms
                FROM ordered
                WHERE {following} IS NOT NULL
                GROUP BY {current}, {following}
                """
            )
        transition_query = f"""
            WITH counts AS ({' UNION ALL '.join(transition_parts)})
            SELECT *, round(transition_count::DOUBLE / sum(transition_count) OVER (PARTITION BY level), 8) AS transition_share
            FROM counts
            ORDER BY level, transition_count DESC, from_value, to_value
        """
        copy_query(connection, transition_query, output_dir / "transition_counts.csv")

        print("Selecting bounded detail set for visualisation...")
        customer_filter = ""
        if args.top_customers:
            customer_filter = f"""
                AND customer_id IN (
                    SELECT customer_id FROM (
                        SELECT customer_id, count(*) AS journey_count
                        FROM ordered
                        GROUP BY customer_id
                        ORDER BY journey_count DESC, customer_id
                        LIMIT {int(args.top_customers)}
                    ) AS selected_customers
                )
            """
        session_filter = ""
        if args.top_sessions:
            session_filter = f"""
                AND session_key IN (
                    SELECT session_key FROM (
                        SELECT session_key, count(*) AS journey_count, min(start_ts_value) AS first_start_ts
                        FROM ordered
                        WHERE true {customer_filter}
                        GROUP BY session_key
                        ORDER BY journey_count DESC, first_start_ts, session_key
                        LIMIT {int(args.top_sessions)}
                    ) AS selected_sessions
                )
            """
        detail_query = f"""
            SELECT customer_id, platform, session_id, session_key,
                   journey_id, session_journey_index,
                   start_ts, end_ts, start_ts_value, end_ts_value,
                   previous_journey_id, next_journey_id,
                   cluster, cluster_id, cluster_name, cluster_name_en,
                   business_family, business_family_code, business_submodule, business_detail,
                   previous_business_family, next_business_family,
                   previous_business_submodule, next_business_submodule,
                   previous_cluster, next_cluster,
                   n_events_raw, n_events_final, sequence,
                   severe_anomaly, friction_flags, next_action
            FROM ordered
            WHERE true {customer_filter} {session_filter}
            ORDER BY customer_id, platform, session_id, session_journey_index
        """
        copy_query(connection, detail_query, output_dir / "journey_path_detail.csv")

        metadata = output_dir / "README.md"
        metadata.write_text(
            "# Customer journey sequence export\n\n"
            "The source rows are journey-level scored records. `start_ts` orders journeys "
            "inside `(customer_id, platform, session_id)`; the token `sequence` has no "
            "per-token timestamp in this input.\n\n"
            "- `summary_total.csv`, `summary_platform.csv`, `summary_level_counts.csv`: "
            "whole-input statistics; no visualisation is required for these files.\n"
            "- `session_sequences.csv`: one compact chronological sequence per session.\n"
            "- `transition_counts.csv`: previous-to-next transitions at business family, "
            "submodule and cluster levels.\n"
            "- `journey_path_detail.csv`: detail used by `visualize/pm4py_visualize.py`; "
            "limited to top sessions unless `--top-sessions 0` was passed.\n",
            encoding="utf-8",
        )
        print(f"Done. Generated sequence data under {output_dir}")
    finally:
        connection.close()
        if not args.keep_duckdb:
            try:
                db_path.unlink()
            except FileNotFoundError:
                pass
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    build(parse_args())
