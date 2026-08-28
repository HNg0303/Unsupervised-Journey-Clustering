"""Shared bounded CSV/DuckDB helpers for independently runnable analyses."""

from __future__ import annotations

import csv
from pathlib import Path

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None


def require_duckdb():
    if duckdb is None:
        raise SystemExit("duckdb is required for multi-GB named CSV post-analysis")
    return duckdb


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def read_header(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return set(next(csv.reader(handle), []))


def create_connection(output_dir: Path, memory_limit: str, threads: int):
    db = require_duckdb()
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = db.connect()
    connection.execute(f"SET memory_limit='{memory_limit}'")
    connection.execute(f"SET threads={max(1, int(threads))}")
    connection.execute("SET preserve_insertion_order=false")
    temp_dir = output_dir / ".duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory={sql_literal(str(temp_dir))}")
    return connection


def register_named_journeys(connection, android: Path, ios: Path) -> None:
    optional_columns = {
        "sequence", "cluster", "cluster_id", "cluster_name", "cluster_name_en",
        "business_family", "business_family_code", "business_submodule", "business_detail",
        "mapping_function_code", "naming_confidence", "assignment_type", "start_ts", "end_ts",
        "customer_id", "journey_id",
    }
    for name, platform, path in (("android_src", "android", android), ("ios_src", "ios", ios)):
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
        existing = read_header(path)
        extras = ",\n               ".join(f"'' AS \"{column}\"" for column in sorted(optional_columns - existing))
        star = "* EXCLUDE (platform)" if "platform" in existing else "*"
        projection = star + (",\n               " + extras if extras else "")
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {name} AS
            SELECT {projection}, {sql_literal(platform)} AS source_platform
            FROM read_csv_auto({sql_literal(str(path.resolve()))}, header=true, all_varchar=true, encoding='utf-8')
            """
        )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW named_journeys AS
        SELECT *, source_platform AS platform,
               try_cast(start_ts AS TIMESTAMPTZ) AS journey_start,
               try_cast(coalesce(nullif(end_ts, ''), start_ts) AS TIMESTAMPTZ) AS journey_end
        FROM android_src
        UNION ALL BY NAME
        SELECT *, source_platform AS platform,
               try_cast(start_ts AS TIMESTAMPTZ) AS journey_start,
               try_cast(coalesce(nullif(end_ts, ''), start_ts) AS TIMESTAMPTZ) AS journey_end
        FROM ios_src
        """
    )
