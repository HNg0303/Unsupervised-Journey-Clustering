#!/usr/bin/env python3
"""Precompute compact, HTML-ready summaries from named inference journeys.

The input inspection CSVs are several GB, so DuckDB scans them out-of-core and can
spill to disk.  The generated CSV/JSON files contain only aggregates and bounded
examples; a static HTML dashboard never needs to download the row-level exports.

Example (from the repository root)::

    python scripts/extract_inference_dashboard_summaries.py \
      --android output/scores/pca48_ngrams12_500/android/model_version=latest/platform=android/inspection/android_all_named.csv \
      --ios output/scores/pca48_ngrams12_500/ios/model_version=latest/platform=ios/inspection/ios_all_named.csv \
      --naming output/scores/pca48_ngrams12_500/Cluster_naming.csv \
      --output-dir output/scores/pca48_ngrams12_500/html_dashboard_summary

All timestamps in output tables are UTC.  ``hour_local`` uses ``--utc-offset-hours``
(default: 7 for Viet Nam).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {
    "journey_id", "session_id", "customer_id", "start_ts", "cluster",
    "cluster_name", "business_family", "n_events_final", "span_seconds",
    "assignment_type", "friction_flags", "behavioral_friction_flags",
}

USED_COLUMNS = REQUIRED_COLUMNS | {
    "end_ts", "cluster_name_en", "business_family_code", "business_submodule",
    "naming_confidence", "back_rate", "distance_to_centroid", "entry_token",
    "exit_token", "effective_next_action", "next_action",
    "effective_next_action_share", "next_action_share", "sequence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android", type=Path, help="Android *_all_named.csv")
    parser.add_argument("--ios", type=Path, help="iOS *_all_named.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--naming", type=Path, help="Authoritative Cluster_naming.csv")
    parser.add_argument("--utc-offset-hours", type=int, default=7)
    parser.add_argument("--example-limit", type=int, default=100)
    parser.add_argument("--top-limit", type=int, default=50)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--keep-duckdb", action="store_true")
    args = parser.parse_args()
    if not args.android and not args.ios:
        parser.error("at least one of --android or --ios is required")
    if args.example_limit < 0 or args.top_limit < 1:
        parser.error("--example-limit must be >= 0 and --top-limit must be >= 1")
    return args


def sql_string(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def read_header(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {value.strip() for value in next(csv.reader(handle), [])}


def copy_query(connection, query: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    connection.execute(
        f"COPY ({query}) TO {sql_string(output_path.resolve())} "
        "(FORMAT CSV, HEADER, DELIMITER ',')"
    )


def create_source_view(
    connection, view: str, path: Path, platform: str, header: set[str]
) -> None:
    missing_projection = ", ".join(
        f"NULL::VARCHAR AS \"{column}\"" for column in sorted(USED_COLUMNS - header)
    )
    if missing_projection:
        missing_projection = ", " + missing_projection
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW {view} AS
        SELECT *, {sql_string(platform)}::VARCHAR AS source_platform{missing_projection}
        FROM read_csv_auto(
            {sql_string(path.resolve())}, header=true, all_varchar=true,
            encoding='utf-8', sample_size=100000, ignore_errors=false
        )
        """
    )


def create_naming_view(connection, naming_path: Path, header: set[str]) -> None:
    """Register a stable naming projection for canonical and monthly mapping files.

    The original dashboard run uses ``Cluster_naming.csv`` with canonical level columns,
    while monthly exports may provide the equivalent ``*_cluster_mapping.csv`` with
    ``business_submodule``/``business_detail`` instead.  Normalising the optional fields
    here keeps the aggregate SQL identical for both layouts.
    """
    expressions = {
        "platform": "NULL::VARCHAR AS platform",
        "cluster_id": 'try_cast(cluster_id AS BIGINT) AS cluster_id',
        "cluster_name": "cluster_name",
        "canonical_level_2_code": "NULL::VARCHAR AS canonical_level_2_code",
        "cluster_name_level_2": "NULL::VARCHAR AS cluster_name_level_2",
        "cluster_name_en": "NULL::VARCHAR AS cluster_name_en",
        "business_family": "business_family",
        "business_family_code": "NULL::VARCHAR AS business_family_code",
        "business_submodule": "NULL::VARCHAR AS business_submodule",
        "naming_confidence": "naming_confidence",
    }
    aliases = {
        "canonical_level_2_code": "cluster_name_en",
        "cluster_name_level_2": "business_submodule",
        "business_family_code": "business_family",
    }
    projection = []
    for column, expression in expressions.items():
        source = column if column in header else aliases.get(column)
        if source and source in header:
            if column == "cluster_id":
                projection.append(expression)
            else:
                projection.append(f'"{source}" AS "{column}"')
        else:
            projection.append(expression)
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW cluster_names AS
        SELECT {', '.join(projection)}
        FROM read_csv_auto(
            {sql_string(naming_path)}, header=true, all_varchar=true,
            encoding='utf-8', sample_size=100000, ignore_errors=false
        )
        """
    )


def write_readme(output_dir: Path) -> None:
    files = {
        "kpi.csv": "KPI tổng quan theo toàn bộ dữ liệu và từng platform.",
        "cluster_summary.csv": "Volume, customer, session, duration, steps và friction theo cluster.",
        "journey_type_summary.csv": "Tổng hợp journey type có tên (nhiều cluster có thể cùng type).",
        "family_summary.csv": "Tổng hợp theo business family.",
        "cluster_heatmap.csv": "Heatmap cluster × giờ địa phương; có count và tỷ lệ chuẩn hóa trong cluster.",
        "family_heatmap.csv": "Heatmap family × giờ địa phương.",
        "daily_trend.csv": "Trend ngày × platform × family × journey type.",
        "daily_kpi.csv": "KPI theo ngày × platform; session/customer là distinct trong ngày.",
        "daily_cluster_summary.csv": "KPI theo ngày × platform × cluster.",
        "daily_family_summary.csv": "KPI theo ngày × platform × business family.",
        "friction_summary.csv": "Tần suất từng friction flag (một journey có thể có nhiều flag).",
        "daily_friction_summary.csv": "Friction flag theo ngày × platform.",
        "daily_friction_by_type.csv": "Friction theo ngày × journey type.",
        "friction_by_type.csv": "Friction rate và thời gian dư theo journey type.",
        "novel_daily.csv": "Novel/unrecognised rate theo ngày và platform.",
        "novel_entry_exit.csv": "Top entry/exit của novel journeys.",
        "next_action_summary.csv": "Next action phổ biến và confidence trung bình.",
        "daily_next_action_summary.csv": "Next action theo ngày × platform.",
        "session_composition.csv": "Phân bố số journey và số family trong session.",
        "journey_examples.csv": "Mẫu bounded cho explorer/audit; không phải dữ liệu thống kê.",
        "manifest.json": "Data contract, tham số chạy, input và danh sách output.",
    }
    lines = ["# HTML dashboard summary", "", "Generated from journey-level inference CSVs.", ""]
    lines.extend(f"- `{name}` — {description}" for name, description in files.items())
    lines += ["", "Tất cả ngày/giờ nguồn là UTC; heatmap dùng offset ghi trong `manifest.json`.", ""]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("duckdb is required; install project requirements first") from exc

    inputs: list[tuple[str, Path, set[str]]] = []
    for platform, path in (("android", args.android), ("ios", args.ios)):
        if path:
            header = read_header(path)
            missing = REQUIRED_COLUMNS - header
            if missing:
                raise SystemExit(f"{path}: missing required columns: {sorted(missing)}")
            inputs.append((platform, path, header))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    naming_path = (args.naming or (output_dir.parent / "Cluster_naming.csv")).resolve()
    if not naming_path.exists():
        raise SystemExit(f"authoritative naming file not found: {naming_path}")
    db_path = output_dir / ".dashboard_summary.duckdb"
    temp_dir = output_dir / ".duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(f"SET memory_limit={sql_string(args.memory_limit)}")
        connection.execute(f"SET threads={int(args.threads)}")
        connection.execute(f"SET temp_directory={sql_string(temp_dir.resolve())}")
        connection.execute("SET TimeZone='UTC'")

        views = []
        for index, (platform, path, header) in enumerate(inputs):
            view = f"source_{index}"
            print(f"Registering {platform}: {path}")
            create_source_view(connection, view, path, platform, header)
            views.append(f"SELECT * FROM {view}")

        # The scored row-level exports can carry stale or row-level fallback labels.  Join
        # the frozen business naming table once here so every aggregate and example uses the
        # same platform + cluster namespace as the dashboard.
        naming_header = read_header(naming_path)
        create_naming_view(connection, naming_path, naming_header)

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE journeys AS
            WITH raw AS ({' UNION ALL BY NAME '.join(views)}),
            names AS (
                SELECT *
                FROM cluster_names
                QUALIFY row_number() OVER (
                    PARTITION BY platform, try_cast(cluster_id AS BIGINT)
                    ORDER BY naming_confidence DESC NULLS LAST
                ) = 1
            )
            SELECT
                r.source_platform AS platform,
                coalesce(nullif(trim(r.journey_id), ''), '__missing__') AS journey_id,
                coalesce(nullif(trim(r.session_id), ''), '__missing__') AS session_id,
                coalesce(nullif(trim(r.customer_id), ''), '__anonymous__') AS customer_id,
                try_cast(r.start_ts AS TIMESTAMPTZ) AS start_ts,
                try_cast(r.end_ts AS TIMESTAMPTZ) AS end_ts,
                try_cast(r.cluster AS BIGINT) AS cluster,
                coalesce(nullif(trim(n.cluster_name), ''), nullif(trim(r.cluster_name), ''), 'Chưa phân loại') AS journey_type,
                coalesce(nullif(trim(n.canonical_level_2_code), ''), nullif(trim(r.cluster_name_en), ''), nullif(trim(r.cluster_name), ''), 'unclassified') AS journey_type_en,
                coalesce(nullif(trim(n.business_family), ''), nullif(trim(r.business_family), ''), 'Chưa phân loại') AS business_family,
                split_part(coalesce(nullif(trim(n.canonical_level_2_code), ''), nullif(trim(r.cluster_name_en), ''), nullif(trim(r.business_family_code), ''), 'unclassified'), '.', 1) AS business_family_code,
                coalesce(nullif(trim(n.cluster_name_level_2), ''), nullif(trim(r.business_submodule), ''), 'Chưa phân loại') AS business_submodule,
                coalesce(nullif(trim(n.naming_confidence), ''), nullif(trim(r.naming_confidence), ''), 'not_applicable') AS naming_confidence,
                try_cast(r.n_events_final AS DOUBLE) AS n_events_final,
                try_cast(r.span_seconds AS DOUBLE) AS span_seconds,
                try_cast(r.back_rate AS DOUBLE) AS back_rate,
                try_cast(r.distance_to_centroid AS DOUBLE) AS distance_to_centroid,
                NOT regexp_matches(lower(coalesce(r.assignment_type, '')), '(novel|unknown|unassigned)') AS is_known,
                coalesce(r.behavioral_friction_flags, '') <> '' AS has_struggle,
                coalesce(r.friction_flags, '') <> '' AS has_any_flag,
                coalesce(r.behavioral_friction_flags, '') AS behavioral_friction_flags,
                coalesce(r.friction_flags, '') AS friction_flags,
                coalesce(nullif(trim(r.entry_token), ''), '__missing__') AS entry_token,
                coalesce(nullif(trim(r.exit_token), ''), '__missing__') AS exit_token,
                coalesce(nullif(trim(r.effective_next_action), ''), nullif(trim(r.next_action), '')) AS next_action,
                try_cast(coalesce(nullif(trim(r.effective_next_action_share), ''), nullif(trim(r.next_action_share), '')) AS DOUBLE) AS next_action_share,
                coalesce(r.sequence, '') AS sequence
            FROM raw r
            LEFT JOIN names n
              ON n.platform = r.source_platform
             AND try_cast(n.cluster_id AS BIGINT) = try_cast(r.cluster AS BIGINT)
            """
        )
        connection.execute("CREATE INDEX journey_session_idx ON journeys(platform, session_id)")

        # Epoch/zero timestamps occasionally appear in scored exports when an upstream
        # parser cannot materialise a source timestamp.  They are not valid production
        # dates and would otherwise create a misleading 1970 bucket in timeframe controls.
        scope = "start_ts IS NOT NULL AND start_ts >= TIMESTAMP '2000-01-01'"
        session_key = "platform || ':' || session_id"
        copy_query(connection, f"""
            WITH scoped AS (SELECT * FROM journeys WHERE {scope}),
            base AS (
                SELECT 'all' AS scope, count(*) AS journeys,
                       count(DISTINCT {session_key}) AS sessions,
                       count(DISTINCT customer_id) FILTER (WHERE customer_id <> '__anonymous__') AS customers,
                       count(DISTINCT platform || ':' || cluster) FILTER (WHERE is_known) AS clusters,
                       count(DISTINCT journey_type_en) FILTER (WHERE is_known) AS journey_types,
                       count(DISTINCT business_family) AS families,
                       avg(is_known::INTEGER) AS known_rate,
                       avg(has_struggle::INTEGER) AS struggle_rate,
                       avg((has_struggle OR has_any_flag)::INTEGER) AS any_flag_rate,
                       avg(n_events_final) AS mean_steps, median(n_events_final) AS median_steps,
                       avg(span_seconds) AS mean_span_seconds, median(span_seconds) AS median_span_seconds,
                       min(start_ts) AS first_start_ts, max(start_ts) AS last_start_ts
                FROM scoped
                UNION ALL
                SELECT platform, count(*), count(DISTINCT {session_key}),
                       count(DISTINCT customer_id) FILTER (WHERE customer_id <> '__anonymous__'),
                       count(DISTINCT cluster) FILTER (WHERE is_known),
                       count(DISTINCT journey_type_en) FILTER (WHERE is_known),
                       count(DISTINCT business_family), avg(is_known::INTEGER),
                       avg(has_struggle::INTEGER), avg((has_struggle OR has_any_flag)::INTEGER),
                       avg(n_events_final), median(n_events_final), avg(span_seconds), median(span_seconds),
                       min(start_ts), max(start_ts)
                FROM scoped GROUP BY platform
            ) SELECT * FROM base ORDER BY scope
        """, output_dir / "kpi.csv")

        common_group_metrics = f"""
            count(*) AS journeys,
            count(DISTINCT {session_key}) AS sessions,
            count(DISTINCT customer_id) FILTER (WHERE customer_id <> '__anonymous__') AS customers,
            avg(has_struggle::INTEGER) AS struggle_rate,
            avg((has_struggle OR has_any_flag)::INTEGER) AS any_flag_rate,
            median(n_events_final) AS median_steps,
            quantile_cont(n_events_final, 0.9) AS p90_steps,
            median(span_seconds) AS median_span_seconds,
            quantile_cont(span_seconds, 0.9) AS p90_span_seconds,
            avg(back_rate) AS mean_back_rate,
            avg(distance_to_centroid) AS mean_distance_to_centroid
        """
        copy_query(connection, f"""
            WITH grouped AS (
                SELECT platform, cluster, journey_type, journey_type_en, business_family,
                       business_family_code, business_submodule, naming_confidence,
                       {common_group_metrics}
                FROM journeys WHERE {scope} GROUP BY ALL
            )
            SELECT *, journeys / sum(journeys) OVER (PARTITION BY platform)::DOUBLE AS platform_share
            FROM grouped ORDER BY platform, journeys DESC, cluster
        """, output_dir / "cluster_summary.csv")
        copy_query(connection, f"""
            WITH grouped AS (
                SELECT platform, journey_type, journey_type_en, business_family, business_family_code,
                       {common_group_metrics}, count(DISTINCT cluster) AS cluster_count
                FROM journeys WHERE {scope} GROUP BY ALL
            )
            SELECT *, journeys / sum(journeys) OVER (PARTITION BY platform)::DOUBLE AS platform_share
            FROM grouped ORDER BY platform, journeys DESC, journey_type
        """, output_dir / "journey_type_summary.csv")
        copy_query(connection, f"""
            WITH grouped AS (
                SELECT platform, business_family, business_family_code,
                       {common_group_metrics}, count(DISTINCT cluster) AS cluster_count,
                       count(DISTINCT journey_type_en) AS journey_type_count
                FROM journeys WHERE {scope} GROUP BY ALL
            )
            SELECT *, journeys / sum(journeys) OVER (PARTITION BY platform)::DOUBLE AS platform_share
            FROM grouped ORDER BY platform, journeys DESC, business_family
        """, output_dir / "family_summary.csv")

        offset = int(args.utc_offset_hours)
        copy_query(connection, f"""
            WITH counts AS (
                SELECT platform, cluster, journey_type, business_family,
                       extract(hour FROM start_ts + INTERVAL '{offset} hours')::INTEGER AS hour_local,
                       count(*) AS journeys
                FROM journeys WHERE {scope} GROUP BY ALL
            )
            SELECT *, journeys / sum(journeys) OVER (PARTITION BY platform, cluster)::DOUBLE AS cluster_hour_share
            FROM counts ORDER BY platform, journeys DESC, cluster, hour_local
        """, output_dir / "cluster_heatmap.csv")
        copy_query(connection, f"""
            WITH counts AS (
                SELECT platform, business_family,
                       extract(hour FROM start_ts + INTERVAL '{offset} hours')::INTEGER AS hour_local,
                       count(*) AS journeys
                FROM journeys WHERE {scope} GROUP BY ALL
            )
            SELECT *, journeys / sum(journeys) OVER (PARTITION BY platform, business_family)::DOUBLE AS family_hour_share
            FROM counts ORDER BY platform, business_family, hour_local
        """, output_dir / "family_heatmap.csv")
        copy_query(connection, f"""
            SELECT CAST(start_ts AS DATE) AS date, platform, business_family,
                   business_family_code, journey_type, is_known,
                   count(*) AS journeys, count(DISTINCT {session_key}) AS sessions,
                   count(DISTINCT customer_id) FILTER (WHERE customer_id <> '__anonymous__') AS customers,
                   avg(has_struggle::INTEGER) AS struggle_rate,
                   median(span_seconds) AS median_span_seconds
            FROM journeys WHERE {scope} GROUP BY ALL ORDER BY date, platform, journeys DESC
        """, output_dir / "daily_trend.csv")

        # Date-grain summaries let the Streamlit page apply the same timeframe to KPIs,
        # family/cluster tables, friction and next-action charts.  They intentionally keep
        # daily distinct session/customer counts; the UI does not add those counts across
        # multiple days because that would double-count people who return.
        copy_query(connection, f"""
            SELECT CAST(start_ts AS DATE) AS date, platform,
                   count(*) AS journeys,
                   count(DISTINCT {session_key}) AS sessions,
                   count(DISTINCT customer_id) FILTER (WHERE customer_id <> '__anonymous__') AS customers,
                   avg(is_known::INTEGER) AS known_rate,
                   avg(has_struggle::INTEGER) AS struggle_rate,
                   avg((has_struggle OR has_any_flag)::INTEGER) AS any_flag_rate,
                   avg(n_events_final) AS mean_steps,
                   avg(span_seconds) AS mean_span_seconds
            FROM journeys WHERE {scope} GROUP BY ALL ORDER BY date, platform
        """, output_dir / "daily_kpi.csv")
        copy_query(connection, f"""
            SELECT CAST(start_ts AS DATE) AS date, platform, cluster, journey_type,
                   journey_type_en, business_family, business_family_code, business_submodule,
                   naming_confidence, {common_group_metrics}
            FROM journeys WHERE {scope} GROUP BY ALL ORDER BY date, platform, journeys DESC, cluster
        """, output_dir / "daily_cluster_summary.csv")
        copy_query(connection, f"""
            SELECT CAST(start_ts AS DATE) AS date, platform, business_family, business_family_code,
                   {common_group_metrics}, count(DISTINCT cluster) AS cluster_count,
                   count(DISTINCT journey_type_en) AS journey_type_count
            FROM journeys WHERE {scope} GROUP BY ALL ORDER BY date, platform, journeys DESC, business_family
        """, output_dir / "daily_family_summary.csv")

        copy_query(connection, f"""
            WITH scoped AS (SELECT * FROM journeys WHERE {scope}),
            flags AS (
                SELECT platform, 'behavioral' AS flag_source, unnest(string_split(behavioral_friction_flags, '|')) AS flag
                FROM scoped WHERE behavioral_friction_flags <> ''
                UNION ALL
                SELECT platform, 'model', unnest(string_split(friction_flags, '|'))
                FROM scoped WHERE friction_flags <> ''
            ), totals AS (SELECT platform, count(*) AS total FROM scoped GROUP BY platform)
            SELECT flags.platform, flag_source, flag, count(*) AS journeys,
                   count(*) / max(total)::DOUBLE AS journey_share
            FROM flags JOIN totals USING (platform) WHERE flag <> ''
            GROUP BY flags.platform, flag_source, flag ORDER BY platform, journeys DESC
        """, output_dir / "friction_summary.csv")
        copy_query(connection, f"""
            WITH scoped AS (SELECT * FROM journeys WHERE {scope}),
            flags AS (
                SELECT CAST(start_ts AS DATE) AS date, platform, 'behavioral' AS flag_source,
                       unnest(string_split(behavioral_friction_flags, '|')) AS flag
                FROM scoped WHERE behavioral_friction_flags <> ''
                UNION ALL
                SELECT CAST(start_ts AS DATE) AS date, platform, 'model',
                       unnest(string_split(friction_flags, '|'))
                FROM scoped WHERE friction_flags <> ''
            ), totals AS (
                SELECT CAST(start_ts AS DATE) AS date, platform, count(*) AS total
                FROM scoped GROUP BY ALL
            )
            SELECT flags.date, flags.platform, flag_source, flag, count(*) AS journeys,
                   count(*) / max(total)::DOUBLE AS journey_share
            FROM flags JOIN totals USING (date, platform) WHERE flag <> ''
            GROUP BY flags.date, flags.platform, flag_source, flag
            ORDER BY date, platform, journeys DESC
        """, output_dir / "daily_friction_summary.csv")
        copy_query(connection, f"""
            WITH scoped AS (SELECT * FROM journeys WHERE {scope}),
            clean AS (
                SELECT platform, journey_type, median(span_seconds) AS clean_median_seconds
                FROM scoped WHERE NOT has_struggle GROUP BY platform, journey_type
            )
            SELECT s.platform, s.journey_type, s.business_family, count(*) AS journeys,
                   sum(s.has_struggle::INTEGER) AS struggling_journeys,
                   avg(s.has_struggle::INTEGER) AS struggle_rate,
                   sum(CASE WHEN s.has_struggle THEN greatest(s.span_seconds - c.clean_median_seconds, 0) ELSE 0 END) AS excess_seconds,
                   median(s.n_events_final) AS median_steps, median(s.span_seconds) AS median_span_seconds
            FROM scoped s LEFT JOIN clean c USING (platform, journey_type)
            GROUP BY ALL ORDER BY struggle_rate DESC, journeys DESC
        """, output_dir / "friction_by_type.csv")
        copy_query(connection, f"""
            WITH scoped AS (SELECT * FROM journeys WHERE {scope}),
            clean AS (
                SELECT platform, CAST(start_ts AS DATE) AS date, journey_type,
                       median(span_seconds) AS clean_median_seconds
                FROM scoped WHERE NOT has_struggle GROUP BY ALL
            )
            SELECT CAST(s.start_ts AS DATE) AS date, s.platform, s.journey_type, s.business_family,
                   count(*) AS journeys, sum(s.has_struggle::INTEGER) AS struggling_journeys,
                   avg(s.has_struggle::INTEGER) AS struggle_rate,
                   sum(CASE WHEN s.has_struggle THEN greatest(s.span_seconds - c.clean_median_seconds, 0) ELSE 0 END) AS excess_seconds,
                   median(s.n_events_final) AS median_steps, median(s.span_seconds) AS median_span_seconds
            FROM scoped s LEFT JOIN clean c
              ON c.platform = s.platform AND c.date = CAST(s.start_ts AS DATE) AND c.journey_type = s.journey_type
            GROUP BY ALL ORDER BY date, struggle_rate DESC, journeys DESC
        """, output_dir / "daily_friction_by_type.csv")

        copy_query(connection, f"""
            SELECT CAST(start_ts AS DATE) AS date, platform, count(*) AS journeys,
                   sum((NOT is_known)::INTEGER) AS novel_journeys,
                   avg((NOT is_known)::INTEGER) AS novel_rate,
                   count(DISTINCT customer_id) FILTER (WHERE NOT is_known AND customer_id <> '__anonymous__') AS novel_customers
            FROM journeys WHERE {scope} GROUP BY ALL ORDER BY date, platform
        """, output_dir / "novel_daily.csv")
        copy_query(connection, f"""
            WITH values AS (
                SELECT platform, 'entry' AS token_type, entry_token AS token FROM journeys WHERE {scope} AND NOT is_known
                UNION ALL SELECT platform, 'exit', exit_token FROM journeys WHERE {scope} AND NOT is_known
            ), grouped AS (
                SELECT platform, token_type, token, count(*) AS journeys FROM values GROUP BY ALL
            ), ranked AS (
                SELECT *, row_number() OVER (PARTITION BY platform, token_type ORDER BY journeys DESC, token) AS rank
                FROM grouped
            ) SELECT * FROM ranked WHERE rank <= {int(args.top_limit)} ORDER BY platform, token_type, rank
        """, output_dir / "novel_entry_exit.csv")
        copy_query(connection, f"""
            WITH grouped AS (
                SELECT platform, next_action, count(*) AS journeys,
                       avg(next_action_share) AS mean_confidence
                FROM journeys WHERE {scope} AND next_action IS NOT NULL GROUP BY platform, next_action
            ), ranked AS (
                SELECT *, journeys / sum(journeys) OVER (PARTITION BY platform)::DOUBLE AS action_share,
                       row_number() OVER (PARTITION BY platform ORDER BY journeys DESC, next_action) AS rank
                FROM grouped
            ) SELECT * FROM ranked WHERE rank <= {int(args.top_limit)} ORDER BY platform, rank
        """, output_dir / "next_action_summary.csv")
        copy_query(connection, f"""
            WITH grouped AS (
                SELECT CAST(start_ts AS DATE) AS date, platform, next_action, count(*) AS journeys,
                       avg(next_action_share) AS mean_confidence
                FROM journeys WHERE {scope} AND next_action IS NOT NULL GROUP BY ALL
            )
            SELECT *, journeys / sum(journeys) OVER (PARTITION BY date, platform)::DOUBLE AS action_share
            FROM grouped ORDER BY date, platform, journeys DESC, next_action
        """, output_dir / "daily_next_action_summary.csv")
        copy_query(connection, f"""
            WITH per_session AS (
                SELECT platform, session_id, count(*) AS journey_count,
                       count(DISTINCT business_family) AS family_count,
                       count(DISTINCT journey_type) AS journey_type_count
                FROM journeys WHERE {scope} GROUP BY platform, session_id
            )
            SELECT platform, least(journey_count, 10) AS journey_count_bucket,
                   least(family_count, 5) AS family_count_bucket,
                   count(*) AS sessions, avg(journey_type_count) AS mean_journey_types
            FROM per_session GROUP BY ALL ORDER BY platform, journey_count_bucket, family_count_bucket
        """, output_dir / "session_composition.csv")

        if args.example_limit:
            copy_query(connection, f"""
                WITH ranked AS (
                    SELECT journey_id, platform, start_ts, cluster, journey_type, business_family,
                           is_known, has_struggle, n_events_final, span_seconds, back_rate,
                           behavioral_friction_flags, friction_flags, entry_token, exit_token,
                           next_action, sequence,
                           row_number() OVER (
                               PARTITION BY platform, is_known, has_struggle
                               ORDER BY span_seconds DESC NULLS LAST, journey_id
                           ) AS example_rank
                    FROM journeys WHERE {scope}
                ) SELECT * FROM ranked WHERE example_rank <= {int(args.example_limit)}
                ORDER BY platform, is_known, has_struggle, example_rank
            """, output_dir / "journey_examples.csv")

        generated = sorted(path.name for path in output_dir.iterdir() if path.suffix in {".csv", ".json"})
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "timezone": {"source": "UTC", "heatmap_utc_offset_hours": offset},
            "parameters": {"top_limit": args.top_limit, "example_limit_per_segment": args.example_limit},
            "inputs": [
                {"platform": platform, "path": str(path.resolve()), "size_bytes": path.stat().st_size}
                for platform, path, _ in inputs
            ],
            "naming": {"path": str(naming_path), "size_bytes": naming_path.stat().st_size},
            "outputs": generated,
            "notes": [
                "Statistics include only rows with a valid start_ts on or after 2000-01-01; malformed epoch timestamps are excluded.",
                "Journey examples are bounded audit samples and must not be used for aggregate statistics.",
                "A journey may contribute to multiple friction flags.",
                "Cluster and business labels are rejoined from the supplied cluster mapping on (platform, cluster_id).",
            ],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_readme(output_dir)
        print(f"Done: {output_dir}")
    finally:
        connection.close()
        if not args.keep_duckdb:
            db_path.unlink(missing_ok=True)
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    build(parse_args())
