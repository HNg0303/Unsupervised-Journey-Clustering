#!/usr/bin/env python3
"""Run the Loyalty customer analysis independently of the full customer pipeline.

DuckDB reads the two large named CSVs in a spill-capable relation and writes the outputs
without materialising them in pandas or Python. This is intended to be run first when
only the Loyalty question is needed.

Example:
    python scripts/post_analysis/analyze_loyalty.py \
      --android output/scores/.../android_all_named.csv \
      --ios output/scores/.../ios_all_named.csv \
      --output-dir output/scores/.../shareholder_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - dependency is declared in requirements.txt
    raise SystemExit("duckdb is required; run `pip install -r requirements.txt`") from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - minimal production/runtime images may omit tqdm
    class _NoopProgress:
        def __init__(self, *args, **kwargs):
            pass

        def set_postfix_str(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

        def close(self):
            pass

    def tqdm(*args, **kwargs):
        return _NoopProgress()


NAMED_COLUMNS = [
    "journey_id", "session_id", "device_id", "customer_id", "start_ts", "end_ts",
    "sequence", "cluster", "cluster_name", "cluster_name_en", "business_family",
    "business_family_code", "business_submodule", "business_detail", "mapping_function_code",
    "naming_confidence", "assignment_type",
]

LOYALTY_RE = (
    r"loyalty|loysdk|voucher|reward|points?|membership|offer|hot deal|"
    r"khách hàng thân thiết|ưu đãi|quà tặng"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android", type=Path, required=True)
    parser.add_argument("--ios", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", default="4GB", help="DuckDB memory limit; overflow spills to temp directory")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def select_named_columns(path: Path) -> str:
    """Project the stable Loyalty schema, filling optional mapping fields when absent."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        existing = set(next(csv.reader(handle), []))
    return ", ".join(
        f'"{column}"' if column in existing else f"'' AS \"{column}\""
        for column in NAMED_COLUMNS
    )


def copy_query(connection, query: str, destination: Path) -> None:
    connection.execute(
        f"COPY ({query}) TO {sql_literal(destination)} (FORMAT CSV, HEADER, DELIMITER ',')"
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for path in (args.android, args.ios):
        if not path.exists():
            raise SystemExit(f"input not found: {path}")

    temp_dir = out / ".duckdb_tmp_loyalty"
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    connection.execute(f"SET memory_limit={sql_literal(args.memory_limit)}")
    connection.execute(f"SET threads={max(1, args.threads)}")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET temp_directory={sql_literal(temp_dir)}")

    android_selected = select_named_columns(args.android)
    ios_selected = select_named_columns(args.ios)
    pbar = tqdm(total=6, desc="Loyalty analysis", unit="phase", disable=args.no_progress)

    def phase(label: str, query: str):
        pbar.set_postfix_str(label)
        connection.execute(query)
        pbar.update(1)

    phase(
        "read Android + iOS CSVs through DuckDB",
        f"""
        CREATE OR REPLACE TEMP TABLE journeys AS
        SELECT 'android' AS platform, {android_selected}
        FROM read_csv_auto({sql_literal(args.android)}, header=true, all_varchar=true, encoding='utf-8')
        UNION ALL BY NAME
        SELECT 'ios' AS platform, {ios_selected}
        FROM read_csv_auto({sql_literal(args.ios)}, header=true, all_varchar=true, encoding='utf-8')
        """,
    )

    evidence = "lower(concat_ws(' ', " + ", ".join(
        f"coalesce({column}, '')"
        for column in (
            "sequence", "cluster_name", "cluster_name_en", "business_family",
            "business_family_code", "business_submodule", "business_detail", "mapping_function_code",
        )
    ) + "))"
    phase(
        "extract Loyalty journeys",
        f"""
        CREATE OR REPLACE TEMP VIEW loyalty_journeys AS
        SELECT *,
               try_cast(coalesce(nullif(end_ts, ''), nullif(start_ts, '')) AS TIMESTAMPTZ) AS journey_end,
               try_cast(start_ts AS TIMESTAMPTZ) AS journey_start
        FROM journeys
        WHERE regexp_matches({evidence}, {sql_literal(LOYALTY_RE)})
        """,
    )

    phase(
        "build customer-level Loyalty cohort",
        """
        CREATE OR REPLACE TEMP VIEW loyalty_anchors AS
        SELECT customer_id,
               min(journey_end) AS signature_anchor_ts,
               arg_min(journey_id, journey_end) AS signature_anchor_journey_id,
               arg_min(platform, journey_end) AS signature_anchor_platform
        FROM loyalty_journeys
        WHERE nullif(trim(customer_id), '') IS NOT NULL AND journey_end IS NOT NULL
        GROUP BY customer_id
        """,
    )
    copy_query(
        connection,
        """
        SELECT customer_id, platform, journey_id AS record_id, start_ts, end_ts,
               cluster, cluster_name, business_family, business_submodule, business_detail,
               mapping_function_code, naming_confidence, assignment_type
        FROM loyalty_journeys
        ORDER BY customer_id, journey_start
        """,
        out / "loyalty_journey_records.csv",
    )
    copy_query(
        connection,
        """
        SELECT customer_id,
               COUNT(*) AS loyalty_journey_count,
               COUNT(DISTINCT platform) AS platform_count,
               string_agg(DISTINCT platform, ';' ORDER BY platform) AS platforms,
               MIN(journey_start) AS first_loyalty_journey_ts,
               MAX(journey_start) AS last_loyalty_journey_ts,
               COUNT(DISTINCT strftime(journey_start, '%Y-%m')) AS active_month_count,
               COUNT(DISTINCT cluster_name) AS distinct_cluster_names
        FROM loyalty_journeys
        WHERE nullif(trim(customer_id), '') IS NOT NULL
        GROUP BY customer_id
        ORDER BY customer_id
        """,
        out / "loyalty_customer_metrics.csv",
    )

    action_filter = """
        nullif(trim(mapping_function_code), '') IS NOT NULL
        AND lower(coalesce(nullif(trim(business_family_code), ''), nullif(trim(business_family), ''))) NOT LIKE 'unclassified%'
        AND lower(coalesce(business_family, '')) NOT IN ('noise', 'chưa phân loại', 'hệ thống kỹ thuật')
        AND lower(coalesce(cluster_name, '')) NOT LIKE 'unclassified%'
    """
    phase(
        "build row-level post-Loyalty actions",
        f"""
        CREATE OR REPLACE TEMP VIEW loyalty_post_actions AS
        SELECT j.customer_id,
               a.signature_anchor_ts,
               a.signature_anchor_journey_id,
               a.signature_anchor_platform,
               j.journey_id AS record_id,
               j.platform,
               j.start_ts AS action_start_ts,
               epoch(j.journey_start - a.signature_anchor_ts) / 86400.0 AS elapsed_days,
               CASE
                 WHEN epoch(j.journey_start - a.signature_anchor_ts) / 86400.0 < 1 THEN 'immediate_0_1d'
                 WHEN epoch(j.journey_start - a.signature_anchor_ts) / 86400.0 < 7 THEN 'after_1d_7d'
                 WHEN epoch(j.journey_start - a.signature_anchor_ts) / 86400.0 < 30 THEN 'after_1w_30d'
                 ELSE 'after_1m_plus'
               END AS time_bucket,
               j.cluster, j.cluster_name, j.business_family, j.business_submodule,
               j.business_detail, j.mapping_function_code, j.naming_confidence, j.assignment_type,
               concat_ws(' | ', nullif(j.business_family, ''), nullif(j.business_submodule, ''), nullif(j.business_detail, '')) AS action_name
        FROM (
          SELECT j.*, try_cast(j.start_ts AS TIMESTAMPTZ) AS journey_start
          FROM journeys j
        ) j
        JOIN loyalty_anchors a USING (customer_id)
        WHERE j.journey_start IS NOT NULL
          AND j.journey_start > a.signature_anchor_ts
          AND j.journey_id <> a.signature_anchor_journey_id
          AND {action_filter}
        """,
    )
    copy_query(
        connection,
        """
        SELECT * FROM loyalty_post_actions
        ORDER BY customer_id, action_start_ts, record_id
        """,
        out / "loyalty_post_action_records.csv",
    )
    copy_query(
        connection,
        """
        WITH active AS (
          SELECT time_bucket, COUNT(DISTINCT customer_id) AS active_customers_in_bucket
          FROM loyalty_post_actions GROUP BY time_bucket
        )
        SELECT p.time_bucket, p.business_family, p.business_submodule, p.mapping_function_code AS function_code,
               COUNT(*) AS record_count, COUNT(DISTINCT p.customer_id) AS unique_customers,
               (SELECT COUNT(*) FROM loyalty_anchors) AS cohort_customers,
               100.0 * COUNT(DISTINCT p.customer_id) / NULLIF((SELECT COUNT(*) FROM loyalty_anchors), 0) AS pct_of_signature_cohort,
               active.active_customers_in_bucket,
               100.0 * COUNT(DISTINCT p.customer_id) / NULLIF(active.active_customers_in_bucket, 0) AS pct_of_active_customers_in_bucket
        FROM loyalty_post_actions p
        JOIN active USING (time_bucket)
        GROUP BY p.time_bucket, p.business_family, p.business_submodule, p.mapping_function_code, active.active_customers_in_bucket
        ORDER BY p.time_bucket, unique_customers DESC, function_code
        """,
        out / "loyalty_post_action_summary.csv",
    )
    pbar.update(2)
    pbar.close()

    counts = connection.execute(
        "SELECT (SELECT COUNT(*) FROM journeys), (SELECT COUNT(*) FROM loyalty_journeys), "
        "(SELECT COUNT(*) FROM loyalty_anchors), (SELECT COUNT(*) FROM loyalty_post_actions)"
    ).fetchone()
    metadata = {
        "schema_version": "loyalty-analysis-v1",
        "input_rows": counts[0],
        "loyalty_journey_records": counts[1],
        "loyalty_customers": counts[2],
        "post_loyalty_action_records": counts[3],
        "inputs": {"android": str(args.android.resolve()), "ios": str(args.ios.resolve())},
        "outputs": [
            "loyalty_journey_records.csv", "loyalty_customer_metrics.csv", "loyalty_cohort.csv",
            "loyalty_post_action_records.csv", "loyalty_post_action_summary.csv", "04_loyalty_raw.md",
        ],
        "rule": LOYALTY_RE,
        "engine": "DuckDB with bounded memory and disk spill",
    }
    copy_query(
        connection,
        "SELECT customer_id, signature_anchor_ts, signature_anchor_journey_id, signature_anchor_platform FROM loyalty_anchors ORDER BY customer_id",
        out / "loyalty_cohort.csv",
    )
    markdown = [
        "# 04 · Loyalty raw output", "",
        "Rule-based cohort: named evidence for Loyalty/voucher/reward/offer/membership/Ưu đãi/Khách hàng thân thiết.",
        "The cohort anchor is the earliest Loyalty journey per `customer_id`; post-Loyalty detail stays one source journey/action per row.", "",
        f"Input rows: **{counts[0]:,}**; Loyalty journeys: **{counts[1]:,}**; Loyalty customers: **{counts[2]:,}**; post-Loyalty action records: **{counts[3]:,}**.", "",
        "Files: `loyalty_journey_records.csv`, `loyalty_customer_metrics.csv`, `loyalty_cohort.csv`, `loyalty_post_action_records.csv`, `loyalty_post_action_summary.csv`.",
    ]
    (out / "04_loyalty_raw.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    connection.close()
    shutil.rmtree(temp_dir, ignore_errors=True)
    return metadata


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
