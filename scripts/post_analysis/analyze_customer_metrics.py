#!/usr/bin/env python3
"""Produce customer-level footprint tables from the all-named inference anchors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.post_analysis.common import create_connection, register_named_journeys  # noqa: E402


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android", type=Path, required=True)
    parser.add_argument("--ios", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def copy(connection, query: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    escaped = str(output.resolve()).replace("'", "''")
    connection.execute(f"COPY ({query}) TO '{escaped}' (FORMAT CSV, HEADER, DELIMITER ',')")


def main() -> int:
    cli = args()
    output = resolve(cli.output_dir)
    db = create_connection(output, cli.memory_limit, cli.threads)
    try:
        register_named_journeys(db, resolve(cli.android), resolve(cli.ios))
        db.execute(
            """
            CREATE OR REPLACE TEMP VIEW valid_customer_journeys AS
            SELECT NULLIF(TRIM(customer_id), '') AS customer_id, platform, journey_start,
                   journey_id, cluster_name, business_family
            FROM named_journeys
            WHERE NULLIF(TRIM(customer_id), '') IS NOT NULL
              AND lower(trim(customer_id)) NOT IN ('anonymous', '__anonymous__', 'nan', 'none')
              AND journey_start IS NOT NULL
            """
        )
        copy(
            db,
            """
            WITH ordered AS (
                SELECT *, lag(journey_start) OVER (PARTITION BY customer_id ORDER BY journey_start, journey_id) AS previous_start
                FROM valid_customer_journeys
            ), gaps AS (
                SELECT customer_id, median(epoch(journey_start - previous_start) / 3600.0) AS median_distance_between_journeys_hours
                FROM ordered WHERE previous_start IS NOT NULL GROUP BY customer_id
            )
            SELECT j.customer_id,
                   COUNT(*) AS journey_count,
                   string_agg(DISTINCT platform, ';' ORDER BY platform) AS platforms,
                   COUNT(DISTINCT platform) AS platform_count,
                   COUNT(DISTINCT strftime(journey_start, '%Y-%m')) AS unique_month_count,
                   MIN(journey_start) AS first_journey_ts,
                   MAX(journey_start) AS last_journey_ts,
                   epoch(MAX(journey_start) - MIN(journey_start)) / 86400.0 AS active_days,
                   g.median_distance_between_journeys_hours,
                   g.median_distance_between_journeys_hours / 24.0 AS median_distance_between_journeys_days,
                   COUNT(*) / NULLIF(COUNT(DISTINCT strftime(journey_start, '%Y-%m')), 0) AS journeys_per_active_month,
                   COUNT(DISTINCT cluster_name) AS distinct_cluster_names,
                   COUNT(DISTINCT business_family) AS distinct_business_families
            FROM valid_customer_journeys j
            LEFT JOIN gaps g USING (customer_id)
            GROUP BY j.customer_id, g.median_distance_between_journeys_hours
            ORDER BY j.customer_id
            """,
            output / "customer_metrics.csv",
        )
        copy(
            db,
            """
            SELECT customer_id, strftime(journey_start, '%Y-%m') AS month,
                   COUNT(*) AS journey_count
            FROM valid_customer_journeys
            GROUP BY customer_id, month ORDER BY customer_id, month
            """,
            output / "customer_month_metrics.csv",
        )
        copy(
            db,
            """
            WITH all_rows AS (
                SELECT 'all' AS scope, 'all' AS platform, strftime(journey_start, '%Y-%m') AS month,
                       customer_id, journey_id FROM valid_customer_journeys
            ), platform_rows AS (
                SELECT 'platform' AS scope, platform, strftime(journey_start, '%Y-%m') AS month,
                       customer_id, journey_id FROM valid_customer_journeys
            ), combined AS (SELECT * FROM all_rows UNION ALL SELECT * FROM platform_rows)
            SELECT scope, platform, month, COUNT(DISTINCT customer_id) AS unique_customers,
                   COUNT(*) AS journey_count
            FROM combined GROUP BY scope, platform, month ORDER BY scope, platform, month
            """,
            output / "customer_month_unique.csv",
        )
    finally:
        db.close()
    print(f"wrote customer metrics -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
