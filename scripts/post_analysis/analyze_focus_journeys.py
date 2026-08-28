#!/usr/bin/env python3
"""Export focus journeys and customer summaries from named inference anchors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.post_analysis.common import create_connection, register_named_journeys, sql_literal  # noqa: E402


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


FOCUS = {
    "vneid_econtract": "vneid|e[-_ ]?contract|econtract",
    "support": "support|hỗ trợ|yeu.?cau.?ho.?tro",
    "payment": "payment|thanh toán|invoice|billing|bill|checkout|vietqr",
    "device_management": "device[_ -]?management|quản lý thiết bị|internet|wifi|modem|camera|smart.?tv",
    "notification": "notification|thông báo",
    "loyalty": "loyalty|voucher|reward|points?|membership|offer|ưu đãi|thân thiết",
}


def main() -> int:
    cli = args()
    output = resolve(cli.output_dir)
    db = create_connection(output, cli.memory_limit, cli.threads)
    try:
        register_named_journeys(db, resolve(cli.android), resolve(cli.ios))
        evidence = "lower(concat_ws(' ', sequence, cluster_name, cluster_name_en, business_family, business_family_code, business_submodule, business_detail, mapping_function_code))"
        focus_union = " UNION ALL ".join(
            f"SELECT *, {sql_literal(name)} AS focus_object FROM named_journeys WHERE regexp_matches({evidence}, {sql_literal(pattern)})"
            for name, pattern in FOCUS.items()
        )
        db.execute(f"CREATE OR REPLACE TEMP VIEW focus_journeys AS {focus_union}")
        outcome = "CASE WHEN regexp_matches(lower(concat_ws(' ', sequence, cluster_name, business_detail)), 'failed?|error|cancel|reject|declin|timeout|unsuccessful') THEN 'failure' WHEN regexp_matches(lower(concat_ws(' ', sequence, cluster_name, business_detail)), 'success|complete|completed') THEN 'success' ELSE 'unknown' END"
        copy(
            db,
            f"""
            SELECT customer_id, journey_id AS record_id, platform, start_ts, end_ts, {outcome} AS outcome,
                   focus_object, cluster, cluster_name, business_family, business_submodule,
                   business_detail, mapping_function_code, naming_confidence, assignment_type
            FROM focus_journeys
            ORDER BY customer_id, start_ts, focus_object
            """,
            output / "focus_journey_records.csv",
        )
        copy(
            db,
            f"""
            SELECT focus_object, {outcome} AS outcome, platform, COUNT(*) AS record_count,
                   COUNT(DISTINCT NULLIF(TRIM(customer_id), '')) AS unique_customers,
                   MIN(start_ts) AS first_record_ts, MAX(start_ts) AS last_record_ts,
                   '' AS median_gap_hours
            FROM focus_journeys
            GROUP BY focus_object, outcome, platform
            ORDER BY focus_object, outcome, platform
            """,
            output / "focus_customer_summary.csv",
        )
        copy(
            db,
            f"""
            SELECT customer_id, focus_object, {outcome} AS outcome, platform, COUNT(*) AS record_count,
                   MIN(start_ts) AS first_record_ts, MAX(start_ts) AS last_record_ts,
                   '' AS median_gap_hours
            FROM focus_journeys
            WHERE NULLIF(TRIM(customer_id), '') IS NOT NULL
            GROUP BY customer_id, focus_object, outcome, platform
            ORDER BY customer_id, focus_object, outcome, platform
            """,
            output / "focus_customer_metrics.csv",
        )
    finally:
        db.close()
    print(f"wrote focus analysis -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
