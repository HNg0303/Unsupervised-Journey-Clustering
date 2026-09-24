#!/usr/bin/env python3
"""Extract every distinct, non-empty customer_id from raw parquet data."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/lake/raw_events/678/678_20260922_105837"
DEFAULT_OUTPUT = REPO_ROOT / "output/customer_ids/678_20260922_105837/customer_ids.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = parse_args()
    input_root = resolve(args.input)
    output_path = resolve(args.output)
    parquet_files = sorted(input_root.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files found below {input_root}")

    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit(
            "duckdb is required; install project requirements first"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    parquet_glob = str(input_root / "**/*.parquet").replace("'", "''")
    temporary_sql = str(temporary).replace("'", "''")

    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            f"""
            COPY (
                WITH normalized AS (
                    SELECT DISTINCT
                        CASE
                            WHEN regexp_full_match(
                                trim(cast(customer_id AS VARCHAR)), '[0-9]+\\.0'
                            )
                            THEN regexp_replace(
                                trim(cast(customer_id AS VARCHAR)), '\\.0$', ''
                            )
                            ELSE trim(cast(customer_id AS VARCHAR))
                        END AS customer_id
                    FROM read_parquet('{parquet_glob}', union_by_name=true)
                )
                SELECT customer_id
                FROM normalized
                WHERE customer_id IS NOT NULL
                  AND lower(customer_id) NOT IN ('', 'nan', 'none', 'null', '<na>')
                ORDER BY customer_id
            ) TO '{temporary_sql}' (HEADER, DELIMITER ',')
            """
        )
        customer_count = connection.execute(
            "SELECT count(*) FROM read_csv_auto(?)", [str(temporary)]
        ).fetchone()[0]
        temporary.replace(output_path)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)

    print(f"Distinct customer_id: {customer_count:,}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
