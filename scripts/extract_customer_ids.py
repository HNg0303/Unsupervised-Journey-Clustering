"""Extract clean, unique customer_ids from the raw monthly event logs.

Reads ``{android,ios}_log_t{6,7,8}.csv`` from the raw data directory and
normalizes ``customer_id`` so that float-formatted values like ``678.0`` and
plain ``678`` collapse into the same id ``678``. Empty, null, zero and
non-numeric values are dropped as junk.

Outputs (in ``--output-dir``):
- ``customer_ids.csv``: one row per customer with event counts per platform/month
- ``customer_ids_rejected.csv``: raw junk values that were dropped, with counts

Example:
    python scripts/extract_customer_ids.py \
        --input-dir data/giga_data/678 --output-dir output/customer_ids
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

PLATFORMS = ("android", "ios")
MONTHS = (6, 7, 8)

# Digits, optionally followed by ".0", ".00", ... (float-formatted integer).
VALID_ID_PATTERN = r"^\d+(\.0+)?$"


def discover_files(input_dir: Path) -> list[tuple[str, int, Path]]:
    files = []
    for platform in PLATFORMS:
        for month in MONTHS:
            path = input_dir / f"{platform}_log_t{month}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")
            files.append((platform, month, path))
    return files


def build_source_sql(files: list[tuple[str, int, Path]]) -> str:
    parts = [
        f"""
        SELECT
            '{platform}' AS platform,
            {month} AS month,
            trim(customer_id) AS raw_id
        FROM read_csv('{path.as_posix()}', header = true, all_varchar = true, delim = ',', quote = '"', escape = '"', strict_mode = false)
        """
        for platform, month, path in files
    ]
    return "\nUNION ALL\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=Path("data/giga_data/678"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/customer_ids"))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()

    files = discover_files(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    if args.threads:
        con.execute(f"SET threads = {args.threads}")

    started = time.time()
    con.execute(
        f"""
        CREATE TEMP TABLE id_counts AS
        SELECT platform, month, raw_id, count(*) AS n_events
        FROM ({build_source_sql(files)})
        GROUP BY ALL
        """
    )
    # Normalize: strip trailing ".0...", then leading zeros; "0" counts as junk.
    con.execute(
        f"""
        CREATE TEMP TABLE id_normalized AS
        SELECT
            *,
            CASE
                WHEN regexp_matches(raw_id, '{VALID_ID_PATTERN}')
                THEN nullif(ltrim(regexp_replace(raw_id, '\\.0+$', ''), '0'), '')
            END AS customer_id
        FROM id_counts
        """
    )

    customers_path = args.output_dir / "customer_ids.csv"
    con.execute(
        f"""
        COPY (
            SELECT
                customer_id,
                sum(n_events) AS n_events,
                sum(n_events) FILTER (WHERE platform = 'android') AS android_events,
                sum(n_events) FILTER (WHERE platform = 'ios') AS ios_events,
                {", ".join(f"sum(n_events) FILTER (WHERE month = {m}) AS t{m}_events" for m in MONTHS)},
                string_agg(DISTINCT platform, '|' ORDER BY platform) AS platforms,
                string_agg(DISTINCT month::VARCHAR, '|' ORDER BY month::VARCHAR) AS months
            FROM id_normalized
            WHERE customer_id IS NOT NULL
            GROUP BY customer_id
            ORDER BY customer_id::HUGEINT
        ) TO '{customers_path.as_posix()}' (HEADER, DELIMITER ',')
        """
    )

    rejected_path = args.output_dir / "customer_ids_rejected.csv"
    con.execute(
        f"""
        COPY (
            SELECT coalesce(raw_id, '<NULL>') AS raw_id, platform, month, sum(n_events) AS n_events
            FROM id_normalized
            WHERE customer_id IS NULL
            GROUP BY ALL
            ORDER BY n_events DESC
        ) TO '{rejected_path.as_posix()}' (HEADER, DELIMITER ',')
        """
    )

    summary = con.sql(
        """
        SELECT
            platform,
            month,
            sum(n_events) AS total_events,
            sum(n_events) FILTER (WHERE customer_id IS NULL) AS junk_events,
            count(DISTINCT customer_id) AS unique_customers
        FROM id_normalized
        GROUP BY ALL
        ORDER BY platform, month
        """
    )
    print(summary)

    total_unique, cross_platform = con.execute(
        """
        SELECT count(*), count(*) FILTER (WHERE n_platforms = 2)
        FROM (
            SELECT customer_id, count(DISTINCT platform) AS n_platforms
            FROM id_normalized
            WHERE customer_id IS NOT NULL
            GROUP BY customer_id
        )
        """
    ).fetchone()
    print(f"Unique customer_ids (all platforms, t{'/'.join(map(str, MONTHS))}): {total_unique:,}")
    print(f"  on both android and ios: {cross_platform:,}")
    print(f"Wrote {customers_path}")
    print(f"Wrote {rejected_path}")
    print(f"Done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
