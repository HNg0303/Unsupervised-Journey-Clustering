#!/usr/bin/env python3
"""Export one day of raw events from the PostgreSQL sink table to CSV.

This is the piece the daily Airflow job runs. The output CSV has the same
columns as the manual exports, so it can be passed straight to
``scripts/run_full_pipeline.sh`` as ``RAW_INPUT``.

The window is ``[day - lookback_days, day + 1)`` on the event time column, so
events that reached the sink late are still picked up the next day. Rows are
de-duplicated on ``_id`` (Debezium delivers at least once) and rows the sink
marked deleted (``__deleted``) are skipped. Debezium bookkeeping columns
(``__op``, ``__source_ts_ms``, ...) and the sink key ``id`` are not exported.

Example:
  python extract_daily.py --dsn postgresql://cdc:cdc@localhost:5432/analytics \
      --day 2026-03-26 --output-root ../exports
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import psycopg2
from psycopg2 import sql


def table_columns(cur, schema: str, table: str) -> list[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (schema, table),
    )
    columns = [row[0] for row in cur.fetchall()]
    if not columns:
        raise SystemExit(f"table {schema}.{table} not found (has the sink written anything yet?)")
    return columns


def extract_day(
    dsn: str,
    day: date,
    output_root: Path,
    table: str = "public.events",
    time_column: str = "client_time",
    id_column: str = "_id",
    lookback_days: int = 1,
) -> Path:
    schema, _, name = table.rpartition(".")
    schema = schema or "public"
    start, end = day - timedelta(days=lookback_days), day + timedelta(days=1)
    target = output_root / day.isoformat() / "events.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        columns = table_columns(cur, schema, name)
        # "id" is the sink's primary key (copied from the Kafka record key);
        # it duplicates _id, so only the document's own fields are exported.
        exported = [c for c in columns if not c.startswith("__") and not (c == "id" and id_column in columns)]
        where = [sql.SQL("{t} >= %s AND {t} < %s").format(t=sql.Identifier(time_column))]
        if "__deleted" in columns:
            where.append(sql.SQL("COALESCE({d}::text, 'false') <> 'true'").format(d=sql.Identifier("__deleted")))
        order = [sql.Identifier(id_column)]
        if "__source_ts_ms" in columns:
            order.append(sql.SQL("{} DESC").format(sql.Identifier("__source_ts_ms")))
        query = sql.SQL(
            "SELECT DISTINCT ON ({id}) {cols} FROM {table} WHERE {where} ORDER BY {order}"
        ).format(
            id=sql.Identifier(id_column),
            cols=sql.SQL(", ").join(map(sql.Identifier, exported)),
            table=sql.Identifier(schema, name),
            where=sql.SQL(" AND ").join(where),
            order=sql.SQL(", ").join(order),
        )
        copy = sql.SQL("COPY ({}) TO STDOUT WITH CSV HEADER").format(
            sql.SQL(cur.mogrify(query, (start, end)).decode())
        )
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            cur.copy_expert(copy.as_string(conn), handle)
        rows = max(sum(1 for _ in tmp.open(encoding="utf-8")) - 1, 0)

    tmp.replace(target)
    print(f"{schema}.{name} {time_column} in [{start}, {end}): {rows:,} rows -> {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default="postgresql://cdc:cdc@localhost:5432/analytics")
    parser.add_argument("--day", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    parser.add_argument("--output-root", type=Path, default=Path("exports"))
    parser.add_argument("--table", default="public.events")
    parser.add_argument("--time-column", default="client_time")
    parser.add_argument("--id-column", default="_id")
    parser.add_argument("--lookback-days", type=int, default=1)
    args = parser.parse_args(argv)
    extract_day(args.dsn, args.day, args.output_root, args.table, args.time_column, args.id_column, args.lookback_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
