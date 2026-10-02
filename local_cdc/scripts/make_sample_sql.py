#!/usr/bin/env python3
"""Write a small SQL dump of raw clickstream events for the local simulation.

The rows come from the Android test fixture that already ships with the repo
(mobile/android/app/src/main/assets/test_data_android.csv), so the dump has the
same columns the model reads. Replace it with the real dump when you have it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "mobile/android/app/src/main/assets/test_data_android.csv"


def literal(value: str) -> str:
    return "NULL" if value == "" else "'" + value.replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "sample/raw_events.sql")
    parser.add_argument("--rows-per-insert", type=int, default=500)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        rows = list(reader)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write(f"-- {len(rows)} raw events generated from {args.csv.name}\n")
        out.write("CREATE TABLE raw_events (\n")
        out.write(",\n".join(f"  {c} TEXT" for c in columns) + "\n);\n\n")
        for start in range(0, len(rows), args.rows_per_insert):
            chunk = rows[start : start + args.rows_per_insert]
            out.write(f"INSERT INTO raw_events ({', '.join(columns)}) VALUES\n")
            out.write(",\n".join("(" + ", ".join(literal(v) for v in row) + ")" for row in chunk))
            out.write(";\n\n")
    print(f"wrote {len(rows):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
