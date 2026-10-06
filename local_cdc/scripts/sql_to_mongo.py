#!/usr/bin/env python3
"""Turn a SQL dump (or CSV export) of raw events into JSON documents and load
them into MongoDB, so the Debezium source sees them as if the app wrote them.

Supported inputs:
  * ``INSERT INTO t (c1, c2, ...) VALUES (...), (...);`` (MySQL / PostgreSQL /
    SQL Server style quoting)
  * ``COPY t (c1, c2, ...) FROM stdin;`` blocks, the pg_dump default
  * ``.csv`` with a header row

Each row becomes one document:
  * ``_id`` that looks like a 24-hex ObjectId is stored as an ObjectId
  * ISO-8601 strings (``2026-03-26T12:20:14.000Z``) become BSON dates
  * ``--nest-prefix segmentation_`` turns ``segmentation_name`` into
    ``{"segmentation": {"name": ...}}`` (columns with a dot nest too)

Examples:
  python sql_to_mongo.py dump.sql --dry-run 3          # print 3 JSON docs
  python sql_to_mongo.py dump.sql --drop               # load into tracking.events
  python sql_to_mongo.py dump.sql --table events --rebase-to 2026-10-01
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from bson import ObjectId, json_util

OBJECT_ID = re.compile(r"^[0-9a-fA-F]{24}$")
ISO_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
NUMBER = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
INSERT_HEAD = re.compile(
    r"INSERT\s+INTO\s+(?P<table>[^\s(]+)\s*\((?P<cols>[^)]*)\)\s*VALUES\s*",
    re.IGNORECASE,
)
COPY_ESCAPE = re.compile(r"\\(.)")
COPY_ESCAPES = {"t": "\t", "n": "\n", "r": "\r"}
COPY_HEAD = re.compile(
    r"^COPY\s+(?P<table>[^\s(]+)\s*\((?P<cols>[^)]*)\)\s+FROM\s+stdin", re.IGNORECASE
)


def strip_identifier(name: str) -> str:
    name = name.strip()
    if name[:1] in "\"`[" and name[-1:] in "\"`]":
        name = name[1:-1]
    return name


def table_name(raw: str) -> str:
    return strip_identifier(raw.split(".")[-1])


def split_columns(raw: str) -> list[str]:
    return [strip_identifier(c) for c in raw.split(",")]


# --------------------------------------------------------------------------
# SQL parsing
# --------------------------------------------------------------------------

def parse_values(text: str, pos: int) -> tuple[list[list[object]], int]:
    """Parse ``(v, v), (v, v);`` starting at ``pos``; return rows and end."""
    rows: list[list[object]] = []
    n = len(text)
    while pos < n:
        while pos < n and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= n or text[pos] == ";":
            return rows, pos + 1
        if text[pos] != "(":
            raise ValueError(f"expected '(' at offset {pos}: {text[pos:pos + 40]!r}")
        pos += 1
        row: list[object] = []
        while True:
            while text[pos] in " \t\r\n":
                pos += 1
            if text[pos] in "'\"" or text[pos : pos + 2].upper() == "N'":
                if text[pos] in "Nn":
                    pos += 1
                quote = text[pos]
                pos += 1
                out = []
                while True:
                    ch = text[pos]
                    if ch == "\\" and pos + 1 < n:
                        nxt = text[pos + 1]
                        out.append({"n": "\n", "t": "\t", "r": "\r", "0": "\0"}.get(nxt, nxt))
                        pos += 2
                    elif ch == quote and text[pos + 1 : pos + 2] == quote:
                        out.append(quote)
                        pos += 2
                    elif ch == quote:
                        pos += 1
                        break
                    else:
                        out.append(ch)
                        pos += 1
                value: object = "".join(out)
            else:
                end = pos
                while text[end] not in ",)":
                    end += 1
                token = text[pos:end].strip()
                pos = end
                upper = token.upper()
                if upper == "NULL":
                    value = None
                elif upper in ("TRUE", "FALSE"):
                    value = upper == "TRUE"
                elif NUMBER.match(token):
                    value = float(token) if any(c in token for c in ".eE") else int(token)
                else:
                    value = token
            row.append(value)
            while text[pos] in " \t\r\n":
                pos += 1
            if text[pos] == ",":
                pos += 1
                continue
            if text[pos] == ")":
                pos += 1
                break
            raise ValueError(f"unexpected {text[pos]!r} at offset {pos}")
        rows.append(row)
    return rows, pos


def copy_value(token: str) -> object:
    if token == r"\N":
        return None
    return COPY_ESCAPE.sub(lambda m: COPY_ESCAPES.get(m[1], m[1]), token)


def iter_sql_rows(path: Path, only_table: str | None) -> Iterator[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # COPY ... FROM stdin blocks (pg_dump default format).
    i = 0
    while i < len(lines):
        match = COPY_HEAD.match(lines[i])
        i += 1
        if not match:
            continue
        columns = split_columns(match["cols"])
        wanted = only_table is None or table_name(match["table"]) == only_table
        while i < len(lines) and lines[i].rstrip("\r\n") != r"\.":
            if wanted:
                values = [copy_value(v) for v in lines[i].rstrip("\r\n").split("\t")]
                yield dict(zip(columns, values))
            i += 1

    # INSERT INTO ... VALUES statements.
    for match in INSERT_HEAD.finditer(text):
        if only_table is not None and table_name(match["table"]) != only_table:
            continue
        columns = split_columns(match["cols"])
        rows, _ = parse_values(text, match.end())
        for row in rows:
            if len(row) != len(columns):
                raise ValueError(f"{len(row)} values for {len(columns)} columns in {match['table']}")
            yield dict(zip(columns, row))


def iter_rows(path: Path, only_table: str | None) -> Iterator[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield {k: (None if v == "" else v) for k, v in row.items()}
    else:
        yield from iter_sql_rows(path, only_table)


# --------------------------------------------------------------------------
# Row -> document
# --------------------------------------------------------------------------

def parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def to_document(row: dict[str, object], nest_prefixes: list[str], shift: timedelta) -> dict:
    doc: dict = {}
    for column, value in row.items():
        if isinstance(value, str):
            if column == "_id" and OBJECT_ID.match(value):
                value = ObjectId(value)
            elif ISO_TIME.match(value) and (parsed := parse_time(value)) is not None:
                value = parsed + shift
        path = column.split(".")
        for prefix in nest_prefixes:
            if len(path) == 1 and column.startswith(prefix) and len(column) > len(prefix):
                path = [prefix.rstrip("_."), column[len(prefix):]]
                break
        target = doc
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = value
    return doc


def rebase_shift(path: Path, table: str | None, time_column: str, target: date) -> timedelta:
    latest: datetime | None = None
    for row in iter_rows(path, table):
        value = row.get(time_column)
        if isinstance(value, str) and (parsed := parse_time(value)) is not None:
            latest = parsed if latest is None or parsed > latest else latest
    if latest is None:
        raise SystemExit(f"no parsable {time_column!r} values found for --rebase-to")
    return timedelta(days=(target - latest.date()).days)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help=".sql dump or .csv export")
    parser.add_argument("--table", help="only rows of this SQL table (default: all INSERT/COPY rows)")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/?directConnection=true")
    parser.add_argument("--db", default="tracking")
    parser.add_argument("--collection", default="events")
    parser.add_argument("--nest-prefix", action="append", default=[], metavar="PREFIX",
                        help="turn PREFIXfield columns into a sub-document (repeatable)")
    parser.add_argument("--rebase-to", type=date.fromisoformat, metavar="YYYY-MM-DD",
                        help="shift every timestamp so the latest --time-column day becomes this date")
    parser.add_argument("--time-column", default="client_time")
    parser.add_argument("--drop", action="store_true", help="drop the collection before loading")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", type=int, metavar="N", help="print N documents as JSON, load nothing")
    parser.add_argument("--jsonl", type=Path, help="also write every document to this JSON Lines file")
    args = parser.parse_args(argv)

    shift = timedelta(0)
    if args.rebase_to:
        shift = rebase_shift(args.input, args.table, args.time_column, args.rebase_to)
        print(f"shifting timestamps by {shift.days} days", file=sys.stderr)

    docs = (to_document(row, args.nest_prefix, shift) for row in iter_rows(args.input, args.table))

    if args.dry_run is not None:
        for _, doc in zip(range(args.dry_run), docs):
            print(json_util.dumps(doc, indent=2, json_options=json_util.RELAXED_JSON_OPTIONS))
        return 0

    from pymongo import MongoClient

    collection = MongoClient(args.mongo_uri)[args.db][args.collection]
    if args.drop:
        collection.drop()
    jsonl = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    total, batch = 0, []
    for doc in docs:
        if jsonl:
            jsonl.write(json_util.dumps(doc, json_options=json_util.RELAXED_JSON_OPTIONS) + "\n")
        batch.append(doc)
        if len(batch) >= args.batch_size:
            collection.insert_many(batch, ordered=False)
            total += len(batch)
            batch = []
    if batch:
        collection.insert_many(batch, ordered=False)
        total += len(batch)
    if jsonl:
        jsonl.close()
    print(f"inserted {total:,} documents into {args.db}.{args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
