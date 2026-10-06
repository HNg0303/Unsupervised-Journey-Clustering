"""End-to-end check of the CDC path: MongoDB -> Debezium -> Kafka -> JDBC sink -> PostgreSQL.

Inserts one document with a unique session_id, then updates and deletes it,
and after each step waits for the matching row state in PostgreSQL.
Exits non-zero (with the step that timed out) if any change does not arrive.

    pip install pymongo psycopg2-binary
    python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone

import psycopg2
from pymongo import MongoClient


def wait_for(pg, sql: str, params: tuple, ok, timeout: int, step: str):
    start = time.time()
    row = None
    while time.time() - start < timeout:
        with pg.cursor() as cur:
            try:
                cur.execute(sql, params)
                row = cur.fetchone()
            except psycopg2.errors.UndefinedTable:
                row = None
            pg.rollback()
        if ok(row):
            print(f"  ok   {step} ({time.time() - start:.1f}s)  row={row}")
            return row
        time.sleep(1)
    sys.exit(f"  FAIL {step}: last row={row} after {timeout}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mongo", default="mongodb://localhost:27017/?replicaSet=rs0&directConnection=true")
    p.add_argument("--pg", default="postgresql://cdc:cdc@localhost:5432/analytics")
    p.add_argument("--table", default="public.events")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    coll = MongoClient(args.mongo, serverSelectionTimeoutMS=5000).tracking.events
    pg = psycopg2.connect(args.pg)
    sid = f"smoke-{uuid.uuid4().hex[:8]}"
    sql = f"select segmentation_name, __op, __deleted from {args.table} where session_id = %s"
    print(f"session_id={sid}")

    doc_id = coll.insert_one({
        "session_id": sid,
        "key": "action",
        "segmentation": {"name": "smoke_insert"},
        "client_time": datetime.now(timezone.utc),
    }).inserted_id
    wait_for(pg, sql, (sid,), lambda r: r is not None and r[0] == "smoke_insert",
             args.timeout, "insert")

    coll.update_one({"_id": doc_id}, {"$set": {"segmentation.name": "smoke_update"}})
    wait_for(pg, sql, (sid,), lambda r: r is not None and r[0] == "smoke_update" and r[1] == "u",
             args.timeout, "update")

    coll.delete_one({"_id": doc_id})
    wait_for(pg, sql, (sid,), lambda r: r is not None and str(r[2]).lower() == "true",
             args.timeout, "delete (soft, __deleted=true)")

    print("smoke test passed")


if __name__ == "__main__":
    main()
