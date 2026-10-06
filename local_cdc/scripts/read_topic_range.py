#!/usr/bin/env python3
"""Read Debezium change events straight from a Kafka topic for a time range.

Kafka has no ``SELECT ... WHERE date BETWEEN``: a topic is an append-only log
per partition, addressed by offset. The only time index Kafka keeps is the
record timestamp (when the change was written to Kafka), so there are two ways
to ask for "a day / a week / a month":

  --time kafka   Seek every partition to the first offset at or after --from
                 (consumer.offsets_for_times) and read until --to. Fast, but it
                 is capture time: the initial snapshot puts *all* historical
                 documents at the moment the connector started.
  --time event   Read every retained record and keep those whose document field
                 (--event-field, default client_time) is inside the range.
                 Correct event-time semantics, but it scans the whole topic.

Either way only data still inside the topic retention can be read
(7 days in docker-compose.yml), so a month needs retention of at least a month.

Each change event is unwrapped to the document after the change, nested fields
are flattened with "_" (segmentation.name -> segmentation_name), the last
change per _id wins and deleted documents are dropped.

Examples:
  python read_topic_range.py --from 2026-10-01 --to 2026-10-02 --time event
  python read_topic_range.py --last 7d --time event --output week.csv
  python read_topic_range.py --last 1h --time kafka        # what arrived in the last hour
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bson import ObjectId, json_util
from confluent_kafka import Consumer, TopicPartition

DURATION = re.compile(r"^(\d+)([hdwm])$")
UNITS = {"h": timedelta(hours=1), "d": timedelta(days=1), "w": timedelta(weeks=1), "m": timedelta(days=30)}


def parse_when(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def flatten(doc: dict, prefix: str = "") -> dict:
    out: dict = {}
    for key, value in doc.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}_"))
        elif isinstance(value, ObjectId):
            out[name] = str(value)
        elif isinstance(value, datetime):
            out[name] = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        else:
            out[name] = value
    return out


def read_range(
    bootstrap: str,
    topic: str,
    start: datetime,
    end: datetime,
    time_mode: str,
    event_field: str,
) -> tuple[dict[str, dict], dict[str, int]]:
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": "read-topic-range",  # never commits: this is an ad-hoc reader
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    stats = {"read": 0, "kept": 0, "deleted": 0}
    try:
        meta = consumer.list_topics(topic, timeout=10)
        if topic not in meta.topics or meta.topics[topic].error:
            raise SystemExit(f"topic {topic!r} not found")
        partitions = [TopicPartition(topic, p) for p in meta.topics[topic].partitions]

        # Where to stop: the end of each partition as of now.
        stop = {tp.partition: consumer.get_watermark_offsets(tp, timeout=10)[1] for tp in partitions}

        if time_mode == "kafka":
            # Kafka's time index: first offset whose record timestamp >= start.
            query = [TopicPartition(topic, tp.partition, int(start.timestamp() * 1000)) for tp in partitions]
            starts = consumer.offsets_for_times(query, timeout=10)
            for tp in starts:
                if tp.offset < 0:  # no record at or after start in this partition
                    tp.offset = stop[tp.partition]
        else:
            starts = [TopicPartition(topic, tp.partition, consumer.get_watermark_offsets(tp, timeout=10)[0])
                      for tp in partitions]
        pending = {tp.partition for tp in starts if tp.offset < stop[tp.partition]}
        consumer.assign(starts)

        docs: dict[str, dict] = {}
        end_ms = int(end.timestamp() * 1000)
        while pending:
            msg = consumer.poll(5)
            if msg is None:
                break
            if msg.error():
                raise SystemExit(str(msg.error()))
            partition, offset = msg.partition(), msg.offset()
            if offset >= stop[partition]:  # arrived after we started reading
                pending.discard(partition)
                continue
            if offset + 1 >= stop[partition]:
                pending.discard(partition)
            stats["read"] += 1
            if time_mode == "kafka" and msg.timestamp()[1] >= end_ms:
                pending.discard(partition)
                continue
            if msg.value() is None:  # tombstone that follows a delete
                continue
            payload = json.loads(msg.value())["payload"]
            key = json.loads(msg.key())["payload"]["id"]
            if payload["op"] == "d":
                if docs.pop(key, None) is not None:
                    stats["deleted"] += 1
                continue
            if not payload.get("after"):
                continue
            doc = flatten(json_util.loads(payload["after"]))
            if time_mode == "event":
                when = doc.get(event_field)
                if not isinstance(when, datetime) or not (start <= when < end):
                    docs.pop(key, None)  # an update may have moved it out of range
                    continue
            doc["__op"] = payload["op"]
            doc["__kafka_ts"] = datetime.fromtimestamp(msg.timestamp()[1] / 1000, timezone.utc)
            docs[key] = doc
        stats["kept"] = len(docs)
        return docs, stats
    finally:
        consumer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", default="localhost:29092")
    parser.add_argument("--topic", default="app.tracking.events")
    parser.add_argument("--from", dest="start", type=parse_when, help="inclusive, e.g. 2026-10-01")
    parser.add_argument("--to", dest="end", type=parse_when, help="exclusive, e.g. 2026-10-02")
    parser.add_argument("--last", help="instead of --from/--to: 1h, 1d, 7d, 1w, 1m (=30d) up to now")
    parser.add_argument("--time", choices=["kafka", "event"], default="event")
    parser.add_argument("--event-field", default="client_time")
    parser.add_argument("--output", type=Path, help="CSV file; default prints a summary only")
    args = parser.parse_args(argv)

    if args.last:
        match = DURATION.match(args.last)
        if not match:
            parser.error("--last must look like 1h, 1d, 7d, 1w or 1m")
        end = datetime.now(timezone.utc)
        start = end - int(match[1]) * UNITS[match[2]]
    elif args.start and args.end:
        start, end = args.start, args.end
    else:
        parser.error("give --from and --to, or --last")

    docs, stats = read_range(args.bootstrap, args.topic, start, end, args.time, args.event_field)
    print(f"{args.topic} {args.time}-time [{start:%Y-%m-%d %H:%M}, {end:%Y-%m-%d %H:%M}) UTC: "
          f"read {stats['read']:,} events, {stats['kept']:,} documents", file=sys.stderr)
    if args.output and docs:
        columns = list(dict.fromkeys(c for doc in docs.values() for c in doc))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for doc in docs.values():
                writer.writerow({k: v.isoformat() if isinstance(v, datetime) else v for k, v in doc.items()})
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
