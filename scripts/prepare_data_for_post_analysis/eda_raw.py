#!/usr/bin/env python3
"""Profile the original event CSVs for the EDA page.

The dashboard does not infer raw-event statistics from prepared journey Parquet
footers.  Run this script explicitly when a full raw-data EDA is wanted.  Use
``--max-rows`` for a quick code/smoke test; without that option every selected
CSV is processed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
USECOLS = ["device_id", "session_id", "customer_id", "platform", "key", "segmentation_name", "client_time"]
QUANTILES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
LONG_NUM_RE = re.compile(r"\b\d{4,}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/giga_data"))
    parser.add_argument("--files", nargs="*", type=Path, help="specific raw CSVs; default discovers *_t*.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("output/scores/pca48_ngrams12_500/post_analysis/eda"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-rows", type=int, help="smoke-test cap per input file; omit for full EDA")
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    return args


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def discover(args: argparse.Namespace) -> list[Path]:
    if args.files:
        paths = [resolve(path) for path in args.files]
    else:
        root = resolve(args.input_root)
        paths = sorted(root.glob("*_t*.csv"))
    if not paths:
        raise SystemExit("no raw CSV files found; pass --files or --input-root")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"input files not found: {', '.join(missing)}")
    return paths


def parse_time(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        # iOS exports use epoch milliseconds; the conditional also handles seconds.
        unit = "ms" if numeric.dropna().median() > 100_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def quantiles(values: list[float] | pd.Series) -> dict[str, float | None]:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return {str(value): None for value in QUANTILES}
    return {str(key): round(float(value), 3) for key, value in series.quantile(QUANTILES).items()}


def histogram(values: list[float] | pd.Series, bins: int, upper: float | None = None) -> dict[str, list[float]]:
    series = pd.Series(values, dtype="float64").dropna()
    if upper is not None:
        series = series.clip(upper=upper)
    if series.empty:
        return {"counts": [], "edges": []}
    counts, edges = np.histogram(series.to_numpy(), bins=bins)
    return {"counts": counts.astype(int).tolist(), "edges": [float(edge) for edge in edges]}


def _reservoir(values: list[float], incoming: pd.Series, limit: int = 100_000) -> None:
    """Keep a bounded gap sample so a full EDA does not retain every event gap."""
    for value in incoming.dropna().astype(float):
        if len(values) < limit:
            values.append(value)
        else:
            index = np.random.randint(0, len(values) + 1)
            if index < len(values):
                values[index] = value


def profile(path: Path, max_rows: int | None, chunksize: int) -> dict:
    platform = "ios" if path.name.lower().startswith("ios") else "android"
    week_match = re.search(r"_(t\d+)(?:[_-]|$)", path.stem.lower())
    week = week_match.group(1).upper() if week_match else "ALL"
    session_state: dict[str, list[object]] = {}
    session_samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    segment_counts: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    devices: set[str] = set()
    customers: set[str] = set()
    gaps: list[float] = []
    action_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()
    total = 0
    first_ts = None
    last_ts = None

    reader = pd.read_csv(path, usecols=lambda column: column in USECOLS, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        if max_rows is not None:
            remaining = max_rows - total
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        if chunk.empty:
            continue
        total += len(chunk)
        chunk["ts"] = parse_time(chunk["client_time"])
        chunk = chunk.dropna(subset=["ts", "session_id"]).copy()
        chunk["session_id"] = chunk["session_id"].astype(str)
        chunk["key"] = chunk["key"].fillna("<missing>").astype(str)
        chunk["segmentation_name"] = chunk["segmentation_name"].fillna("<missing>").astype(str)
        segment_counts.update(chunk["segmentation_name"].value_counts().to_dict())
        key_counts.update(chunk["key"].value_counts().to_dict())
        action_counts.update(chunk.loc[chunk["key"].eq("action"), "segmentation_name"].value_counts().to_dict())
        view_counts.update(chunk.loc[chunk["key"].eq("view"), "segmentation_name"].value_counts().to_dict())
        devices.update(chunk["device_id"].dropna().astype(str))
        customers.update(chunk["customer_id"].dropna().astype(str))
        first_ts = chunk["ts"].min() if first_ts is None else min(first_ts, chunk["ts"].min())
        last_ts = chunk["ts"].max() if last_ts is None else max(last_ts, chunk["ts"].max())

        for session_id, group in chunk.sort_values(["session_id", "ts"], kind="stable").groupby("session_id", sort=False):
            state = session_state.setdefault(session_id, [0, None, None])
            state[0] = int(state[0]) + len(group)
            state[1] = group["ts"].min() if state[1] is None else min(state[1], group["ts"].min())
            state[2] = group["ts"].max() if state[2] is None else max(state[2], group["ts"].max())
            if len(session_samples[session_id]) < 40:
                session_samples[session_id].extend(
                    {
                        "key": row.key,
                        "segmentation_name": row.segmentation_name,
                        "client_time": row.ts.isoformat(),
                    }
                    for row in group.head(40 - len(session_samples[session_id])).itertuples()
                )
            _reservoir(gaps, group["ts"].diff().dt.total_seconds())
        if max_rows is not None and total >= max_rows:
            break

    session_lengths = [int(state[0]) for state in session_state.values()]
    session_spans = [max((state[2] - state[1]).total_seconds(), 0.0) for state in session_state.values() if state[1] is not None and state[2] is not None]
    demo = []
    for session_id, state in session_state.items():
        if 12 <= int(state[0]) <= 40:
            rows = session_samples.get(session_id, [])[:30]
            for index in range(1, len(rows)):
                previous = pd.Timestamp(rows[index - 1]["client_time"])
                current = pd.Timestamp(rows[index]["client_time"])
                rows[index]["gap_s"] = round((current - previous).total_seconds(), 2)
            if rows:
                rows[0]["gap_s"] = None
                demo = rows
                break

    vocabulary = pd.Series(list(segment_counts.keys()), dtype="string")
    depth = vocabulary.str.count("/")
    taxonomy = {
        "vocab_size": int(len(vocabulary)),
        "singletons": int(sum(value == 1 for value in segment_counts.values())),
        "singleton_share": round(float(sum(value == 1 for value in segment_counts.values()) / max(len(vocabulary), 1)), 4),
        "top100_event_coverage": round(float(sum(value for _, value in segment_counts.most_common(100)) / max(total, 1)), 4),
        "top500_event_coverage": round(float(sum(value for _, value in segment_counts.most_common(500)) / max(total, 1)), 4),
        "with_uuid": int(vocabulary.str.contains(UUID_RE).sum()),
        "with_long_number": int(vocabulary.str.contains(LONG_NUM_RE).sum()),
        "with_url": int(vocabulary.str.contains("http", case=False).sum()),
        "with_query_string": int(vocabulary.str.contains(r"\?").sum()),
        "depth_hist": {str(int(key)): int(value) for key, value in Counter(depth.clip(upper=6).tolist()).items()},
    }
    return {
        "file": path.name,
        "platform": platform,
        "week": week,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "rows": total,
        "n_sessions": len(session_state),
        "n_devices": len(devices),
        "n_customers": len(customers),
        "ts_min": first_ts.isoformat() if first_ts is not None else "",
        "ts_max": last_ts.isoformat() if last_ts is not None else "",
        "key_counts": dict(key_counts),
        "action_share": round(key_counts.get("action", 0) / max(total, 1), 4),
        "session_len_q": quantiles(session_lengths),
        "session_span_q": quantiles(session_spans),
        "gap_q": quantiles(gaps),
        "gap_over_90s_share": round(sum(value > 90 for value in gaps) / max(len(gaps), 1), 4),
        "gap_over_1800s_share": round(sum(value > 1800 for value in gaps) / max(len(gaps), 1), 4),
        "sessions_per_customer_q": {},
        "events_per_day": {},
        "hour_hist": {},
        "top_segments": [[key, value] for key, value in segment_counts.most_common(40)],
        "top_action_tokens": [[key, value] for key, value in action_counts.most_common(25)],
        "top_view_tokens": [[key, value] for key, value in view_counts.most_common(25)],
        "taxonomy_signals": taxonomy,
        "session_len_hist": histogram(session_lengths, 40, 400),
        "demo_session": demo,
        "source": "raw event CSV",
        "_vocab": set(segment_counts),
    }


def main() -> int:
    args = parse_args()
    paths = discover(args)
    per_file = {}
    vocab_by_platform: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        print(f"profiling {path.name}{f' (max {args.max_rows:,} rows)' if args.max_rows else ''} ...")
        stats = profile(path, args.max_rows, args.chunksize)
        vocab_by_platform[stats["platform"]].update(stats.pop("_vocab"))
        per_file[path.name] = stats
    overlap = {}
    if {"android", "ios"} <= set(vocab_by_platform):
        android, ios = vocab_by_platform["android"], vocab_by_platform["ios"]
        shared = android & ios
        overlap = {
            "android_only": len(android - ios),
            "ios_only": len(ios - android),
            "shared": len(shared),
            "jaccard": round(len(shared) / max(len(android | ios), 1), 4),
            "shared_examples": sorted(shared)[:15],
        }
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "raw-eda-v1",
        "status": "ready",
        "mode": "smoke_test" if args.max_rows else "full",
        "input_root": str(resolve(args.input_root)),
        "per_file": per_file,
        "platform_vocab_overlap": overlap,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    target = output_dir / "eda_summary.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote raw EDA summary -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
