"""Partition large production CSV exports into session-safe parquet files.

Examples:

    python scripts/partition_raw_events.py \
        --input data/giga_data \
        --output data/lake/raw_events \
        --buckets 128

The output is partitioned by normalized platform and a stable hash of
``(platform, session_id)``.  All rows for one logical session therefore land
in one parquet file, even when the source CSV is read in many chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.large_data import require_pyarrow, stable_session_bucket  # noqa: E402
from src.production import normalize_platform, normalize_production_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/giga_data/678"),
        help="CSV file or directory containing CSV files (default: data/giga_data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lake/raw_events/678"),
        help="new parquet dataset directory (default: data/lake/raw_events)",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--buckets", type=int, default=128, help="stable session buckets per platform")
    parser.add_argument("--platforms", nargs="+", choices=["android", "ios"], help="optional platform filter")
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    if args.buckets < 1:
        parser.error("--buckets must be positive")
    return args


def input_csvs(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() != ".csv":
            raise ValueError(f"input must be a CSV file or directory: {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted(
        path
        for path in root.rglob("*.csv")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    )
    if not paths:
        raise FileNotFoundError(f"no CSV files found below {root}")
    return paths


def union_columns(paths: list[Path]) -> list[str]:
    columns: list[str] = []
    known: set[str] = set()
    for path in paths:
        header = normalize_production_columns(pd.read_csv(path, nrows=0)).columns.tolist()
        for name in header:
            if name not in known:
                known.add(name)
                columns.append(name)
    return columns


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    source_root = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    if output.exists() and any(output.rglob("*.parquet")):
        raise SystemExit(
            f"output already contains parquet files: {output}; choose a new output "
            "directory to keep the run recoverable"
        )
    output.mkdir(parents=True, exist_ok=True)

    pa, _, pq = require_pyarrow()
    paths = input_csvs(source_root)
    source_columns = union_columns(paths)
    allowed = {value.lower() for value in args.platforms} if args.platforms else None

    writers: dict[Path, object] = {}
    schemas: dict[Path, object] = {}
    partition_rows: Counter[str] = Counter()
    source_reports: list[dict[str, object]] = []
    total_rows = 0

    try:
        for source_path in paths:
            relative_source = str(source_path.relative_to(source_root)) if source_root.is_dir() else source_path.name
            offset = 0
            source_rows = 0
            for chunk in pd.read_csv(source_path, chunksize=args.chunksize, low_memory=False):
                raw_chunk_len = len(chunk)
                source_rows += raw_chunk_len
                total_rows += raw_chunk_len
                chunk = normalize_production_columns(chunk)
                if "session_id" not in chunk.columns:
                    raise ValueError(f"{source_path} is missing session_id")
                platform_column = "platform" if "platform" in chunk.columns else "segmentation.segment"
                if platform_column not in chunk.columns:
                    raise ValueError(f"{source_path} is missing platform/segmentation.segment")
                source_row_numbers = pd.Series(
                    range(offset, offset + raw_chunk_len), index=chunk.index, dtype="int64"
                )

                normalized_platform = normalize_platform(chunk[platform_column])
                if allowed is not None:
                    keep = normalized_platform.isin(allowed)
                    chunk = chunk.loc[keep].copy()
                    normalized_platform = normalized_platform.loc[keep]
                if chunk.empty:
                    offset += raw_chunk_len
                    continue

                sessions = chunk["session_id"].astype("string").fillna("<missing>")
                buckets = [
                    stable_session_bucket(platform, session, args.buckets)
                    for platform, session in zip(normalized_platform, sessions)
                ]
                chunk["_source_file"] = relative_source
                chunk["_source_row_number"] = source_row_numbers.loc[chunk.index]
                chunk["_partition_platform"] = normalized_platform.astype("string").tolist()
                chunk["_session_bucket"] = buckets

                # Strings keep a stable parquet schema across CSV chunks while
                # preserving the raw values. The existing canonicalizer parses
                # timestamps and numeric-looking fields later.
                for column in source_columns:
                    if column not in chunk.columns:
                        chunk[column] = pd.NA
                ordered = source_columns + [
                    "_source_file", "_source_row_number", "_partition_platform", "_session_bucket"
                ]
                for column in source_columns + ["_source_file", "_partition_platform"]:
                    chunk[column] = chunk[column].astype("string")
                chunk["_source_row_number"] = pd.Series(
                    chunk["_source_row_number"].to_numpy(dtype="int64"), index=chunk.index, dtype="int64"
                )
                chunk["_session_bucket"] = pd.Series(
                    chunk["_session_bucket"].to_numpy(dtype="int64"), index=chunk.index, dtype="int64"
                )

                for (platform, bucket), part in chunk.groupby(
                    ["_partition_platform", "_session_bucket"], sort=True
                ):
                    platform = str(platform)
                    bucket = int(bucket)
                    partition_key = f"platform={platform}/bucket={bucket:03d}"
                    partition_path = output / partition_key / "part-00000.parquet"
                    table = pa.Table.from_pandas(
                        part[ordered].reset_index(drop=True), preserve_index=False
                    )
                    if partition_path not in writers:
                        partition_path.parent.mkdir(parents=True, exist_ok=True)
                        schemas[partition_path] = table.schema
                        writers[partition_path] = pq.ParquetWriter(
                            partition_path, table.schema, compression="zstd"
                        )
                    table = pa.Table.from_pandas(
                        part[ordered].reset_index(drop=True),
                        schema=schemas[partition_path],
                        preserve_index=False,
                        safe=False,
                    )
                    writers[partition_path].write_table(table)
                    partition_rows[partition_key] += len(part)
                offset += raw_chunk_len
            source_reports.append({"source_file": relative_source, "rows_read": source_rows})
            print(f"partitioned {relative_source}: {source_rows:,} rows")
    finally:
        for writer in writers.values():
            writer.close()

    manifest = {
        "schema_version": "raw-events-parquet-v1",
        "source_root": str(source_root),
        "output_root": str(output),
        "buckets": args.buckets,
        "chunksize": args.chunksize,
        "platforms": sorted(allowed) if allowed else ["android", "ios", "unknown"],
        "source_reports": source_reports,
        "rows_read": total_rows,
        "partition_rows": dict(sorted(partition_rows.items())),
        "partition_count": len(partition_rows),
    }
    (output / "partition_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {len(partition_rows):,} session-safe parquet partitions to {output}")
    print(f"rows read: {total_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
