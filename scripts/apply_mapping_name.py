#!/usr/bin/env python3
"""Apply the authoritative business naming CSV to scored journey output.

This is the only naming step used after full-data inference.  The input is the
row-level score output (CSV, Parquet, or a directory of Parquet partitions) and
the mapping is the manually reviewed ``Cluster_naming.csv`` keyed by
``(platform, cluster_id)``.  The output is the ``*_all_named.csv`` anchor used
by every post-analysis script.

Examples
--------
    python scripts/apply_mapping_name.py \
      --input output/scores/.../android/model_version=latest/platform=android \
      --platform android \
      --mapping output/scores/.../Cluster_naming.csv \
      --output output/scores/.../inspection/android_all_named.csv

    python scripts/apply_mapping_name.py \
      --input output/scores/.../android_scored.csv \
      --platform android \
      --mapping output/scores/.../Cluster_naming.csv \
      --output output/scores/.../android_all_named.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DERIVED_COLUMNS = (
    "cluster_id",
    "cluster_name",
    "cluster_name_vi",
    "cluster_name_en",
    "business_family",
    "business_family_vi",
    "business_family_code",
    "business_submodule",
    "business_detail",
    "naming_confidence",
    "mapping_function_code",
    "mapping_evidence_share",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="scored CSV, scored Parquet, or a Parquet directory")
    parser.add_argument("--mapping", required=True, type=Path, help="authoritative Cluster_naming.csv")
    parser.add_argument("--output", required=True, type=Path, help="combined named CSV output")
    parser.add_argument("--platform", choices=["android", "ios"], help="required for files without a platform column")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-files", type=int, help="smoke-test limit for partition directories")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")
    return args


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_mapping(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"mapping not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"platform", "cluster_id"}
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        raise SystemExit(f"mapping is missing required columns: {', '.join(missing)}")

    table: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        platform = str(row.get("platform", "")).strip().lower()
        if platform not in {"android", "ios"}:
            continue
        try:
            cluster = int(float(row.get("cluster_id", row.get("cluster", -1))))
        except (TypeError, ValueError):
            continue
        table[(platform, cluster)] = row
    for platform in ("android", "ios"):
        if (platform, -1) not in table:
            raise SystemExit(f"mapping must define ({platform}, -1) for unresolved journeys")
    return table


def input_parts(path: Path, max_files: int | None) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".csv", ".parquet"}:
            raise SystemExit(f"input must be CSV or Parquet: {path}")
        return [path]
    if not path.is_dir():
        raise SystemExit(f"input not found: {path}")
    files = sorted(path.rglob("*.parquet")) + sorted(path.rglob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"no CSV/Parquet files found below {path}")
    return files


def read_parts(parts: Iterable[Path], chunksize: int) -> Iterator[pd.DataFrame]:
    for path in parts:
        if path.suffix.lower() == ".csv":
            yield from pd.read_csv(path, chunksize=chunksize, encoding="utf-8-sig", low_memory=False)
        else:
            yield pd.read_parquet(path)


def _text(row: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return default


def apply_mapping(frame: pd.DataFrame, mapping: dict[tuple[str, int], dict[str, str]], platform: str | None) -> pd.DataFrame:
    if "cluster" not in frame.columns and "cluster_id" not in frame.columns:
        raise SystemExit("score output must contain cluster or cluster_id")

    out = frame.drop(columns=[c for c in DERIVED_COLUMNS if c in frame.columns], errors="ignore").copy()
    raw_cluster = out["cluster"] if "cluster" in out.columns else out["cluster_id"]
    out["cluster"] = pd.to_numeric(raw_cluster, errors="coerce").fillna(-1).astype("int64")

    if "platform" in out.columns:
        platforms = out["platform"].fillna(platform or "").astype(str).str.lower()
    elif platform:
        platforms = pd.Series(platform, index=out.index)
        out["platform"] = platforms.to_numpy()
    else:
        raise SystemExit("input has no platform column; pass --platform")

    records = []
    for source_platform, cluster in zip(platforms, out["cluster"]):
        key = (str(source_platform), int(cluster))
        row = mapping.get(key) or mapping.get((str(source_platform), -1))
        if row is None:
            row = {"platform": source_platform, "cluster_id": "-1", "cluster_name": "Unclassified / mixed journeys"}
        code = _text(row, "canonical_level_2_code", "function_code", "cluster_name_en", default="unclassified.dynamic")
        family_code = code.split(".", 1)[0] if code else "unclassified"
        family_vi = _text(row, "business_family_vi", "business_family", default="Chưa phân loại")
        submodule = _text(row, "cluster_name_level_2", "business_submodule")
        records.append(
            {
                "cluster_id": int(cluster),
                "cluster_name": _text(row, "cluster_name", "mapping_name", default="Unclassified / mixed journeys"),
                "cluster_name_vi": _text(row, "cluster_name_vi", "cluster_name", "mapping_name", default="Chưa phân loại"),
                "cluster_name_en": code,
                "business_family": family_vi,
                "business_family_vi": family_vi,
                "business_family_code": family_code,
                "business_submodule": submodule,
                "business_detail": _text(row, "business_detail"),
                "naming_confidence": _text(row, "naming_confidence", "confidence", default="not_applicable"),
                "mapping_function_code": code,
                "mapping_evidence_share": _text(row, "journey_share", "evidence_share"),
            }
        )

    names = pd.DataFrame(records, index=out.index)
    insert_at = out.columns.get_loc("cluster") + 1
    for column in reversed(list(names.columns)):
        out.insert(insert_at, column, names[column].to_numpy())
    return out


def main() -> int:
    args = parse_args()
    source = resolve(args.input)
    mapping_path = resolve(args.mapping)
    output = resolve(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output}; pass --overwrite or choose another path")
    if output in input_parts(source, args.max_files):
        raise SystemExit("output must not overwrite one of the input partitions")

    mapping = load_mapping(mapping_path)
    parts = input_parts(source, args.max_files)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    total = 0
    platforms: set[str] = set()
    first = True
    try:
        for frame in read_parts(parts, args.chunksize):
            named = apply_mapping(frame, mapping, args.platform)
            platforms.update(named["platform"].dropna().astype(str).str.lower().unique())
            # Write the BOM only once. Repeating it on every appended chunk
            # creates embedded ``\\ufeff`` characters in downstream headers.
            named.to_csv(
                temporary,
                mode="w" if first else "a",
                header=first,
                index=False,
                encoding="utf-8-sig" if first else "utf-8",
            )
            total += len(named)
            first = False
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema_version": "all-named-csv-v1",
        "input": str(source),
        "mapping": str(mapping_path),
        "output": str(output),
        "platforms": sorted(platforms),
        "partitions": [str(path) for path in parts],
        "rows": total,
        "anchor": "all_named.csv is the source for post-analysis and dashboard preparation",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mapped {total:,} scored journeys -> {output}")
    print(f"mapping anchor manifest -> {output.with_suffix('.manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
