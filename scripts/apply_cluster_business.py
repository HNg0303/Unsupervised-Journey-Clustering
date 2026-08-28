#!/usr/bin/env python3
"""Apply cluster_mapping.csv to a raw inference CSV/Parquet and catalog.

The join key is ``(platform, cluster_id)``.  Unknown non-noise cluster ids are
errors by default so a model/mapping version mismatch cannot silently receive
the noise label.

Example
-------
python scripts/apply_cluster_business.py \
  --input raw_android_inference.parquet \
  --platform android \
  --mapping output/scores/pca48_ngrams12_500/cluster_mapping.csv \
  --output android_inference_named.csv
  --catalog-output

This also creates ``shareholder_cluster_catalog.json`` next to the named CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAPPED_COLUMNS = (
    "cluster_name",
    "business_family",
    "business_submodule",
    "business_detail",
    "canonical_level_2_code",
    "naming_confidence",
)
REQUIRED_MAPPING_COLUMNS = {"platform", "cluster_id", *MAPPED_COLUMNS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="raw inference CSV or Parquet")
    parser.add_argument("--platform", required=True, choices=("android", "ios"))
    parser.add_argument("--mapping", required=True, type=Path, help="cluster_mapping.csv")
    parser.add_argument("--output", required=True, type=Path, help="named inference CSV")
    parser.add_argument(
        "--catalog-output",
        type=Path,
        help="default: shareholder_cluster_catalog.json next to --output",
    )
    parser.add_argument("--cluster-column", choices=("cluster", "cluster_id"), help="auto-detected by default")
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=50_000,
        help="rows read per Parquet batch (default: 50000)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _cluster_id(value: object, *, context: str) -> int:
    try:
        number = float(str(value).strip())
        cluster = int(number)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid cluster id {value!r} at {context}") from exc
    if number != cluster:
        raise ValueError(f"cluster id must be an integer, got {value!r} at {context}")
    return cluster


def load_mapping(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_MAPPING_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"mapping is missing columns: {', '.join(missing)}")
        table: dict[tuple[str, int], dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            platform = row["platform"].strip().lower()
            if platform not in {"android", "ios"}:
                raise ValueError(f"unsupported platform {platform!r} at mapping line {line_number}")
            cluster = _cluster_id(row["cluster_id"], context=f"mapping line {line_number}")
            key = (platform, cluster)
            if key in table:
                raise ValueError(f"duplicate mapping key {key} at line {line_number}")
            if cluster != -1:
                empty = [column for column in ("cluster_name", "business_family", "business_submodule") if not row[column].strip()]
                if empty:
                    raise ValueError(f"non-noise mapping {key} has empty columns: {', '.join(empty)}")
                if row["business_family"].strip().casefold() == "chưa phân loại":
                    raise ValueError(f"non-noise mapping {key} cannot use 'Chưa phân loại'")
            table[key] = row
    for platform in ("android", "ios"):
        if (platform, -1) not in table:
            raise ValueError(f"mapping must contain the noise key {(platform, -1)}")
    return table


def _mapping_text(row: dict[str, str], *columns: str) -> str:
    for column in columns:
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return ""


def build_shareholder_catalog(
    mapping: dict[tuple[str, int], dict[str, str]],
    platform: str,
    cluster_counts: dict[int, int],
) -> dict[str, object]:
    """Return the shareholder catalog in the established business-v1 schema."""
    clusters = []
    platform_rows = sorted(
        (
            (cluster, row)
            for (source_platform, cluster), row in mapping.items()
            if source_platform == platform
        ),
        key=lambda item: item[0],
    )
    for cluster, row in platform_rows:
        top_ngrams = _mapping_text(row, "top_ngrams", "top_mass_ngrams")
        ngram_count = len([value for value in top_ngrams.split(" || ") if value.strip()])
        clusters.append(
            {
                "cluster": str(cluster),
                "size": _mapping_text(row, "journey_count", "size"),
                "share": _mapping_text(row, "journey_share", "share"),
                "mapping_name": _mapping_text(row, "cluster_name", "mapping_name"),
                "business_family": _mapping_text(row, "business_family"),
                "business_submodule": _mapping_text(row, "business_submodule"),
                "business_detail": _mapping_text(row, "business_detail"),
                "confidence": _mapping_text(row, "naming_confidence", "confidence"),
                "evidence_share": _mapping_text(row, "evidence_share"),
                "secondary_function_codes": _mapping_text(row, "secondary_function_codes"),
                "ngram_primary_function": _mapping_text(row, "ngram_primary_function"),
                "ngram_evidence_share": _mapping_text(row, "ngram_evidence_share"),
                "all_8_ngrams_used": str(ngram_count),
                "dominant_cluster_intents": _mapping_text(row, "dominant_cluster_intents"),
                "top_mass_ngrams": top_ngrams,
                "rare_token_caution": _mapping_text(row, "rare_token_caution"),
                "medoid_path_support_only": _mapping_text(row, "medoid_path", "medoid_path_support_only"),
                "cluster_id": cluster,
                "row_count_in_input": cluster_counts.get(cluster, 0),
                "platform": platform,
            }
        )
    return {
        "schema_version": "shareholder-cluster-catalog-business-v1",
        "platforms": [{"platform": platform, "clusters": clusters}],
    }


def write_shareholder_catalog(
    path: Path,
    mapping: dict[tuple[str, int], dict[str, str]],
    platform: str,
    cluster_counts: dict[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(
                build_shareholder_catalog(mapping, platform, cluster_counts),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_parquet() -> object:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ValueError(
            "Parquet input requires pyarrow; install project dependencies with "
            "`pip install -r requirements.txt`"
        ) from exc
    return parquet


def _input_columns(input_path: Path) -> list[str]:
    if input_path.suffix.lower() == ".csv":
        with input_path.open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source).fieldnames or ())
    parquet = _require_parquet()
    return list(parquet.ParquetFile(input_path).schema_arrow.names)


def _input_rows(input_path: Path, parquet_batch_size: int):
    if input_path.suffix.lower() == ".csv":
        with input_path.open(encoding="utf-8-sig", newline="") as source:
            yield from csv.DictReader(source)
        return
    parquet = _require_parquet()
    parquet_file = parquet.ParquetFile(input_path)
    for batch in parquet_file.iter_batches(batch_size=parquet_batch_size):
        yield from batch.to_pylist()


def apply_input(
    input_path: Path,
    output_path: Path,
    mapping: dict[tuple[str, int], dict[str, str]],
    platform: str,
    cluster_column: str | None = None,
    catalog_output: Path | None = None,
    parquet_batch_size: int = 50_000,
) -> dict[str, object]:
    if parquet_batch_size < 1:
        raise ValueError("parquet batch size must be positive")
    input_columns = _input_columns(input_path)
    selected_cluster = cluster_column or ("cluster" if "cluster" in input_columns else "cluster_id")
    if selected_cluster not in input_columns:
        raise ValueError("input must contain cluster or cluster_id")

    base_columns = [column for column in input_columns if column not in MAPPED_COLUMNS]
    if "platform" not in base_columns:
        base_columns.append("platform")
    if "cluster_id" not in base_columns:
        insert_at = base_columns.index(selected_cluster) + 1
        base_columns.insert(insert_at, "cluster_id")
    mapped_insert_at = base_columns.index("cluster_id") + 1
    output_columns = base_columns[:mapped_insert_at] + list(MAPPED_COLUMNS) + base_columns[mapped_insert_at:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    rows = 0
    cluster_counts: dict[int, int] = {}
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=output_columns, extrasaction="ignore")
            writer.writeheader()
            for row_number, row in enumerate(_input_rows(input_path, parquet_batch_size), start=1):
                source_platform = str(row.get("platform", "") or "").strip().lower()
                if source_platform and source_platform != platform:
                    raise ValueError(
                        f"input platform {source_platform!r} does not match --platform {platform!r} at row {row_number}"
                    )
                cluster = _cluster_id(row.get(selected_cluster), context=f"input row {row_number}")
                name = mapping.get((platform, cluster))
                if name is None:
                    raise ValueError(
                        f"cluster {(platform, cluster)} is absent from mapping; regenerate mapping for this model version"
                    )
                row["platform"] = platform
                row["cluster_id"] = str(cluster)
                for column in MAPPED_COLUMNS:
                    row[column] = name[column]
                writer.writerow(row)
                rows += 1
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    if catalog_output is not None:
        write_shareholder_catalog(catalog_output, mapping, platform, cluster_counts)

    return {
        "input": str(input_path),
        "input_format": input_path.suffix.lower().lstrip("."),
        "mapping_platform": platform,
        "output": str(output_path),
        "rows": rows,
        "clusters": len(cluster_counts),
        "noise_rows": cluster_counts.get(-1, 0),
        "shareholder_catalog": str(catalog_output) if catalog_output is not None else None,
    }


def apply_csv(
    input_path: Path,
    output_path: Path,
    mapping: dict[tuple[str, int], dict[str, str]],
    platform: str,
    cluster_column: str | None = None,
    catalog_output: Path | None = None,
) -> dict[str, object]:
    """Backward-compatible wrapper; ``apply_input`` also supports Parquet."""
    return apply_input(
        input_path,
        output_path,
        mapping,
        platform,
        cluster_column,
        catalog_output,
    )


def main() -> int:
    args = parse_args()
    input_path = _resolve(args.input)
    mapping_path = _resolve(args.mapping)
    output_path = _resolve(args.output)
    catalog_path = _resolve(args.catalog_output) if args.catalog_output else output_path.parent / "shareholder_cluster_catalog.json"
    if not input_path.is_file() or input_path.suffix.lower() not in {".csv", ".parquet"}:
        raise SystemExit(f"input must be an existing CSV or Parquet file: {input_path}")
    if not mapping_path.is_file():
        raise SystemExit(f"mapping not found: {mapping_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output_path}; pass --overwrite or choose another path")
    if catalog_path.exists() and not args.overwrite:
        raise SystemExit(f"catalog output exists: {catalog_path}; pass --overwrite or choose another path")
    if output_path == input_path:
        raise SystemExit("output must differ from input")
    if len({input_path, mapping_path, output_path, catalog_path}) != 4:
        raise SystemExit("input, mapping, output, and catalog output must be different files")
    if args.parquet_batch_size < 1:
        raise SystemExit("--parquet-batch-size must be positive")

    try:
        summary = apply_input(
            input_path,
            output_path,
            load_mapping(mapping_path),
            args.platform,
            args.cluster_column,
            catalog_path,
            args.parquet_batch_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    manifest = output_path.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
