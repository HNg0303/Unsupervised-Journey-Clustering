"""Load and apply the business taxonomy assigned to clustering output.

The reviewed business names live once per cluster in the
``journey_cluster_taxonomy`` table (``scripts/build_cluster_taxonomy.py``).
``load_cluster_taxonomy`` and ``apply_cluster_taxonomy`` attach them to scored
journeys of either platform, and ``map_scores_file`` does it chunk by chunk for
the large ``<platform>_scores.csv`` files (CLI: ``journey-map``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


MAPPING_COLUMNS = (
    "class_group_code",
    "class_group",
    "class_code",
    "class_name",
    "class_description",
    "naming_confidence",
)

NAME_MAPPING_COLUMNS = (
    "business_family",
    "cluster_name",
    "naming_confidence",
)


def _records_from_payload(payload: Any, *, platform: str | None = None) -> tuple[list[dict[str, Any]], bool]:
    """Extract records from either taxonomy JSON or the shareholder catalog.

    The class taxonomy is a flat ``classes`` list.  The shareholder catalog is
    grouped under ``platforms[*].clusters`` and uses ``cluster_id`` instead of
    ``cluster``.  Returning whether the source is a catalog lets callers keep
    the two output contracts separate.
    """
    if isinstance(payload, dict) and "platforms" in payload:
        platforms = payload["platforms"]
        if not isinstance(platforms, list):
            raise ValueError("shareholder catalog field 'platforms' must be a list")
        if platform is None:
            if len(platforms) != 1:
                raise ValueError("platform is required when the catalog contains multiple platforms")
            selected = platforms[0]
        else:
            selected = next((item for item in platforms if item.get("platform") == platform), None)
            if selected is None:
                available = [item.get("platform") for item in platforms]
                raise ValueError(f"platform {platform!r} not found in catalog; available: {available}")
        records = selected.get("clusters")
        if not isinstance(records, list):
            raise ValueError("shareholder catalog platform entry must contain a 'clusters' list")
        return records, True

    if isinstance(payload, dict):
        records = payload.get("classes")
        if records is None:
            raise ValueError("mapping JSON must contain 'classes' or 'platforms'")
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("mapping records must be a list")
    return records, False


def _load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def load_cluster_name_mapping(
    path: str | Path,
    *,
    platform: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Load ``cluster_id -> name`` records from a shareholder catalog.

    The Vietnamese catalog contains both Android and iOS entries, so callers
    must pass ``platform`` for that file.  The old flat class-mapping format is
    also accepted and is normalized to the name-mapping fields.
    """
    records, is_catalog = _records_from_payload(_load_json(path), platform=platform)
    mapping: dict[int, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("each cluster mapping record must be an object")
        key = row.get("cluster_id") if is_catalog else row.get("cluster")
        if key is None:
            raise ValueError("cluster mapping record is missing cluster id")
        normalized = dict(row)
        normalized["cluster"] = int(key)
        normalized["cluster_name"] = row.get("cluster_name", row.get("class_name", ""))
        normalized["business_family"] = row.get(
            "business_family", row.get("class_group", row.get("class_group_code", ""))
        )
        mapping[int(key)] = normalized
    if -1 not in mapping:
        raise ValueError("cluster name mapping must define cluster -1 (unknown/noise)")
    return mapping


def _apply_rows(
    frame: pd.DataFrame,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    cluster_column: str,
) -> pd.DataFrame:
    out = frame.copy()
    # Re-applying a mapping should replace its derived columns rather than
    # fail with pandas' duplicate-column guard.
    out = out.drop(columns=[column for column in columns if column in out.columns])
    position = out.columns.get_loc(cluster_column) + 1
    for column in reversed(columns):
        out.insert(position, column, [row.get(column, "") for row in rows])
    return out


def apply_cluster_name_mapping(
    frame: pd.DataFrame,
    mapping: dict[int, dict[str, Any]] | str | Path,
    *,
    platform: str | None = None,
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    """Attach catalog names and business families to clustered journeys."""
    if cluster_column not in frame.columns:
        raise KeyError(f"missing cluster column: {cluster_column}")
    table = (
        load_cluster_name_mapping(mapping, platform=platform)
        if isinstance(mapping, (str, Path))
        else mapping
    )
    unknown = table[-1]
    rows = [table.get(int(cluster), unknown) for cluster in frame[cluster_column]]
    return _apply_rows(frame, rows, NAME_MAPPING_COLUMNS, cluster_column)


def load_cluster_mapping(path: str | Path) -> dict[int, dict[str, Any]]:
    """Return a mapping keyed by integer cluster id.

    The JSON may contain either a top-level ``classes`` list (the exported
    format) or be a plain list of class records.
    """
    records, is_catalog = _records_from_payload(_load_json(path))
    if is_catalog:
        raise ValueError("shareholder catalog needs a platform; use load_cluster_name_mapping(..., platform=...)")
    mapping = {int(row["cluster"]): row for row in records}
    if -1 not in mapping:
        raise ValueError("cluster mapping must define cluster -1 (unknown/noise)")
    return mapping


def apply_cluster_mapping(
    frame: pd.DataFrame,
    mapping: dict[int, dict[str, Any]] | str | Path,
    *,
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    """Attach group and meaningful class fields to clustered journeys.

    Cluster ids absent from the mapping deliberately fall back to the -1
    class. This makes inference safe after a model/mapping version mismatch:
    an unseen id is labelled unknown instead of receiving a wrong name.
    """
    if cluster_column not in frame.columns:
        raise KeyError(f"missing cluster column: {cluster_column}")
    table = load_cluster_mapping(mapping) if isinstance(mapping, (str, Path)) else mapping
    unknown = table[-1]
    out = frame.copy()
    rows = [table.get(int(cluster), unknown) for cluster in out[cluster_column]]
    return _apply_rows(out, rows, MAPPING_COLUMNS, cluster_column)



# Name fields of `journey_cluster_taxonomy` attached to every journey.
TAXONOMY_NAME_COLUMNS = (
    "taxonomy_id",
    "cluster_name",
    "business_family",
    "business_submodule",
    "business_detail",
    "naming_confidence",
    "naming_source",
    "needs_review",
)
# Cluster -1 holds the journeys no cluster accepted, a mix of every business
# area, so it always carries the noise name.
NOISE_NAME = {
    "taxonomy_id": "",
    "cluster_name": "Chưa phân loại | Journey hỗn hợp/nhiễu",
    "business_family": "Chưa phân loại",
    "business_submodule": "Journey hỗn hợp/nhiễu",
    "business_detail": "",
    "naming_confidence": "low",
    "naming_source": "noise_cluster",
    "needs_review": 0,
}
# Identifiers pandas would otherwise read as floats (6039276 -> 6039276.0).
ID_COLUMNS = ("journey_id", "session_id", "device_id", "customer_id")


def _needs_review(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true"})


def load_cluster_taxonomy(
    path: str | Path,
    *,
    platform: str,
    model_version: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Load ``cluster_id -> business name`` for one platform.

    ``path`` is the ``journey_cluster_taxonomy.csv`` table (both platforms,
    filtered by ``platform`` and ``model_version``) or a labeling app export
    ``<platform>_named_clusters.csv``. Cluster -1 always gets ``NOISE_NAME``.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if records and "platform" in records[0]:
        records = [row for row in records if row["platform"] == platform]
    if model_version is not None and records and "model_version" in records[0]:
        versions = sorted({row["model_version"] for row in records})
        records = [row for row in records if row["model_version"] == model_version]
        if not records:
            raise ValueError(f"{path} has no {platform} names for model {model_version!r}; it has {versions}")
    if not records:
        raise ValueError(f"{path} has no cluster names for platform {platform!r}")
    table: dict[int, dict[str, Any]] = {}
    for row in records:
        cluster_id = int(row["cluster_id"])
        if cluster_id in table:
            raise ValueError(f"{path}: cluster {cluster_id} is named twice for {platform}")
        names = {column: row.get(column, "") or "" for column in TAXONOMY_NAME_COLUMNS}
        names["needs_review"] = _needs_review(names["needs_review"])
        names["model_version"] = row.get("model_version", "")
        table[cluster_id] = names
    table[-1] = {**table.get(-1, {"model_version": ""}), **NOISE_NAME}
    return table


def apply_cluster_taxonomy(
    frame: pd.DataFrame,
    taxonomy: dict[int, dict[str, Any]],
    *,
    cluster_column: str | None = None,
) -> pd.DataFrame:
    """Insert the business name fields right after the cluster column.

    Works on the current output (``cluster_id``) and on older exports
    (``cluster``, with name columns that are replaced). A cluster id the
    taxonomy does not know gets the -1 noise name instead of a wrong one, and
    names from another model are refused.
    """
    if cluster_column is None:
        cluster_column = next((c for c in ("cluster_id", "cluster") if c in frame.columns), None)
    if cluster_column is None or cluster_column not in frame.columns:
        raise KeyError("missing cluster column: expected cluster_id or cluster")
    names_version = {row["model_version"] for row in taxonomy.values() if row.get("model_version")}
    if "model_version" in frame.columns and names_version:
        other = set(frame["model_version"].dropna().astype(str)) - names_version
        if other:
            raise ValueError(f"journeys from model {sorted(other)} but names are for {sorted(names_version)}")
    unknown = taxonomy[-1]
    rows = [taxonomy.get(int(cluster), unknown) for cluster in frame[cluster_column]]
    return _apply_rows(frame, rows, TAXONOMY_NAME_COLUMNS, cluster_column)


def _read_chunks(path: Path, chunksize: int):
    if path.is_dir() or path.suffix == ".parquet":
        import pyarrow.parquet as pq

        files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=chunksize):
                yield batch.to_pandas()
        return
    dtype = {column: "string" for column in ID_COLUMNS}
    yield from pd.read_csv(path, chunksize=chunksize, dtype=dtype, keep_default_na=True)


def map_scores_file(
    input_path: str | Path,
    output_path: str | Path,
    taxonomy: dict[int, dict[str, Any]],
    *,
    chunksize: int = 200_000,
) -> dict[str, Any]:
    """Write ``input_path`` (CSV, parquet file or folder) with names attached.

    The file is read ``chunksize`` rows at a time, so a multi-GB scores file
    never sits in memory. Returns the row count and journeys per cluster name.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    rows, unknown = 0, 0
    per_family: dict[str, int] = {}
    known = set(taxonomy)
    try:
        for index, chunk in enumerate(_read_chunks(input_path, chunksize)):
            named = apply_cluster_taxonomy(chunk, taxonomy)
            cluster_column = "cluster_id" if "cluster_id" in named.columns else "cluster"
            unknown += int((~named[cluster_column].astype(int).isin(known)).sum())
            for family, count in named["business_family"].value_counts().items():
                per_family[family] = per_family.get(family, 0) + int(count)
            named.to_csv(temporary, mode="w" if index == 0 else "a", header=index == 0, index=False)
            rows += len(named)
        if rows == 0:
            raise ValueError(f"{input_path} has no journeys")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"rows": rows, "unknown_cluster_rows": unknown, "per_business_family": per_family}
