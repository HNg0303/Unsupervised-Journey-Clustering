"""Load and apply the business taxonomy assigned to clustering output."""

from __future__ import annotations

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
