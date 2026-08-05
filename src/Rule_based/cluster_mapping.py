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


def load_cluster_mapping(path: str | Path) -> dict[int, dict[str, Any]]:
    """Return a mapping keyed by integer cluster id.

    The JSON may contain either a top-level ``classes`` list (the exported
    format) or be a plain list of class records.
    """
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)
    records = payload["classes"] if isinstance(payload, dict) else payload
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
    position = out.columns.get_loc(cluster_column) + 1
    for column in reversed(MAPPING_COLUMNS):
        out.insert(position, column, [row.get(column, "") for row in rows])
    return out

