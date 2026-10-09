"""Load and apply the business taxonomy assigned to clustering output.

Business names follow the HiFPT sitemap taxonomy (module > submodule > feature,
ids such as ``M05``, ``M05.06`` and ``M05.06.03``). Every reviewed cluster
points at one sitemap id, at the deepest level the reviewer could name.
``load_sitemap`` reads that taxonomy, ``load_cluster_taxonomy`` reads the
per-cluster file of one model, and ``apply_cluster_taxonomy`` /
``map_scores_file`` put the ``taxonomy_id`` on scored journeys of either
platform (CLI: ``journey-map``).
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



# Repository copy of the HiFPT sitemap taxonomy (`journey_sitemap_taxonomy` table).
DEFAULT_SITEMAP = Path(__file__).resolve().parents[2] / "taxonomy" / "hifpt_sitemap_taxonomy.csv"
# Columns of the `journey_sitemap_taxonomy` table: id, module, submodule, feature.
SITEMAP_COLUMNS = ("taxonomy_id", "business_family", "business_submodule", "business_detail")
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


def load_sitemap(path: str | Path = DEFAULT_SITEMAP) -> dict[str, dict[str, str]]:
    """Return ``taxonomy_id -> names`` for every module, submodule and feature.

    ``path`` is the sitemap JSON (``features`` list with module_id, module,
    submodule_id, submodule, feature_id, feature) or the CSV ``write_sitemap``
    produces from it. Modules and submodules get their own rows so a cluster
    named only to level 1 or 2 still has an id.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows: dict[str, dict[str, str]] = {}
        for item in _load_json(path)["features"]:
            levels = (
                (item["module_id"], item["module"], "", ""),
                (item["submodule_id"], item["module"], item["submodule"], ""),
                (item["feature_id"], item["module"], item["submodule"], item["feature"]),
            )
            for row in levels:
                named = dict(zip(SITEMAP_COLUMNS, (value.strip() for value in row)))
                if rows.setdefault(named["taxonomy_id"], named) != named:
                    raise ValueError(f"{path}: id {named['taxonomy_id']} has two different names")
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = {row["taxonomy_id"]: {c: row[c] or "" for c in SITEMAP_COLUMNS} for row in csv.DictReader(handle)}
    return dict(sorted(rows.items()))


def write_sitemap(sitemap: dict[str, dict[str, str]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SITEMAP_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sitemap.values())


def sitemap_id(row: dict[str, Any], sitemap: dict[str, dict[str, str]]) -> tuple[str, str | None]:
    """Return the sitemap id of a named cluster, or an error.

    A given ``taxonomy_id`` must exist and carry the same three names. Without
    one, the names are looked up: a cluster named to level 2 gets the
    submodule id, a cluster named to level 1 the module id.
    """
    names = tuple((row.get(column) or "").strip() for column in SITEMAP_COLUMNS[1:])
    given = (row.get("taxonomy_id") or "").strip()
    if given:
        expected = sitemap.get(given)
        if expected is None:
            return given, f"taxonomy_id {given} is not in the sitemap"
        if tuple(expected[c] for c in SITEMAP_COLUMNS[1:]) != names:
            return given, f"taxonomy_id {given} is {expected['business_family']} | " \
                f"{expected['business_submodule']} | {expected['business_detail']}, not {' | '.join(names)}"
        return given, None
    for taxonomy_id, expected in sitemap.items():
        if tuple(expected[c] for c in SITEMAP_COLUMNS[1:]) == names:
            return taxonomy_id, None
    return "", f"names {' | '.join(names)} are not in the sitemap"


def load_cluster_taxonomy(
    path: str | Path,
    *,
    platform: str,
    model_version: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Load ``cluster_id -> taxonomy_id`` (plus names) for one platform.

    ``path`` is the per-cluster file ``journey_cluster_taxonomy.csv`` (both
    platforms, filtered by ``platform`` and ``model_version``) built by
    ``journey-cluster-taxonomy``. Cluster -1 always gets ``NOISE_NAME`` (no id).
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
        table[cluster_id] = {column: row.get(column, "") or "" for column in SITEMAP_COLUMNS}
        table[cluster_id]["model_version"] = row.get("model_version", "")
    table[-1] = {**table.get(-1, {"model_version": ""}), **NOISE_NAME}
    return table


def apply_cluster_taxonomy(
    frame: pd.DataFrame,
    taxonomy: dict[int, dict[str, Any]],
    *,
    sitemap: dict[str, dict[str, str]] | None = None,
    columns: tuple[str, ...] | None = None,
    cluster_column: str | None = None,
) -> pd.DataFrame:
    """Put ``taxonomy_id`` right after the cluster column.

    With ``sitemap`` the three sitemap names follow it, for reading the file
    without a join. ``columns`` instead copies those fields of the taxonomy
    rows as they are (e.g. the reviewer's ``cluster_name``). Works on the current output (``cluster_id``) and on older
    exports (``cluster``). A cluster id the taxonomy does not know gets the -1
    noise entry (no id) instead of a wrong one, and names from another model
    are refused.
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
    if columns is not None:
        return _apply_rows(frame, rows, columns, cluster_column)
    if sitemap is None:
        return _apply_rows(frame, rows, SITEMAP_COLUMNS[:1], cluster_column)
    empty = dict.fromkeys(SITEMAP_COLUMNS, "")
    named = [sitemap.get(row["taxonomy_id"], empty) for row in rows]
    return _apply_rows(frame, named, SITEMAP_COLUMNS, cluster_column)
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
    sitemap: dict[str, dict[str, str]] | None = None,
    columns: tuple[str, ...] | None = None,
    chunksize: int = 200_000,
) -> dict[str, Any]:
    """Write ``input_path`` (CSV, parquet file or folder) with ``taxonomy_id``.

    The file is read ``chunksize`` rows at a time, so a multi-GB scores file
    never sits in memory. Returns the row count, the journeys of clusters the
    taxonomy does not know, those left without an id (cluster -1) and the
    journeys per sitemap module.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    rows, unknown, without_id = 0, 0, 0
    per_module: dict[str, int] = {}
    known = set(taxonomy)
    try:
        for index, chunk in enumerate(_read_chunks(input_path, chunksize)):
            named = apply_cluster_taxonomy(chunk, taxonomy, sitemap=sitemap, columns=columns)
            cluster_column = "cluster_id" if "cluster_id" in named.columns else "cluster"
            unknown += int((~named[cluster_column].astype(int).isin(known)).sum())
            ids = named["taxonomy_id"].fillna("").astype(str)
            without_id += int(ids.eq("").sum())
            for module, count in ids[ids.ne("")].str[:3].value_counts().items():
                per_module[module] = per_module.get(module, 0) + int(count)
            named.to_csv(temporary, mode="w" if index == 0 else "a", header=index == 0, index=False)
            rows += len(named)
        if rows == 0:
            raise ValueError(f"{input_path} has no journeys")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"rows": rows, "unknown_cluster_rows": unknown, "rows_without_id": without_id, "per_module": per_module}
