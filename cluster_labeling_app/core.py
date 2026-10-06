"""Pure data helpers for the taxonomy-driven cluster review app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PLATFORMS = ("android", "ios")
TAXONOMY_COLUMNS = (
    "taxonomy_id",
    "business_family",
    "business_submodule",
    "business_detail",
)
TAXONOMY_ALIASES = {
    "taxonomy id": "taxonomy_id",
    "taxonomy_id": "taxonomy_id",
    "business family": "business_family",
    "business submodule": "business_submodule",
    "business details": "business_detail",
    "business detail": "business_detail",
    "business_family": "business_family",
    "business_submodule": "business_submodule",
    "business_detail": "business_detail",
}


@dataclass(frozen=True)
class PlatformEvidence:
    """Read-only model evidence published alongside named_clusters."""

    catalog: pd.DataFrame
    ngrams: pd.DataFrame


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).casefold() in {"1", "true", "yes", "y", "có"}


def normalize_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an editable taxonomy while preserving incomplete rows for validation."""

    renamed = {
        column: TAXONOMY_ALIASES.get(clean_text(column).casefold(), clean_text(column))
        for column in frame.columns
    }
    out = frame.rename(columns=renamed).copy()
    for column in TAXONOMY_COLUMNS:
        if column not in out:
            out[column] = ""
        out[column] = out[column].map(clean_text)
    out = out.loc[:, list(TAXONOMY_COLUMNS)]
    populated = out.ne("").any(axis=1)
    return out.loc[populated].reset_index(drop=True)


def validate_taxonomy(frame: pd.DataFrame) -> list[str]:
    """Validate the strict three-level production taxonomy."""

    taxonomy = normalize_taxonomy(frame)
    errors: list[str] = []
    label_map = {
        "business_family": "Business Family",
        "business_submodule": "Business Submodule",
        "business_detail": "Business Detail",
    }
    for column, label in label_map.items():
        missing = taxonomy[column].eq("")
        if missing.any():
            rows = ", ".join(str(index + 1) for index in taxonomy.index[missing][:8])
            errors.append(f"Thiếu {label} ở dòng: {rows}.")
    nonblank_ids = taxonomy["taxonomy_id"].ne("")
    duplicate_ids = nonblank_ids & taxonomy["taxonomy_id"].duplicated(keep=False)
    if duplicate_ids.any():
        errors.append(f"Có {int(duplicate_ids.sum())} dòng trùng taxonomy_id.")
    duplicate_paths = taxonomy.duplicated(
        ["business_family", "business_submodule", "business_detail"], keep=False
    )
    if duplicate_paths.any():
        errors.append(f"Có {int(duplicate_paths.sum())} đường dẫn taxonomy bị trùng.")
    return errors


def taxonomy_choices(frame: pd.DataFrame) -> dict[str, object]:
    """Build ordered cascading choices plus path-to-ID lookup."""

    taxonomy = normalize_taxonomy(frame)
    taxonomy = taxonomy.loc[
        taxonomy["business_family"].ne("")
        & taxonomy["business_submodule"].ne("")
        & taxonomy["business_detail"].ne("")
    ].drop_duplicates(["business_family", "business_submodule", "business_detail"])
    families = list(dict.fromkeys(taxonomy["business_family"].tolist()))
    submodules: dict[str, list[str]] = {}
    details: dict[tuple[str, str], list[str]] = {}
    path_to_id: dict[tuple[str, str, str], str] = {}
    for row in taxonomy.itertuples(index=False):
        path = (row.business_family, row.business_submodule, row.business_detail)
        path_to_id[path] = row.taxonomy_id
        submodules.setdefault(row.business_family, [])
        if row.business_submodule not in submodules[row.business_family]:
            submodules[row.business_family].append(row.business_submodule)
        details.setdefault((row.business_family, row.business_submodule), [])
        if row.business_detail not in details[(row.business_family, row.business_submodule)]:
            details[(row.business_family, row.business_submodule)].append(row.business_detail)
    return {
        "families": families,
        "submodules": submodules,
        "details": details,
        "pairs": {
            (family, submodule)
            for family, values in submodules.items()
            for submodule in values
        },
        "triples": set(path_to_id),
        "path_to_id": path_to_id,
    }


def assignment_status(
    frame: pd.DataFrame,
    family: str,
    submodule: str,
    detail: str,
    choices: dict[str, object] | None = None,
) -> tuple[bool, str]:
    """Check a cluster assignment against the active taxonomy."""

    if family == "Chưa phân loại":
        return True, ""
    if choices is None:
        choices = taxonomy_choices(frame)
    if family not in choices["families"]:
        return False, "Business Family hiện tại không còn trong taxonomy."
    if (family, submodule) not in choices["pairs"]:
        return False, "Business Submodule hiện tại không thuộc Family này."
    if detail and (family, submodule, detail) not in choices["triples"]:
        return False, "Business Detail hiện tại không thuộc cặp Family/Submodule này."
    return True, ""


def cluster_name(family: str, submodule: str, detail: str = "") -> str:
    return " | ".join(value for value in (family, submodule, detail) if value)


def build_platform_evidence(
    platform: str, catalog: pd.DataFrame, ngrams: pd.DataFrame
) -> PlatformEvidence:
    """Validate and normalise one platform's catalog and n-grams."""

    if platform not in PLATFORMS:
        raise ValueError(f"platform không hỗ trợ: {platform}")
    catalog = catalog.rename(columns={"cluster": "cluster_id"})
    catalog["cluster_id"] = pd.to_numeric(catalog["cluster_id"], errors="raise").astype(int)
    if catalog["cluster_id"].duplicated().any():
        raise ValueError(f"{platform}: catalog có cluster_id bị trùng")
    ngrams = ngrams.copy()
    ngrams["cluster"] = pd.to_numeric(ngrams["cluster"], errors="raise").astype(int)
    ngrams["rank"] = pd.to_numeric(ngrams["rank"], errors="raise").astype(int)
    ngrams["lift"] = pd.to_numeric(ngrams["lift"], errors="coerce")
    ngrams["cluster_mass"] = pd.to_numeric(ngrams["cluster_mass"], errors="coerce")
    ngrams["ngram"] = ngrams["ngram"].map(clean_text)
    catalog_ids = set(catalog["cluster_id"])
    ngram_ids = set(ngrams["cluster"])
    if catalog_ids != ngram_ids:
        raise ValueError(f"{platform}: catalog và n-gram không cùng cluster_id")
    return PlatformEvidence(
        catalog.sort_values("cluster_id").reset_index(drop=True),
        ngrams.sort_values(["cluster", "rank"]).reset_index(drop=True),
    )


def load_platform_evidence(asset_dir: Path, platform: str) -> PlatformEvidence:
    """Read evidence from pipeline files (publish step only; the app reads the database)."""

    return build_platform_evidence(
        platform,
        pd.read_json(asset_dir / f"{platform}_cluster_catalog.json"),
        pd.read_csv(asset_dir / f"{platform}_cluster_ngrams.csv", encoding="utf-8-sig"),
    )


def evidence_records(evidence: PlatformEvidence) -> list[dict[str, object]]:
    """One JSON-safe record per cluster: {cluster_id, catalog, ngrams}."""

    def records(frame: pd.DataFrame) -> list[dict[str, object]]:
        return json.loads(frame.to_json(orient="records", force_ascii=False))

    ngram_columns = ["rank", "ngram", "lift", "cluster_mass"]
    grouped = {
        int(cluster_id): records(group[ngram_columns])
        for cluster_id, group in evidence.ngrams.groupby("cluster")
    }
    return [
        {
            "cluster_id": int(row["cluster_id"]),
            "catalog": row,
            "ngrams": grouped.get(int(row["cluster_id"]), []),
        }
        for row in records(evidence.catalog)
    ]


def evidence_from_records(platform: str, rows: Iterable[dict[str, object]]) -> PlatformEvidence:
    """Rebuild PlatformEvidence from database rows written by evidence_records()."""

    rows = list(rows)
    catalog = pd.DataFrame([{**row["catalog"], "cluster_id": row["cluster_id"]} for row in rows])
    ngrams = pd.DataFrame(
        [{**ngram, "cluster": row["cluster_id"]} for row in rows for ngram in row["ngrams"]],
        columns=["cluster", "rank", "ngram", "lift", "cluster_mass"],
    )
    if catalog.empty:
        raise ValueError(f"{platform}: database chưa có cluster_evidence")
    return build_platform_evidence(platform, catalog, ngrams)


def catalog_row(catalog: pd.DataFrame, cluster_id: int) -> pd.Series:
    rows = catalog.loc[catalog["cluster_id"].eq(int(cluster_id))]
    if rows.empty:
        raise KeyError(cluster_id)
    return rows.iloc[0]


def cluster_ngrams(ngrams: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
    columns = ["rank", "ngram", "lift", "cluster_mass"]
    return ngrams.loc[ngrams["cluster"].eq(int(cluster_id)), columns].copy()


def filter_named_clusters(
    frame: pd.DataFrame,
    *,
    review_values: Iterable[bool],
    confidence_values: Iterable[str],
    query: str,
) -> pd.DataFrame:
    out = frame.copy()
    reviews = list(review_values)
    confidences = list(confidence_values)
    if reviews:
        out = out.loc[out["needs_review"].isin(reviews)]
    if confidences:
        out = out.loc[out["naming_confidence"].isin(confidences)]
    normalized_query = clean_text(query).casefold()
    if normalized_query:
        columns = [
            "cluster_id",
            "taxonomy_id",
            "cluster_name",
            "business_family",
            "business_submodule",
            "business_detail",
            "supporting_evidence",
        ]
        haystack = out[columns].fillna("").astype(str).agg(" ".join, axis=1).str.casefold()
        out = out.loc[haystack.str.contains(normalized_query, regex=False)]
    return out.sort_values(
        ["needs_review", "size", "cluster_id"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def export_taxonomy_csv(frame: pd.DataFrame) -> bytes:
    out = normalize_taxonomy(frame)
    return out.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def export_named_clusters_csv(frame: pd.DataFrame) -> bytes:
    out = frame.copy()
    out["needs_review"] = out["needs_review"].map(lambda value: str(bool(value)).lower())
    return out.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
