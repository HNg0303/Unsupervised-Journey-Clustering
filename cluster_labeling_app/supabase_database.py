"""Supabase/PostgREST repository with the same contract as database.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from supabase import Client, create_client

from core import PLATFORMS, clean_text, cluster_name, normalize_taxonomy, validate_taxonomy
from database import _read_csv, check_cluster_taxonomy, read_named_cluster_assets, utc_now


@dataclass(frozen=True)
class SupabaseTarget:
    url: str
    key: str

    @property
    def display_name(self) -> str:
        return self.url


@lru_cache(maxsize=4)
def _client(target: SupabaseTarget) -> Client:
    # One client per target keeps the HTTP connection pool (and TLS session) warm.
    return create_client(target.url, target.key)


def _rows(response: Any) -> list[dict[str, Any]]:
    return list(response.data or [])


def _fetch_all(
    client: Client,
    table: str,
    columns: str = "*",
    *,
    filters: Iterable[tuple[str, object]] = (),
    order: str | None = None,
    descending: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_size = 1000
    start = 0
    while True:
        query = client.table(table).select(columns)
        for column, value in filters:
            query = query.eq(column, value)
        if order:
            query = query.order(order, desc=descending)
        batch = _rows(query.range(start, start + page_size - 1).execute())
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        start += page_size


def _count(client: Client, table: str, filters: Iterable[tuple[str, object]] = ()) -> int:
    query = client.table(table).select("*", count="exact", head=True)
    for column, value in filters:
        query = query.eq(column, value)
    return int(query.execute().count or 0)


def _chunks(rows: list[dict[str, Any]], size: int = 250):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def initialize_database(target: SupabaseTarget, asset_dir: Path) -> dict[str, int]:
    """Check the Supabase schema and seed it once when it is completely empty."""

    client = _client(target)
    try:
        taxonomy_ids = _rows(
            client.table("taxonomy_features").select("taxonomy_id").limit(1).execute()
        )
        cluster_ids = _rows(
            client.table("named_clusters").select("cluster_id").limit(1).execute()
        )
    except Exception as error:
        raise ValueError(
            "Không đọc được schema Supabase. Hãy chạy supabase_schema.sql trong SQL Editor "
            "và cấu hình SUPABASE_SECRET_KEY trong Streamlit Secrets."
        ) from error
    if not taxonomy_ids and not cluster_ids:
        _seed_database(client, asset_dir)
    elif not taxonomy_ids or not cluster_ids:
        raise ValueError("Supabase chỉ được seed một phần; cần kiểm tra taxonomy_features/named_clusters")
    client.table("app_metadata").upsert(
        {"key": "schema_version", "value": "1"}, on_conflict="key"
    ).execute()
    return database_stats(target)


def _seed_database(client: Client, asset_dir: Path) -> None:
    timestamp = utc_now()
    taxonomy_source = _read_csv(asset_dir / "taxonomy_features.csv")
    taxonomy_rows = [
        {
            "taxonomy_id": clean_text(row["taxonomy_id"]),
            "business_family": clean_text(row["business_family"]),
            "business_submodule": clean_text(row["business_submodule"]),
            "business_detail": clean_text(row["business_detail"]),
            "sort_order": index,
            "is_active": True,
            "source": "taxonomy_pipeline",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        for index, row in enumerate(taxonomy_source)
    ]
    for batch in _chunks(taxonomy_rows):
        client.table("taxonomy_features").upsert(batch, on_conflict="taxonomy_id").execute()

    cluster_rows = read_named_cluster_assets(asset_dir, timestamp)
    for batch in _chunks(cluster_rows):
        client.table("named_clusters").upsert(
            batch, on_conflict="platform,cluster_id"
        ).execute()
    client.table("app_metadata").upsert(
        [
            {"key": "seeded_at", "value": timestamp},
            {"key": "seed_source", "value": "taxonomy_naming"},
        ],
        on_conflict="key",
    ).execute()


def database_stats(target: SupabaseTarget) -> dict[str, int]:
    client = _client(target)
    return {
        "taxonomy_features": _count(client, "taxonomy_features", (("is_active", True),)),
        "android_clusters": _count(client, "named_clusters", (("platform", "android"),)),
        "ios_clusters": _count(client, "named_clusters", (("platform", "ios"),)),
        "audit_events": _count(client, "change_audit"),
    }


def load_taxonomy(target: SupabaseTarget, active_only: bool = True) -> pd.DataFrame:
    filters = (("is_active", True),) if active_only else ()
    rows = _fetch_all(
        _client(target),
        "taxonomy_features",
        "taxonomy_id,business_family,business_submodule,business_detail,sort_order,is_active",
        filters=filters,
        order="sort_order",
    )
    columns = ["taxonomy_id", "business_family", "business_submodule", "business_detail"]
    return pd.DataFrame(rows, columns=columns)


def load_named_clusters(target: SupabaseTarget, platform: str) -> pd.DataFrame:
    if platform not in PLATFORMS:
        raise ValueError(platform)
    rows = _fetch_all(
        _client(target),
        "named_clusters",
        filters=(("platform", platform),),
        order="cluster_id",
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["taxonomy_id"] = frame["taxonomy_id"].fillna("")
    frame["needs_review"] = frame["needs_review"].astype(bool)
    return frame


def _next_custom_counter(rows: dict[str, dict[str, Any]]) -> int:
    values = [
        int(value[4:])
        for value in rows
        if value.startswith("USR.") and value[4:].isdigit()
    ]
    return max(values, default=0) + 1


def _audit_rows(client: Client, rows: list[dict[str, Any]]) -> None:
    if rows:
        client.table("change_audit").insert(rows).execute()


def save_taxonomy(target: SupabaseTarget, edited: pd.DataFrame) -> dict[str, int]:
    frame = normalize_taxonomy(edited)
    errors = validate_taxonomy(frame)
    if errors:
        raise ValueError(" ".join(errors))
    client = _client(target)
    current_rows = {
        row["taxonomy_id"]: row
        for row in _fetch_all(client, "taxonomy_features", order="sort_order")
    }
    counter = _next_custom_counter(current_rows)
    timestamp = utc_now()
    normalized: list[dict[str, Any]] = []
    for sort_order, row in enumerate(frame.to_dict("records")):
        taxonomy_id = clean_text(row["taxonomy_id"])
        if not taxonomy_id:
            taxonomy_id = f"USR.{counter:06d}"
            counter += 1
        normalized.append(
            {
                "taxonomy_id": taxonomy_id,
                "business_family": clean_text(row["business_family"]),
                "business_submodule": clean_text(row["business_submodule"]),
                "business_detail": clean_text(row["business_detail"]),
                "sort_order": sort_order,
            }
        )
    ids = [row["taxonomy_id"] for row in normalized]
    paths = [
        (row["business_family"], row["business_submodule"], row["business_detail"])
        for row in normalized
    ]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("taxonomy_id hoặc đường dẫn taxonomy bị trùng")

    inserted = updated = deactivated = propagated = 0
    audit: list[dict[str, Any]] = []
    upserts: list[dict[str, Any]] = []
    for row in normalized:
        old = current_rows.get(row["taxonomy_id"])
        old_path = (
            {
                "business_family": old["business_family"],
                "business_submodule": old["business_submodule"],
                "business_detail": old["business_detail"],
            }
            if old
            else {}
        )
        new_path = {key: row[key] for key in ("business_family", "business_submodule", "business_detail")}
        changed = old is None or old_path != new_path or not bool(old.get("is_active", True))
        upserts.append(
            {
                **row,
                "is_active": True,
                "source": "business_review" if changed else old.get("source", "taxonomy_pipeline"),
                "created_at": old.get("created_at", timestamp) if old else timestamp,
                "updated_at": timestamp if changed else old.get("updated_at", timestamp),
            }
        )
        if changed:
            if old is None:
                inserted += 1
            else:
                updated += 1
                affected = _fetch_all(
                    client,
                    "named_clusters",
                    "platform,cluster_id",
                    filters=(("taxonomy_id", row["taxonomy_id"]),),
                )
                propagated += len(affected)
                client.table("named_clusters").update(
                    {
                        **new_path,
                        "cluster_name": cluster_name(*new_path.values()),
                        "naming_source": "business_review_taxonomy_update",
                        "needs_review": True,
                        "updated_at": timestamp,
                    }
                ).eq("taxonomy_id", row["taxonomy_id"]).execute()
            audit.append(
                {
                    "entity_type": "taxonomy_feature",
                    "entity_key": row["taxonomy_id"],
                    "old_value": old_path,
                    "new_value": new_path,
                    "changed_at": timestamp,
                }
            )
    for batch in _chunks(upserts):
        client.table("taxonomy_features").upsert(batch, on_conflict="taxonomy_id").execute()

    incoming = set(ids)
    for taxonomy_id, old in current_rows.items():
        if bool(old.get("is_active")) and taxonomy_id not in incoming:
            client.table("taxonomy_features").update(
                {"is_active": False, "updated_at": timestamp}
            ).eq("taxonomy_id", taxonomy_id).execute()
            affected = _fetch_all(
                client,
                "named_clusters",
                "platform,cluster_id",
                filters=(("taxonomy_id", taxonomy_id),),
            )
            propagated += len(affected)
            client.table("named_clusters").update(
                {
                    "naming_source": "business_review_taxonomy_removed",
                    "needs_review": True,
                    "updated_at": timestamp,
                }
            ).eq("taxonomy_id", taxonomy_id).execute()
            deactivated += 1
            audit.append(
                {
                    "entity_type": "taxonomy_feature",
                    "entity_key": taxonomy_id,
                    "old_value": {"is_active": True},
                    "new_value": {"is_active": False},
                    "changed_at": timestamp,
                }
            )
    _audit_rows(client, audit)
    return {
        "inserted": inserted,
        "updated": updated,
        "deactivated": deactivated,
        "propagated_clusters": propagated,
    }


def update_named_cluster(
    target: SupabaseTarget,
    platform: str,
    cluster_id: int,
    *,
    family: str,
    submodule: str,
    detail: str,
    confidence: str,
    needs_review: bool,
) -> bool:
    if platform not in PLATFORMS:
        raise ValueError(platform)
    client = _client(target)
    old_rows = _rows(
        client.table("named_clusters")
        .select("*")
        .eq("platform", platform)
        .eq("cluster_id", int(cluster_id))
        .limit(1)
        .execute()
    )
    if not old_rows:
        raise KeyError((platform, cluster_id))
    old = old_rows[0]
    family, submodule, detail = map(clean_text, (family, submodule, detail))
    taxonomy_id: str | None = None
    if family != "Chưa phân loại":
        if not family:
            raise ValueError("Business Family không được để trống")
        if not submodule:
            detail = ""
            needs_review = True
        elif detail:
            matches = _rows(
                client.table("taxonomy_features")
                .select("taxonomy_id")
                .eq("is_active", True)
                .eq("business_family", family)
                .eq("business_submodule", submodule)
                .eq("business_detail", detail)
                .limit(1)
                .execute()
            )
            if not matches:
                raise ValueError("Business Detail không tồn tại trong taxonomy đang active")
            taxonomy_id = matches[0]["taxonomy_id"]
        else:
            matches = _rows(
                client.table("taxonomy_features")
                .select("taxonomy_id")
                .eq("is_active", True)
                .eq("business_family", family)
                .eq("business_submodule", submodule)
                .limit(1)
                .execute()
            )
            if not matches:
                raise ValueError("Business Submodule không tồn tại trong taxonomy đang active")
    new_values = {
        "taxonomy_id": taxonomy_id or "",
        "cluster_name": cluster_name(family, submodule, detail),
        "business_family": family,
        "business_submodule": submodule,
        "business_detail": detail,
        "naming_confidence": clean_text(confidence),
        "needs_review": bool(needs_review),
    }
    old_values = {
        key: (bool(old[key]) if key == "needs_review" else clean_text(old.get(key)))
        for key in new_values
    }
    if old_values == new_values:
        return False
    timestamp = utc_now()
    payload = {**new_values, "taxonomy_id": taxonomy_id, "naming_source": "business_review", "updated_at": timestamp}
    client.table("named_clusters").update(payload).eq("platform", platform).eq(
        "cluster_id", int(cluster_id)
    ).execute()
    _audit_rows(
        client,
        [
            {
                "entity_type": "named_cluster",
                "entity_key": f"{platform}:{cluster_id}",
                "old_value": old_values,
                "new_value": new_values,
                "changed_at": timestamp,
            }
        ],
    )
    return True


def replace_named_clusters(
    target: SupabaseTarget, asset_dir: Path, model_version: str
) -> dict[str, int]:
    """Swap named_clusters for a newly trained model; taxonomy and audit history are kept.

    PostgREST has no multi-request transaction, so new rows are upserted before stale
    cluster IDs are deleted: a failure midway never leaves the table empty.
    """

    timestamp = utc_now()
    rows = read_named_cluster_assets(asset_dir, timestamp)
    problems = check_cluster_taxonomy(rows, load_taxonomy(target))
    if problems:
        raise ValueError(
            f"{len(problems)} cluster tham chiếu taxonomy_id không khớp taxonomy active: "
            + ", ".join(problems[:10])
        )
    client = _client(target)
    previous = _rows(
        client.table("app_metadata").select("value").eq("key", "model_version").execute()
    )
    old_counts = {
        platform: _count(client, "named_clusters", (("platform", platform),))
        for platform in PLATFORMS
    }
    for batch in _chunks(rows):
        client.table("named_clusters").upsert(batch, on_conflict="platform,cluster_id").execute()
    new_counts: dict[str, int] = {}
    for platform in PLATFORMS:
        keep = {row["cluster_id"] for row in rows if row["platform"] == platform}
        new_counts[platform] = len(keep)
        existing = _fetch_all(
            client, "named_clusters", "cluster_id", filters=(("platform", platform),)
        )
        stale = sorted({row["cluster_id"] for row in existing} - keep)
        for start in range(0, len(stale), 200):
            client.table("named_clusters").delete().eq("platform", platform).in_(
                "cluster_id", stale[start : start + 200]
            ).execute()
    client.table("app_metadata").upsert(
        {"key": "model_version", "value": model_version}, on_conflict="key"
    ).execute()
    _audit_rows(
        client,
        [
            {
                "entity_type": "cluster_refresh",
                "entity_key": model_version,
                "old_value": {
                    "model_version": previous[0]["value"] if previous else "",
                    **old_counts,
                },
                "new_value": {"model_version": model_version, **new_counts},
                "changed_at": timestamp,
            }
        ],
    )
    return new_counts


def load_recent_audit(target: SupabaseTarget, limit: int = 20) -> pd.DataFrame:
    rows = _rows(
        _client(target)
        .table("change_audit")
        .select("entity_type,entity_key,old_value,new_value,changed_at")
        .order("audit_id", desc=True)
        .limit(int(limit))
        .execute()
    )
    for row in rows:
        for key in ("old_value", "new_value"):
            if isinstance(row.get(key), dict):
                row[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
    return pd.DataFrame(
        rows, columns=["entity_type", "entity_key", "old_value", "new_value", "changed_at"]
    )
