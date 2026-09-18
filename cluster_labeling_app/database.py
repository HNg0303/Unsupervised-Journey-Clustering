"""SQLite repository for editable taxonomy features and named clusters."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from core import (
    PLATFORMS,
    as_bool,
    clean_text,
    cluster_name,
    normalize_taxonomy,
    validate_taxonomy,
)


SCHEMA_VERSION = "1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS app_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS taxonomy_features (
            taxonomy_id TEXT PRIMARY KEY,
            business_family TEXT NOT NULL,
            business_submodule TEXT NOT NULL,
            business_detail TEXT NOT NULL,
            sort_order INTEGER NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
            source TEXT NOT NULL DEFAULT 'taxonomy_pipeline',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (business_family, business_submodule, business_detail)
        );

        CREATE TABLE IF NOT EXISTS named_clusters (
            platform TEXT NOT NULL CHECK (platform IN ('android', 'ios')),
            cluster_id INTEGER NOT NULL,
            taxonomy_id TEXT,
            cluster_name TEXT NOT NULL,
            business_family TEXT NOT NULL,
            business_submodule TEXT NOT NULL,
            business_detail TEXT NOT NULL DEFAULT '',
            naming_confidence TEXT NOT NULL,
            naming_source TEXT NOT NULL,
            needs_review INTEGER NOT NULL CHECK (needs_review IN (0, 1)),
            score_share REAL,
            evidence_mass_coverage REAL,
            evidence_row_coverage REAL,
            supporting_evidence TEXT NOT NULL DEFAULT '',
            top_ngrams TEXT NOT NULL DEFAULT '',
            size INTEGER NOT NULL DEFAULT 0,
            share REAL NOT NULL DEFAULT 0,
            row_count_in_input INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (platform, cluster_id),
            FOREIGN KEY (taxonomy_id) REFERENCES taxonomy_features(taxonomy_id)
                ON UPDATE CASCADE ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS change_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            old_value TEXT NOT NULL,
            new_value TEXT NOT NULL,
            changed_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_taxonomy_active_path
            ON taxonomy_features(is_active, business_family, business_submodule, business_detail);
        CREATE INDEX IF NOT EXISTS idx_clusters_review
            ON named_clusters(platform, needs_review, naming_confidence);
        CREATE INDEX IF NOT EXISTS idx_clusters_taxonomy
            ON named_clusters(taxonomy_id);
        CREATE INDEX IF NOT EXISTS idx_audit_entity
            ON change_audit(entity_type, entity_key, changed_at);
        """
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_catalog(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    platforms = payload.get("platforms", [])
    if len(platforms) != 1:
        raise ValueError(f"shareholder catalog phải chứa đúng một platform: {path}")
    return {int(row["cluster_id"]): row for row in platforms[0].get("clusters", [])}


def initialize_database(db_path: Path, asset_dir: Path) -> dict[str, int]:
    """Create and seed the database once; existing business edits are preserved."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as connection:
        _create_schema(connection)
        current_version = connection.execute(
            "SELECT value FROM app_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if current_version and current_version[0] != SCHEMA_VERSION:
            raise ValueError(
                f"SQLite schema version {current_version[0]} != app version {SCHEMA_VERSION}"
            )
        taxonomy_count = connection.execute(
            "SELECT COUNT(*) FROM taxonomy_features"
        ).fetchone()[0]
        cluster_count = connection.execute("SELECT COUNT(*) FROM named_clusters").fetchone()[0]
        if taxonomy_count == 0 and cluster_count == 0:
            _seed_database(connection, asset_dir)
        elif taxonomy_count == 0 or cluster_count == 0:
            raise ValueError("SQLite database chỉ được seed một phần; cần kiểm tra lại file")
        connection.execute(
            "INSERT OR REPLACE INTO app_metadata(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        connection.execute(
            "INSERT OR IGNORE INTO app_metadata(key, value) VALUES('seed_source', ?)",
            ("taxonomy_naming",),
        )
    return database_stats(db_path)


def _seed_database(connection: sqlite3.Connection, asset_dir: Path) -> None:
    timestamp = utc_now()
    taxonomy_rows = _read_csv(asset_dir / "taxonomy_features.csv")
    mapping_rows = _read_csv(asset_dir / "cluster_mapping.csv")
    catalogs = {
        platform: _read_catalog(asset_dir / f"{platform}_shareholder_catalog.json")
        for platform in PLATFORMS
    }
    if not taxonomy_rows or not mapping_rows:
        raise ValueError("taxonomy hoặc cluster mapping nguồn đang rỗng")
    connection.executemany(
        """
        INSERT INTO taxonomy_features(
            taxonomy_id, business_family, business_submodule, business_detail,
            sort_order, is_active, source, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 1, 'taxonomy_pipeline', ?, ?)
        """,
        [
            (
                clean_text(row["taxonomy_id"]),
                clean_text(row["business_family"]),
                clean_text(row["business_submodule"]),
                clean_text(row["business_detail"]),
                index,
                timestamp,
                timestamp,
            )
            for index, row in enumerate(taxonomy_rows)
        ],
    )
    cluster_values = []
    for row in mapping_rows:
        platform = clean_text(row["platform"]).lower()
        cluster_id = int(row["cluster_id"])
        catalog = catalogs[platform].get(cluster_id)
        if catalog is None:
            raise ValueError(f"catalog thiếu cluster {(platform, cluster_id)}")
        taxonomy_id = clean_text(row.get("taxonomy_id", "")) or None
        cluster_values.append(
            (
                platform,
                cluster_id,
                taxonomy_id,
                clean_text(row["cluster_name"]),
                clean_text(row["business_family"]),
                clean_text(row["business_submodule"]),
                clean_text(row.get("business_detail", "")),
                clean_text(row["naming_confidence"]),
                clean_text(row["naming_source"]),
                int(as_bool(row["needs_review"])),
                _float_or_none(row.get("score_share")),
                _float_or_none(row.get("evidence_mass_coverage")),
                _float_or_none(row.get("evidence_row_coverage")),
                clean_text(row.get("supporting_evidence", "")),
                clean_text(row.get("top_ngrams", "")),
                int(catalog.get("size", 0)),
                float(catalog.get("share", 0) or 0),
                int(catalog.get("row_count_in_input", catalog.get("size", 0))),
                timestamp,
            )
        )
    connection.executemany(
        """
        INSERT INTO named_clusters(
            platform, cluster_id, taxonomy_id, cluster_name,
            business_family, business_submodule, business_detail,
            naming_confidence, naming_source, needs_review,
            score_share, evidence_mass_coverage, evidence_row_coverage,
            supporting_evidence, top_ngrams, size, share,
            row_count_in_input, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        cluster_values,
    )
    connection.execute(
        "INSERT INTO app_metadata(key, value) VALUES('seeded_at', ?)", (timestamp,)
    )


def _float_or_none(value: object) -> float | None:
    text = clean_text(value)
    return float(text) if text else None


def database_stats(db_path: Path) -> dict[str, int]:
    with connect(db_path) as connection:
        return {
            "taxonomy_features": int(
                connection.execute(
                    "SELECT COUNT(*) FROM taxonomy_features WHERE is_active = 1"
                ).fetchone()[0]
            ),
            "android_clusters": int(
                connection.execute(
                    "SELECT COUNT(*) FROM named_clusters WHERE platform = 'android'"
                ).fetchone()[0]
            ),
            "ios_clusters": int(
                connection.execute(
                    "SELECT COUNT(*) FROM named_clusters WHERE platform = 'ios'"
                ).fetchone()[0]
            ),
            "audit_events": int(
                connection.execute("SELECT COUNT(*) FROM change_audit").fetchone()[0]
            ),
        }


def load_taxonomy(db_path: Path, active_only: bool = True) -> pd.DataFrame:
    where = "WHERE is_active = 1" if active_only else ""
    with connect(db_path) as connection:
        return pd.read_sql_query(
            f"""
            SELECT taxonomy_id, business_family, business_submodule, business_detail
            FROM taxonomy_features
            {where}
            ORDER BY sort_order, taxonomy_id
            """,
            connection,
        )


def load_named_clusters(db_path: Path, platform: str) -> pd.DataFrame:
    if platform not in PLATFORMS:
        raise ValueError(platform)
    with connect(db_path) as connection:
        frame = pd.read_sql_query(
            """
            SELECT platform, cluster_id, taxonomy_id, cluster_name,
                   business_family, business_submodule, business_detail,
                   naming_confidence, naming_source, needs_review,
                   score_share, evidence_mass_coverage, evidence_row_coverage,
                   supporting_evidence, top_ngrams, size, share,
                   row_count_in_input, updated_at
            FROM named_clusters
            WHERE platform = ?
            ORDER BY cluster_id
            """,
            connection,
            params=(platform,),
        )
    frame["taxonomy_id"] = frame["taxonomy_id"].fillna("")
    frame["needs_review"] = frame["needs_review"].astype(bool)
    return frame


def save_taxonomy(db_path: Path, edited: pd.DataFrame) -> dict[str, int]:
    """Persist valid editor changes and propagate renamed taxonomy nodes to mappings."""

    frame = normalize_taxonomy(edited)
    errors = validate_taxonomy(frame)
    if errors:
        raise ValueError(" ".join(errors))
    timestamp = utc_now()
    inserted = updated = deactivated = propagated = 0
    with connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current_rows = {
            row["taxonomy_id"]: dict(row)
            for row in connection.execute("SELECT * FROM taxonomy_features")
        }
        custom_counter = _next_custom_counter(current_rows)
        normalized_rows: list[dict[str, str]] = []
        for row in frame.to_dict("records"):
            taxonomy_id = clean_text(row["taxonomy_id"])
            if not taxonomy_id:
                taxonomy_id = f"USR.{custom_counter:06d}"
                custom_counter += 1
            normalized_rows.append(
                {
                    "taxonomy_id": taxonomy_id,
                    "business_family": clean_text(row["business_family"]),
                    "business_submodule": clean_text(row["business_submodule"]),
                    "business_detail": clean_text(row["business_detail"]),
                }
            )
        ids = [row["taxonomy_id"] for row in normalized_rows]
        paths = [
            (row["business_family"], row["business_submodule"], row["business_detail"])
            for row in normalized_rows
        ]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("taxonomy_id hoặc đường dẫn taxonomy bị trùng")

        incoming_ids = set(ids)
        for sort_order, row in enumerate(normalized_rows):
            taxonomy_id = row["taxonomy_id"]
            old = current_rows.get(taxonomy_id)
            new_path = {
                "business_family": row["business_family"],
                "business_submodule": row["business_submodule"],
                "business_detail": row["business_detail"],
            }
            if old is None:
                connection.execute(
                    """
                    INSERT INTO taxonomy_features(
                        taxonomy_id, business_family, business_submodule,
                        business_detail, sort_order, is_active, source,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, 'business_review', ?, ?)
                    """,
                    (
                        taxonomy_id,
                        row["business_family"],
                        row["business_submodule"],
                        row["business_detail"],
                        sort_order,
                        timestamp,
                        timestamp,
                    ),
                )
                _audit(connection, "taxonomy_feature", taxonomy_id, {}, new_path, timestamp)
                inserted += 1
                continue
            old_path = {
                "business_family": old["business_family"],
                "business_submodule": old["business_submodule"],
                "business_detail": old["business_detail"],
            }
            changed = old_path != new_path or not bool(old["is_active"])
            connection.execute(
                """
                UPDATE taxonomy_features
                SET business_family = ?, business_submodule = ?, business_detail = ?,
                    sort_order = ?, is_active = 1,
                    source = CASE WHEN ? THEN 'business_review' ELSE source END,
                    updated_at = CASE WHEN ? THEN ? ELSE updated_at END
                WHERE taxonomy_id = ?
                """,
                (
                    row["business_family"],
                    row["business_submodule"],
                    row["business_detail"],
                    sort_order,
                    int(changed),
                    int(changed),
                    timestamp,
                    taxonomy_id,
                ),
            )
            if changed:
                result = connection.execute(
                    """
                    UPDATE named_clusters
                    SET business_family = ?, business_submodule = ?, business_detail = ?,
                        cluster_name = ?, naming_source = 'business_review_taxonomy_update',
                        needs_review = 1, updated_at = ?
                    WHERE taxonomy_id = ?
                    """,
                    (
                        row["business_family"],
                        row["business_submodule"],
                        row["business_detail"],
                        cluster_name(*new_path.values()),
                        timestamp,
                        taxonomy_id,
                    ),
                )
                propagated += result.rowcount
                _audit(connection, "taxonomy_feature", taxonomy_id, old_path, new_path, timestamp)
                updated += 1

        for taxonomy_id, old in current_rows.items():
            if bool(old["is_active"]) and taxonomy_id not in incoming_ids:
                connection.execute(
                    "UPDATE taxonomy_features SET is_active = 0, updated_at = ? WHERE taxonomy_id = ?",
                    (timestamp, taxonomy_id),
                )
                result = connection.execute(
                    """
                    UPDATE named_clusters
                    SET naming_source = 'business_review_taxonomy_removed',
                        needs_review = 1, updated_at = ?
                    WHERE taxonomy_id = ?
                    """,
                    (timestamp, taxonomy_id),
                )
                propagated += result.rowcount
                _audit(
                    connection,
                    "taxonomy_feature",
                    taxonomy_id,
                    {"is_active": True},
                    {"is_active": False},
                    timestamp,
                )
                deactivated += 1
    return {
        "inserted": inserted,
        "updated": updated,
        "deactivated": deactivated,
        "propagated_clusters": propagated,
    }


def _next_custom_counter(rows: dict[str, dict[str, Any]]) -> int:
    values = []
    for taxonomy_id in rows:
        if taxonomy_id.startswith("USR.") and taxonomy_id[4:].isdigit():
            values.append(int(taxonomy_id[4:]))
    return max(values, default=0) + 1


def update_named_cluster(
    db_path: Path,
    platform: str,
    cluster_id: int,
    *,
    family: str,
    submodule: str,
    detail: str,
    confidence: str,
    needs_review: bool,
) -> bool:
    """Persist one business decision and record an audit event."""

    if platform not in PLATFORMS:
        raise ValueError(platform)
    family, submodule, detail = map(clean_text, (family, submodule, detail))
    timestamp = utc_now()
    with connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        old_row = connection.execute(
            "SELECT * FROM named_clusters WHERE platform = ? AND cluster_id = ?",
            (platform, int(cluster_id)),
        ).fetchone()
        if old_row is None:
            raise KeyError((platform, cluster_id))
        taxonomy_id: str | None = None
        if family != "Chưa phân loại":
            if not family:
                raise ValueError("Business Family không được để trống")
            if not submodule:
                # Cascading Streamlit selects rerun after each level. Persist the
                # intermediate family selection so the next dropdown can be chosen,
                # but keep the row visibly pending review.
                detail = ""
                needs_review = True
            elif detail:
                taxonomy = connection.execute(
                    """
                    SELECT taxonomy_id FROM taxonomy_features
                    WHERE is_active = 1 AND business_family = ?
                      AND business_submodule = ? AND business_detail = ?
                    """,
                    (family, submodule, detail),
                ).fetchone()
                if taxonomy is None:
                    raise ValueError("Business Detail không tồn tại trong taxonomy đang active")
                taxonomy_id = str(taxonomy[0])
            else:
                pair = connection.execute(
                    """
                    SELECT 1 FROM taxonomy_features
                    WHERE is_active = 1 AND business_family = ? AND business_submodule = ?
                    LIMIT 1
                    """,
                    (family, submodule),
                ).fetchone()
                if pair is None:
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
            key: (bool(old_row[key]) if key == "needs_review" else clean_text(old_row[key]))
            for key in new_values
        }
        if old_values == new_values:
            return False
        connection.execute(
            """
            UPDATE named_clusters
            SET taxonomy_id = ?, cluster_name = ?, business_family = ?,
                business_submodule = ?, business_detail = ?,
                naming_confidence = ?, naming_source = 'business_review',
                needs_review = ?, updated_at = ?
            WHERE platform = ? AND cluster_id = ?
            """,
            (
                taxonomy_id,
                new_values["cluster_name"],
                family,
                submodule,
                detail,
                new_values["naming_confidence"],
                int(needs_review),
                timestamp,
                platform,
                int(cluster_id),
            ),
        )
        _audit(
            connection,
            "named_cluster",
            f"{platform}:{cluster_id}",
            old_values,
            new_values,
            timestamp,
        )
    return True


def load_recent_audit(db_path: Path, limit: int = 20) -> pd.DataFrame:
    with connect(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT entity_type, entity_key, old_value, new_value, changed_at
            FROM change_audit
            ORDER BY audit_id DESC
            LIMIT ?
            """,
            connection,
            params=(int(limit),),
        )


def _audit(
    connection: sqlite3.Connection,
    entity_type: str,
    entity_key: str,
    old_value: dict[str, Any],
    new_value: dict[str, Any],
    timestamp: str,
) -> None:
    connection.execute(
        """
        INSERT INTO change_audit(entity_type, entity_key, old_value, new_value, changed_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            entity_type,
            entity_key,
            json.dumps(old_value, ensure_ascii=False, sort_keys=True),
            json.dumps(new_value, ensure_ascii=False, sort_keys=True),
            timestamp,
        ),
    )
