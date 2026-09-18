PRAGMA foreign_keys = ON;

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
