from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from refresh_clusters import DEFAULT_TAXONOMY, source_files  # noqa: E402

# Tests publish the newest local pipeline run into a throwaway SQLite database, the same way
# refresh_clusters.py does; the app itself never reads these files.
SCORES_RUN = max(
    (REPO_ROOT / "output" / "scores").glob("*/*/taxonomy_naming/cluster_mapping.csv"),
    key=lambda path: path.stat().st_mtime,
).parents[1]
TRAINING_RUN = REPO_ROOT / "output" / "partitioned_runs" / SCORES_RUN.parent.name / SCORES_RUN.name
_STAGING = tempfile.TemporaryDirectory()
ASSET_DIR = Path(_STAGING.name)
for _name, _path in source_files(TRAINING_RUN, SCORES_RUN).items():
    shutil.copy2(_path, ASSET_DIR / _name)
shutil.copy2(DEFAULT_TAXONOMY, ASSET_DIR / "taxonomy_features.csv")
# Cluster counts change with every retrained model; take them from the mapping.
MAPPING_COUNTS = (
    pd.read_csv(ASSET_DIR / "cluster_mapping.csv", encoding="utf-8-sig")["platform"]
    .value_counts()
    .to_dict()
)

from core import (  # noqa: E402
    assignment_status,
    catalog_row,
    cluster_ngrams,
    export_named_clusters_csv,
    export_taxonomy_csv,
    filter_named_clusters,
    load_platform_evidence,
    normalize_taxonomy,
    taxonomy_choices,
    validate_taxonomy,
)
from database import (  # noqa: E402
    connect,
    database_stats,
    initialize_database,
    load_cluster_evidence,
    load_model_version,
    publish_evidence,
    replace_named_clusters,
    load_named_clusters,
    load_recent_audit,
    load_taxonomy,
    save_taxonomy,
    update_named_cluster,
)


class TaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy = pd.DataFrame(
            [
                ["M01.01.01", "Home", "Thanh toán", "QR"],
                ["M01.01.02", "Home", "Thanh toán", "Trả trước"],
                ["M02.01.01", "Tài khoản", "Hồ sơ", "Xem hồ sơ"],
            ],
            columns=[
                "taxonomy_id",
                "business_family",
                "business_submodule",
                "business_detail",
            ],
        )

    def test_choices_and_validation(self) -> None:
        normalized = normalize_taxonomy(self.taxonomy)
        self.assertFalse(validate_taxonomy(normalized))
        choices = taxonomy_choices(normalized)
        self.assertEqual(choices["families"], ["Home", "Tài khoản"])
        self.assertEqual(choices["details"][("Home", "Thanh toán")], ["QR", "Trả trước"])
        self.assertEqual(choices["path_to_id"][("Home", "Thanh toán", "QR")], "M01.01.01")
        self.assertTrue(assignment_status(normalized, "Home", "Thanh toán", "QR")[0])
        self.assertFalse(assignment_status(normalized, "Home", "Hồ sơ", "")[0])

    def test_validation_rejects_incomplete_and_duplicate_paths(self) -> None:
        invalid = pd.concat(
            [
                self.taxonomy,
                pd.DataFrame(
                    [["", "Home", "Thanh toán", "QR"], ["", "", "Thiếu", ""]],
                    columns=self.taxonomy.columns,
                ),
            ],
            ignore_index=True,
        )
        errors = validate_taxonomy(invalid)
        self.assertTrue(any("trùng" in error for error in errors))
        self.assertTrue(any("Thiếu Business Family" in error for error in errors))

    def test_export_has_utf8_bom(self) -> None:
        self.assertTrue(export_taxonomy_csv(self.taxonomy).startswith(b"\xef\xbb\xbf"))


class EvidenceTests(unittest.TestCase):
    def test_both_platforms_have_eight_ngrams_per_cluster(self) -> None:
        for platform, expected_clusters in MAPPING_COUNTS.items():
            evidence = load_platform_evidence(ASSET_DIR, platform)
            self.assertEqual(len(evidence.catalog), expected_clusters)
            counts = evidence.ngrams.groupby("cluster").size()
            self.assertTrue((counts == 8).all())
            first_cluster = int(evidence.catalog.iloc[0].cluster_id)
            self.assertEqual(len(cluster_ngrams(evidence.ngrams, first_cluster)), 8)


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "review.sqlite"
        initialize_database(self.db_path, ASSET_DIR, "seed_model", "test")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_bootstrap_contract_and_persistence(self) -> None:
        stats = database_stats(self.db_path)
        self.assertEqual(stats["taxonomy_features"], 486)
        self.assertEqual(stats["android_clusters"], MAPPING_COUNTS["android"])
        self.assertEqual(stats["ios_clusters"], MAPPING_COUNTS["ios"])
        self.assertEqual(len(load_taxonomy(self.db_path)), 486)
        self.assertFalse(load_named_clusters(self.db_path, "android").cluster_id.duplicated().any())
        # Reinitialization must preserve the same database instead of reseeding it.
        self.assertEqual(initialize_database(self.db_path, ASSET_DIR), stats)
        # The app initializes without files and must not need them.
        self.assertEqual(initialize_database(self.db_path), stats)
        self.assertEqual(load_model_version(self.db_path), "seed_model")
        self.assertEqual(load_recent_audit(self.db_path).iloc[0].entity_type, "database_seed")

    def test_app_never_seeds_from_files(self) -> None:
        empty = Path(self.tempdir.name) / "empty.sqlite"
        with self.assertRaisesRegex(ValueError, "chưa có dữ liệu"):
            initialize_database(empty)

    def test_evidence_round_trips_through_database(self) -> None:
        for platform in MAPPING_COUNTS:
            stored = load_cluster_evidence(self.db_path, platform)
            source = load_platform_evidence(ASSET_DIR, platform)
            self.assertEqual(len(stored.catalog), len(source.catalog))
            cluster_id = int(source.catalog.iloc[-1].cluster_id)
            self.assertEqual(
                catalog_row(stored.catalog, cluster_id)["medoid_path"],
                catalog_row(source.catalog, cluster_id)["medoid_path"],
            )
            self.assertEqual(
                cluster_ngrams(stored.ngrams, cluster_id).ngram.tolist(),
                cluster_ngrams(source.ngrams, cluster_id).ngram.tolist(),
            )

    def test_evidence_only_keeps_reviewed_labels(self) -> None:
        update_named_cluster(
            self.db_path,
            "android",
            0,
            family="Chưa phân loại",
            submodule="",
            detail="",
            confidence="high",
            needs_review=False,
        )
        with connect(self.db_path) as connection:
            connection.execute("DELETE FROM cluster_evidence")
        with self.assertRaisesRegex(ValueError, "cluster_evidence"):
            initialize_database(self.db_path)
        counts = publish_evidence(self.db_path, ASSET_DIR, "seed_model", "test")
        self.assertEqual(counts["android_evidence"], MAPPING_COUNTS["android"])
        initialize_database(self.db_path)
        row = load_named_clusters(self.db_path, "android").set_index("cluster_id").loc[0]
        self.assertEqual(row.business_family, "Chưa phân loại")
        self.assertEqual(load_recent_audit(self.db_path).iloc[0].entity_type, "evidence_refresh")

    def test_replace_named_clusters_keeps_taxonomy_and_audit(self) -> None:
        android = load_named_clusters(self.db_path, "android")
        cluster_id = int(android.iloc[0].cluster_id)
        update_named_cluster(
            self.db_path,
            "android",
            cluster_id,
            family="Chưa phân loại",
            submodule="",
            detail="",
            confidence="high",
            needs_review=False,
        )
        counts = replace_named_clusters(self.db_path, ASSET_DIR, "test_model")
        self.assertEqual(counts, MAPPING_COUNTS)
        self.assertEqual(len(load_taxonomy(self.db_path)), 486)
        refreshed = load_named_clusters(self.db_path, "android").set_index("cluster_id")
        self.assertEqual(refreshed.loc[cluster_id, "naming_confidence"], android.iloc[0].naming_confidence)
        audit = load_recent_audit(self.db_path, limit=5)
        self.assertEqual(audit.iloc[0].entity_type, "cluster_refresh")
        self.assertEqual(audit.iloc[1].entity_type, "named_cluster")

    def test_taxonomy_edit_propagates_to_named_clusters(self) -> None:
        android = load_named_clusters(self.db_path, "android")
        mapped = android.loc[android["taxonomy_id"].ne("")].iloc[0]
        taxonomy_id = mapped.taxonomy_id
        taxonomy = load_taxonomy(self.db_path)
        mask = taxonomy["taxonomy_id"].eq(taxonomy_id)
        taxonomy.loc[mask, "business_detail"] = taxonomy.loc[mask, "business_detail"] + " QA"
        result = save_taxonomy(self.db_path, taxonomy)
        self.assertEqual(result["updated"], 1)
        self.assertGreaterEqual(result["propagated_clusters"], 1)
        refreshed = load_named_clusters(self.db_path, "android")
        changed = refreshed.loc[refreshed["cluster_id"].eq(int(mapped.cluster_id))].iloc[0]
        self.assertTrue(changed.business_detail.endswith(" QA"))
        self.assertEqual(changed.naming_source, "business_review_taxonomy_update")
        self.assertTrue(bool(changed.needs_review))
        self.assertEqual(len(load_recent_audit(self.db_path)), 2)

    def test_new_taxonomy_row_gets_stable_custom_id(self) -> None:
        taxonomy = load_taxonomy(self.db_path)
        new_row = pd.DataFrame(
            [["", "Family thử nghiệm", "Module thử nghiệm", "Detail thử nghiệm"]],
            columns=taxonomy.columns,
        )
        result = save_taxonomy(self.db_path, pd.concat([taxonomy, new_row], ignore_index=True))
        self.assertEqual(result["inserted"], 1)
        saved = load_taxonomy(self.db_path)
        custom = saved.loc[saved.business_family.eq("Family thử nghiệm")].iloc[0]
        self.assertEqual(custom.taxonomy_id, "USR.000001")

    def test_cluster_dropdown_selection_updates_sqlite_and_audit(self) -> None:
        taxonomy = load_taxonomy(self.db_path)
        target = taxonomy.iloc[0]
        changed = update_named_cluster(
            self.db_path,
            "android",
            0,
            family=target.business_family,
            submodule=target.business_submodule,
            detail=target.business_detail,
            confidence="high",
            needs_review=False,
        )
        self.assertTrue(changed)
        row = load_named_clusters(self.db_path, "android").loc[lambda df: df.cluster_id.eq(0)].iloc[0]
        self.assertEqual(row.taxonomy_id, target.taxonomy_id)
        self.assertEqual(row.naming_source, "business_review")
        self.assertFalse(bool(row.needs_review))
        self.assertEqual(len(load_recent_audit(self.db_path)), 2)

    def test_filter_and_mapping_export(self) -> None:
        frame = load_named_clusters(self.db_path, "ios")
        sample = frame.iloc[0]
        filtered = filter_named_clusters(
            frame,
            review_values=[bool(sample.needs_review)],
            confidence_values=[sample.naming_confidence],
            query=str(sample.cluster_id),
        )
        self.assertIn(int(sample.cluster_id), filtered.cluster_id.tolist())
        self.assertTrue(export_named_clusters_csv(frame).startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main()
