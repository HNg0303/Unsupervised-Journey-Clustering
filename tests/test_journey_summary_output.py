from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from journey_clustering import cluster_mapping, naming  # noqa: E402
from journey_clustering.cli import cluster_taxonomy, infer  # noqa: E402


def scored_frame() -> pd.DataFrame:
    """One journey as JourneyScorer.score_prepared returns it, plus run metadata."""
    return pd.DataFrame([{
        "journey_id": "p::J000001", "session_id": "s1", "device_id": "d1", "customer_id": "c1",
        "os": "android", "start_ts": pd.Timestamp("2026-08-10T15:33:15Z"),
        "end_ts": pd.Timestamp("2026-08-10T15:34:30Z"), "boundary_reason": "session_start",
        "n_events_raw": 9, "n_events_final": 5, "n_unique_tokens": 4, "n_dropped_screens": 2,
        "n_dedup_removed": 1, "n_loop_removed": 1, "action_ratio": 0.4, "back_rate": 0.2,
        "revisit_ratio": 0.2, "span_seconds": 75.0, "median_gap_s": 2.0, "p90_gap_s": 9.0,
        "max_gap_s": 30.0, "entry_token": "view@home", "exit_token": "view@pay",
        "cluster": 7, "nearest_cluster": 7, "distance_to_centroid": 0.2, "distance_limit": 0.6,
        "markov_logprob": -3.1, "geometric_anomaly": False, "generative_anomaly": True,
        "severe_anomaly": False, "sequence": "view@home -> view@pay",
        "friction_flags": "excessive_back|improbable_transitions", "next_action": "view@done",
        "next_action_share": 0.5, "platform": "android", "model_version": "m1",
        "scored_at": pd.Timestamp("2026-09-29T12:00:00Z"),
    }])


class OutputSchemaTest(unittest.TestCase):
    def test_keeps_only_journey_summary_columns(self) -> None:
        out = infer.to_output_schema(scored_frame())
        self.assertEqual(list(out.columns), list(infer.OUTPUT_COLUMNS))
        self.assertEqual(out.loc[0, "cluster_id"], 7)
        # the two model flags are already the anomaly booleans
        self.assertEqual(out.loc[0, "friction_flags"], "excessive_back")

    def test_empty_partition_has_the_same_columns(self) -> None:
        out = infer.to_output_schema(pd.DataFrame(columns=["journey_id"]))
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), list(infer.OUTPUT_COLUMNS))


def mapping_row(platform: str, cluster_id: int, name: str) -> dict[str, object]:
    return {
        "platform": platform, "cluster_id": cluster_id, "taxonomy_id": "", "cluster_name": name,
        "module": name, "submodule": "", "detail": "", "business_family": name,
        "business_submodule": "", "business_detail": "", "naming_confidence": "high",
        "naming_source": "validated_taxonomy_detail", "needs_review": "false", "score_share": 1.0,
        "evidence_mass_coverage": 1.0, "evidence_row_coverage": 1.0,
        "supporting_evidence": "", "top_ngrams": f"ngram-{cluster_id}",
    }


class CatalogScoresTest(unittest.TestCase):
    def test_counts_rows_and_writes_names_once_per_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "android_scores.csv"
            infer.to_output_schema(pd.concat([scored_frame()] * 3)).to_csv(scores, index=False)
            mapping = [mapping_row("android", -1, "Noise"), mapping_row("android", 7, "Pay")]
            result = naming.catalog_scores(
                scores, root / "android_cluster_mapping.csv",
                root / "android_taxonomy_shareholder_catalog.json", "android", mapping, False,
            )
            self.assertEqual(result["rows"], 3)
            with (root / "android_cluster_mapping.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual([row["cluster_id"] for row in csv.DictReader(handle)], ["-1", "7"])
            catalog = json.loads((root / "android_taxonomy_shareholder_catalog.json").read_text())
            sizes = {row["cluster_id"]: row["size"] for row in catalog["platforms"][0]["clusters"]}
            self.assertEqual(sizes, {-1: 0, 7: 3})


class ClusterTaxonomyTest(unittest.TestCase):
    catalog = {
        -1: {"size": 10, "top_mass_ngrams": "ngram--1"},
        7: {"size": 3, "top_mass_ngrams": "ngram-7"},
    }

    def named(self, **overrides: str) -> list[dict[str, str]]:
        rows = []
        for cluster_id, size in ((-1, "10"), (7, "3")):
            row = {key: str(value) for key, value in mapping_row("android", cluster_id, "Pay").items()}
            row.update(size=size, updated_at="2026-10-09T03:00:00+00:00", needs_review="true")
            rows.append(row)
        rows[1].update(overrides)
        return rows

    def test_aligned_names_become_table_rows(self) -> None:
        named = self.named()
        self.assertEqual(cluster_taxonomy.alignment_errors(named, self.catalog, "android"), [])
        rows = cluster_taxonomy.taxonomy_rows(named, "m1", "android")
        self.assertEqual(list(rows[0]), list(cluster_taxonomy.TAXONOMY_COLUMNS))
        self.assertEqual((rows[1]["model_version"], rows[1]["cluster_id"], rows[1]["needs_review"]), ("m1", 7, 1))
        self.assertEqual(rows[1]["named_at"], "2026-10-09 03:00:00")
        # cluster -1 keeps the noise name whatever the reviewer saved
        self.assertEqual((rows[0]["cluster_name"], rows[0]["naming_source"]), (
            cluster_taxonomy.NOISE_NAME["cluster_name"], "noise_cluster"))

    def test_names_from_another_model_are_rejected(self) -> None:
        errors = cluster_taxonomy.alignment_errors(
            self.named(size="4", top_ngrams="other"), self.catalog, "android"
        )
        self.assertEqual(len(errors), 2)
        missing = cluster_taxonomy.alignment_errors(self.named()[:1], self.catalog, "android")
        self.assertIn("no name", missing[0])


class MapClusterNamesTest(unittest.TestCase):
    def write_taxonomy(self, root: Path) -> Path:
        named = ClusterTaxonomyTest().named()
        rows = cluster_taxonomy.taxonomy_rows(named, "m1", "android")
        rows += cluster_taxonomy.taxonomy_rows(named, "m1", "ios")
        rows[1].update(cluster_name="Thanh toán | Hóa đơn", business_family="Thanh toán")
        path = root / "journey_cluster_taxonomy.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_names_every_journey_of_the_platform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            taxonomy = cluster_mapping.load_cluster_taxonomy(
                self.write_taxonomy(root), platform="android", model_version="m1")
            frame = infer.to_output_schema(pd.concat([scored_frame()] * 3, ignore_index=True))
            frame.loc[2, "cluster_id"] = 99  # a cluster the names do not know
            frame["customer_id"] = "6039276"
            scores = root / "android_scores.csv"
            frame.to_csv(scores, index=False)
            result = cluster_mapping.map_scores_file(scores, root / "named.csv", taxonomy, chunksize=2)
            self.assertEqual((result["rows"], result["unknown_cluster_rows"]), (3, 1))
            out = pd.read_csv(root / "named.csv", dtype={"customer_id": str})
            position = list(out.columns).index("cluster_id")
            self.assertEqual(list(out.columns[position + 1:position + 9]),
                             list(cluster_mapping.TAXONOMY_NAME_COLUMNS))
            self.assertEqual(list(out["business_family"]), ["Thanh toán", "Thanh toán", "Chưa phân loại"])
            self.assertEqual(out.loc[0, "customer_id"], "6039276")

    def test_old_exports_and_other_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_taxonomy(Path(directory))
            taxonomy = cluster_mapping.load_cluster_taxonomy(path, platform="ios", model_version="m1")
            old = scored_frame().assign(cluster_name="old name")
            named = cluster_mapping.apply_cluster_taxonomy(old, taxonomy)
            self.assertEqual(named.loc[0, "cluster_name"], "Pay")
            self.assertEqual(list(named.columns).count("cluster_name"), 1)
            with self.assertRaises(ValueError):
                cluster_mapping.apply_cluster_taxonomy(old.assign(model_version="m2"), taxonomy)
            with self.assertRaises(ValueError):
                cluster_mapping.load_cluster_taxonomy(path, platform="ios", model_version="m2")


if __name__ == "__main__":
    unittest.main()
