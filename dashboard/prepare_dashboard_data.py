"""Register the selected train/inference/post-analysis products for Streamlit.

This script is intentionally lightweight. It only registers already-produced
dashboard artifacts; raw EDA and presentation aggregation are outside the core
training/inference command set.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

BUNDLE_DIR = ROOT / "output" / "scores" / "pca48_ngrams12_500"
TRAIN_RUN_DIR = ROOT / "output" / "partitioned_runs" / "pca48_svd48_ngrams12_500k" / "latest"
SUMMARY_DIR = BUNDLE_DIR / "html_dashboard_summary"
POST_ANALYSIS_DIR = BUNDLE_DIR / "post_analysis"
CACHE_DIR = ROOT / "output" / "dashboard_cache"


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_train_cache() -> dict:
    payload = {"run_slug": TRAIN_RUN_DIR.parent.name, "source": str(TRAIN_RUN_DIR), "platforms": {}}
    for platform in ("android", "ios"):
        model = TRAIN_RUN_DIR / platform
        cfg = read_json(model / f"{platform}_run_config.json")
        inspection = BUNDLE_DIR / platform / "model_version=latest" / f"platform={platform}" / "inspection"
        summary = read_json(inspection / f"{platform}_summary.json")
        payload["platforms"][platform] = {"run_config": cfg, "summary": summary, "holdout": {}, "journeys": {}}
    return payload


def build_threshold_cache() -> dict:
    try:
        from journey_clustering.score import JourneyScorer
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency for registration
        return {"error": f"scorer dependencies unavailable: {exc}"}
    thresholds = {}
    for platform in ("android", "ios"):
        path = TRAIN_RUN_DIR / platform / f"{platform}_journey_scorer.pkl"
        if not path.exists():
            continue
        try:
            scorer = JourneyScorer.load(path)
            value = scorer.thresholds
            thresholds[platform] = dataclasses.asdict(value) if dataclasses.is_dataclass(value) else dict(vars(value))
        except Exception as exc:  # noqa: BLE001 - cache preparation should remain diagnosable
            thresholds[platform] = {"error": str(exc)}
    return thresholds


def build_bundle_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    POST_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    train = build_train_cache()
    (CACHE_DIR / "train_run.json").write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")
    (CACHE_DIR / "thresholds.json").write_text(json.dumps(build_threshold_cache(), ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": "dashboard-registration-v2",
        "bundle": str(BUNDLE_DIR.relative_to(ROOT)),
        "train_run": str(TRAIN_RUN_DIR.relative_to(ROOT)),
        "naming": str((BUNDLE_DIR / "Cluster_naming.csv").relative_to(ROOT)),
        "summary_dir": str(SUMMARY_DIR.relative_to(ROOT)),
        "raw_eda": str((POST_ANALYSIS_DIR / "eda" / "eda_summary.json").relative_to(ROOT)),
        "train_output": str((POST_ANALYSIS_DIR / "train_output_summary.json").relative_to(ROOT)),
        "customer_analysis": str((BUNDLE_DIR / "shareholder_analysis").relative_to(ROOT)),
        "platforms": {
            platform: {
                "inference_partitions": len(list((BUNDLE_DIR / platform / "model_version=latest" / f"platform={platform}").glob("platform=*.parquet"))),
                "inspection_summary": str((BUNDLE_DIR / platform / "model_version=latest" / f"platform={platform}" / "inspection" / f"{platform}_summary.json").relative_to(ROOT)),
            }
            for platform in ("android", "ios")
        },
        "contract": {
            "raw_eda": "optional precomputed post_analysis/eda/eda_summary.json",
            "training": "one row per modelled journey plus *_scored_holdout.csv",
            "inference": "one row per scored journey in partitioned output",
            "named_anchor": "*_all_named.csv",
            "dashboard_aggregates": "html_dashboard_summary/*.csv",
            "dashboard_examples": "html_dashboard_summary/journey_examples.csv (bounded audit only)",
            "customer_post_analysis": "shareholder_analysis/* from named anchors",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (CACHE_DIR / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"registered dashboard bundle -> {BUNDLE_DIR}")
    print("raw EDA is optional and must be supplied as a precomputed artifact")


if __name__ == "__main__":
    build_bundle_cache()
