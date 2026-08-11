"""Export the C-first, B-fallback inference bundle used by the mobile apps."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from export_scorer_to_onnx import (
    REPO_ROOT,
    JourneyScorer,
    _install_pickle_compatibility,
    export_centroid_onnx,
    export_friction_config,
    export_markov,
    export_preprocessing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=["android", "ios"])
    parser.add_argument("--primary-model-dir", required=True, help="C run directory")
    parser.add_argument("--secondary-model-dir", required=True, help="B run directory")
    parser.add_argument("--postprocess-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mapping_for_namespace(names: dict, namespace: str) -> dict:
    rows = []
    for item in names["clusters"]:
        if item["namespace"] != namespace:
            continue
        rows.append({
            "cluster": int(item["cluster"]),
            "effective_cluster_key": item["effective_cluster_key"],
            "class_group_code": item["business_family_code"],
            "class_group": item["business_family"],
            "class_code": item["effective_cluster_key"].replace(":", "_").lower(),
            "class_name": item["cluster_name_vi"],
            "cluster_name": item["cluster_name_vi"],
            "cluster_name_en": item["cluster_name_en"],
            "naming_confidence": item["naming_confidence"],
        })
    return {"schema_version": "3.0", "namespace": namespace, "classes": rows}


def main() -> int:
    args = parse_args()
    primary_dir = REPO_ROOT / args.primary_model_dir
    secondary_dir = REPO_ROOT / args.secondary_model_dir
    post_dir = REPO_ROOT / args.postprocess_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _install_pickle_compatibility()

    names_path = post_dir / f"{args.platform}_hierarchical_cluster_names.json"
    names = load_json(names_path)
    scorers = {
        "c": JourneyScorer.load(primary_dir / f"{args.platform}_journey_scorer.pkl"),
        "b": JourneyScorer.load(secondary_dir / f"{args.platform}_journey_scorer.pkl"),
    }
    for prefix, scorer in scorers.items():
        export_centroid_onnx(scorer, output_dir / f"{prefix}_journey_classifier.onnx", args.opset)
        export_preprocessing(scorer, output_dir / f"{prefix}_preprocessing.json", args.platform)
        export_markov(scorer, output_dir / f"{prefix}_markov.json")
        (output_dir / f"{prefix}_class_mapping.json").write_text(
            json.dumps(mapping_for_namespace(names, prefix.upper()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    export_friction_config(scorers["c"], output_dir / "friction_config.json")
    shutil.copyfile(names_path, output_dir / "hierarchical_cluster_names.json")
    c_thresholds = load_json(post_dir / f"{args.platform}_C_cluster_thresholds.json")
    b_thresholds = load_json(post_dir / f"{args.platform}_B_cluster_thresholds.json")
    post_config = load_json(post_dir / "postprocess_config.json")
    config = {
        "schema_version": "3.0",
        "strategy": "C_primary_then_B_noise_fallback",
        "min_distance_margin": post_config["thresholds"]["min_distance_margin"],
        "c_distance_limits": c_thresholds["distance_limits"],
        "b_distance_limits": b_thresholds["distance_limits"],
        "b_markov_limits": b_thresholds["markov_limits"],
    }
    (output_dir / "hierarchical_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files = [
        *[f"{prefix}_{name}" for prefix in ("c", "b") for name in (
            "journey_classifier.onnx", "preprocessing.json", "markov.json", "class_mapping.json"
        )],
        "friction_config.json", "hierarchical_cluster_names.json", "hierarchical_config.json",
    ]
    manifest = {
        "schema_version": "3.0",
        "platform": args.platform,
        "strategy": config["strategy"],
        "primary_namespace": "C",
        "fallback_namespace": "B",
        "files": files,
        "sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in files},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"hierarchical mobile bundle -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
