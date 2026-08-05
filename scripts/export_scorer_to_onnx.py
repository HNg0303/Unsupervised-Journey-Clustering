"""Export the portable part of JourneyScorer as an ONNX mobile bundle.

The custom pandas journey preparation is intentionally exported as parameters
in preprocessing.json. The ONNX graph receives the final feature vector and
performs nearest-centroid assignment plus the learned distance rejection.

    python scripts/export_scorer_to_onnx.py --platform android
    python scripts/export_scorer_to_onnx.py --platform ios
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based.score import JourneyScorer  # noqa: E402


def _install_pickle_compatibility() -> None:
    """Allow a Python 3.12 exporter to read pathlib objects pickled by 3.13."""
    if "pathlib._local" not in sys.modules:
        module = types.ModuleType("pathlib._local")
        module.WindowsPath = pathlib.WindowsPath
        module.PosixPath = pathlib.PosixPath
        module.Path = pathlib.Path
        sys.modules["pathlib._local"] = module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=["android", "ios"])
    parser.add_argument("--model-dir", default="output/clusters")
    parser.add_argument("--output-dir", help="default: output/mobile/<platform>")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def _as_list(value: np.ndarray) -> list:
    return np.asarray(value).tolist()


def export_preprocessing(scorer: JourneyScorer, destination: Path, platform: str) -> None:
    vectorizer = scorer.vectorizer
    if vectorizer.svd is None:
        raise ValueError("scorer vectorizer has no fitted SVD")
    vocabulary = sorted(vectorizer.tfidf.vocabulary_.items(), key=lambda item: item[1])
    payload = {
        "schema_version": "1.0",
        "platform": platform,
        "contract": "ONNX input is the concatenated sequence and numeric feature vector.",
        "ngram_range": list(vectorizer.tfidf.ngram_range),
        "sublinear_tf": bool(vectorizer.tfidf.sublinear_tf),
        "lowercase": bool(vectorizer.tfidf.lowercase),
        "vocabulary": [term for term, _ in vocabulary],
        "idf": _as_list(vectorizer.tfidf.idf_),
        "svd_components": _as_list(vectorizer.svd.components_),
        "numeric_columns": list(vectorizer.numeric_columns),
        "numeric_log1p_columns": [
            "n_events_final", "n_unique_tokens", "n_loop_removed",
            "n_dedup_removed", "span_seconds", "total_dwell_s", "median_gap_s",
        ],
        "numeric_mean": _as_list(vectorizer.scaler.mean_),
        "numeric_scale": _as_list(vectorizer.scaler.scale_),
        "numeric_block_weight": float(vectorizer.cfg.numeric_block_weight),
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "distance_p95": float(scorer.thresholds.distance_p95),
        "feature_dimension": int(next(iter(scorer.centroids.values())).shape[0]),
        "journey_contract": scorer.cfg.to_dict(),
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def export_centroid_onnx(scorer: JourneyScorer, destination: Path, opset: int) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise SystemExit("Install export dependencies: pip install onnx") from exc

    cluster_ids = np.array(sorted(scorer.centroids), dtype=np.int64)
    centroids = np.vstack([scorer.centroids[int(key)] for key in cluster_ids]).astype(np.float32)
    feature_dim = int(centroids.shape[1])
    threshold = np.array(float(scorer.thresholds.distance_p95), dtype=np.float32)
    unknown = np.array(-1, dtype=np.int64)

    initializers = [
        numpy_helper.from_array(centroids, "centroids"),
        numpy_helper.from_array(cluster_ids, "cluster_ids"),
        numpy_helper.from_array(threshold, "distance_threshold"),
        numpy_helper.from_array(unknown, "unknown_cluster"),
    ]
    nodes = [
        helper.make_node("Unsqueeze", ["features", "axes_1"], ["features_3d"]),
        helper.make_node("Sub", ["features_3d", "centroids"], ["delta"]),
        helper.make_node("Mul", ["delta", "delta"], ["squared"]),
        helper.make_node("ReduceSum", ["squared", "axes_2"], ["sum_squared"], keepdims=0),
        helper.make_node("Sqrt", ["sum_squared"], ["distances"]),
        helper.make_node("ArgMin", ["distances"], ["best_index"], axis=1, keepdims=0),
        helper.make_node("Gather", ["cluster_ids", "best_index"], ["nearest_cluster"]),
        helper.make_node("ReduceMin", ["distances"], ["distance"], axes=[1], keepdims=0),
        helper.make_node("Greater", ["distance", "distance_threshold"], ["geometric_anomaly"]),
        helper.make_node("Where", ["geometric_anomaly", "unknown_cluster", "nearest_cluster"], ["cluster"]),
    ]
    # Axes are inputs in modern opsets for Unsqueeze and ReduceSum.
    initializers.extend([
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_1"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "axes_2"),
    ])
    graph = helper.make_graph(
        nodes,
        "journey_nearest_archetype",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, feature_dim])],
        [
            helper.make_tensor_value_info("cluster", TensorProto.INT64, [None]),
            helper.make_tensor_value_info("distance", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("geometric_anomaly", TensorProto.BOOL, [None]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="unsupervised-journey-clustering",
    )
    model.ir_version = min(model.ir_version, 9)
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def export_markov(scorer: JourneyScorer, destination: Path) -> None:
    """Export sparse Markov transitions used by next-action and friction scoring."""
    bank = scorer.markov

    def bucket(label: int) -> dict[str, object]:
        transitions: dict[str, list[dict[str, object]]] = {}
        totals: dict[str, float] = {}
        for (source, target), count in bank.counts.get(label, {}).items():
            source_token = bank.vocab[source]
            target_token = bank.vocab[target]
            transitions.setdefault(source_token, []).append(
                {"token": target_token, "count": float(count)}
            )
            totals[source_token] = totals.get(source_token, 0.0) + float(count)
        for values in transitions.values():
            values.sort(key=lambda row: (-float(row["count"]), str(row["token"])))
        return {"transitions": transitions, "totals": totals}

    payload = {
        "schema_version": "1.0",
        "smoothing": float(bank.smoothing),
        "vocabulary": list(bank.vocab),
        "global": bucket(-999),
        "clusters": {
            str(label): bucket(int(label))
            for label in sorted(bank.counts)
            if int(label) != -999
        },
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def export_friction_config(scorer: JourneyScorer, destination: Path) -> None:
    """Export thresholds and journey rules needed for offline Android scoring."""
    thresholds = scorer.thresholds
    segment = scorer.cfg.segment
    payload = {
        "schema_version": "1.0",
        "distance_p95": float(thresholds.distance_p95),
        "markov_p05": float(thresholds.markov_p05),
        "markov_p01": float(thresholds.markov_p01),
        "back_rate_p90": float(thresholds.back_rate_p90),
        "loops_p90": float(thresholds.loops_p90),
        "revisit_p90": float(thresholds.revisit_p90),
        "span_p95": float(thresholds.span_p95),
        "idle_gap_seconds": float(segment.idle_gap_seconds),
        "root_return_min_events": int(segment.root_return_min_events),
        "max_journey_length": int(segment.max_journey_length),
        "min_journey_length": int(segment.min_journey_length),
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_dir = REPO_ROOT / args.model_dir
    output_dir = REPO_ROOT / (args.output_dir or f"output/mobile/{args.platform}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _install_pickle_compatibility()
    scorer = JourneyScorer.load(model_dir / f"{args.platform}_journey_scorer.pkl")
    export_centroid_onnx(scorer, output_dir / "journey_classifier.onnx", args.opset)
    export_preprocessing(scorer, output_dir / "preprocessing.json", args.platform)
    export_markov(scorer, output_dir / "markov.json")
    export_friction_config(scorer, output_dir / "friction_config.json")
    shutil.copyfile(
        REPO_ROOT / "output" / f"{args.platform}_cluster_class_mapping.json",
        output_dir / "class_mapping.json",
    )
    manifest = {
        "platform": args.platform,
        "model": "journey_classifier.onnx",
        "preprocessing": "preprocessing.json",
        "taxonomy": "class_mapping.json",
        "markov": "markov.json",
        "friction_config": "friction_config.json",
        "outputs": [
            "cluster", "distance", "geometric_anomaly", "markov_logprob",
            "friction_flags", "next_action",
        ],
        "note": "Markov transitions and friction thresholds are exported as compact JSON for Android.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"mobile bundle -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
