"""Verify Python feature/scoring parity with an exported ONNX mobile bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from Rule_based.production import canonicalize_frame  # noqa: E402
from Rule_based.score import JourneyScorer, _assign  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("android", "ios"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", default="output/clusters")
    parser.add_argument("--bundle-dir", help="default: output/mobile/<platform>")
    parser.add_argument("--rows", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    import onnxruntime as ort

    args = parse_args()
    model_dir = REPO_ROOT / args.model_dir
    bundle_dir = REPO_ROOT / (args.bundle_dir or f"output/mobile/{args.platform}")
    scorer = JourneyScorer.load(model_dir / f"{args.platform}_journey_scorer.pkl")
    source = pd.read_csv(REPO_ROOT / args.input, nrows=args.rows, low_memory=False)
    canonical, _ = canonicalize_frame(source, source_file=Path(args.input).name, cfg=scorer.cfg.canonize)
    canonical = canonical.loc[canonical.platform.eq(args.platform)]
    journeys, sequences = scorer.prepare(canonical)
    if journeys.empty:
        raise ValueError("fixture produced no scoreable journey")
    features = scorer.vectorizer.transform(journeys, sequences).astype(np.float32)
    expected, distance = _assign(features, scorer.centroids)
    expected = np.where(distance > scorer.thresholds.distance_p95, -1, expected)
    session = ort.InferenceSession(str(bundle_dir / "journey_classifier.onnx"), providers=["CPUExecutionProvider"])
    actual_cluster, actual_distance, actual_anomaly = session.run(None, {"features": features})
    np.testing.assert_array_equal(actual_cluster, expected)
    np.testing.assert_allclose(actual_distance, distance, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual_anomaly, distance > scorer.thresholds.distance_p95)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "2.0":
        raise AssertionError("expected mobile schema_version 2.0")
    print(f"parity OK: {len(journeys)} journeys, feature_dim={features.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
