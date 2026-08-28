"""Export fitted JourneyScorer models (Android & iOS) to ultra-lightweight mobile bundles (<5MB).

This script packages the complete real-time scoring engine for iOS and Android:
1. Multi-channel Sequence Encoders (TF-IDF vocabularies + SVD components + Folded SVD*IDF)
2. Global PCA Projection matrix and mean vector
3. StandardScaler parameters and numeric column definitions
4. Centroids dictionary and binary distance matrix
5. Markov Bank (vocabulary, transition counts, row totals, smoothing)
6. Calibrated ScoreThresholds & Friction rules
7. Pipeline Configuration & Segmentation rules
8. Cluster-to-Business Naming & Archetype mappings from cluster_mapping.csv

Outputs both clean JSON descriptors and ultra-compact FP16/FP32 binary blobs suitable
for zero-copy mmap execution on iOS (Swift) and Android (Kotlin / C++).
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .score import JourneyScorer


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_cluster_mapping(
    mapping_path: str | Path | None,
    platform: str,
) -> dict[str, dict[str, Any]]:
    """Parse cluster_mapping.csv for the given platform into a clean lookup dictionary."""
    if mapping_path is None or not Path(mapping_path).exists():
        LOGGER.warning("No cluster mapping CSV provided or found at: %s", mapping_path)
        return {}

    path = Path(mapping_path)
    LOGGER.info("Loading cluster mappings from %s for platform=%s...", path, platform)
    df = pd.read_csv(path)
    if "platform" in df.columns:
        df = df[df["platform"].astype(str).str.lower() == platform.lower()]

    mapping_dict: dict[str, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        cid = str(int(getattr(row, "cluster_id")))
        mapping_dict[cid] = {
            "cluster_id": int(getattr(row, "cluster_id")),
            "cluster_name": str(getattr(row, "cluster_name", f"Archetype {cid}")),
            "business_family": str(getattr(row, "business_family", "Chưa phân loại")),
            "business_submodule": str(getattr(row, "business_submodule", "") if pd.notna(getattr(row, "business_submodule", None)) else ""),
            "business_detail": str(getattr(row, "business_detail", "") if pd.notna(getattr(row, "business_detail", None)) else ""),
            "naming_confidence": str(getattr(row, "naming_confidence", "medium")),
            "naming_source": str(getattr(row, "naming_source", "")),
            "needs_review": bool(getattr(row, "needs_review", False)),
            "journey_count": int(getattr(row, "journey_count", 0)) if pd.notna(getattr(row, "journey_count", None)) else 0,
            "journey_share": round(float(getattr(row, "journey_share", 0.0)), 4) if pd.notna(getattr(row, "journey_share", None)) else 0.0,
            "medoid_path": str(getattr(row, "medoid_path", "")) if pd.notna(getattr(row, "medoid_path", None)) else "",
        }

    # Default fallback for unknown / unmapped noise
    if "-1" not in mapping_dict:
        mapping_dict["-1"] = {
            "cluster_id": -1,
            "cluster_name": "Chưa phân loại | Journey hỗn hợp/nhiễu",
            "business_family": "Chưa phân loại",
            "business_submodule": "Journey hỗn hợp/nhiễu",
            "business_detail": "",
            "naming_confidence": "low",
            "naming_source": "noise_cluster",
            "needs_review": True,
            "journey_count": 0,
            "journey_share": 0.0,
            "medoid_path": "",
        }

    LOGGER.info("Loaded %d cluster name mappings for %s.", len(mapping_dict), platform)
    return mapping_dict


def export_single_platform(
    scorer: JourneyScorer,
    platform: str,
    output_dir: str | Path,
    *,
    mapping_path: str | Path | None = None,
    precision: str = "float16",
) -> dict[str, Any]:
    """Export a single platform's JourneyScorer into an optimized mobile bundle."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if precision == "float16" else np.float32

    LOGGER.info("Exporting [%s] JourneyScorer to %s (precision: %s)...", platform.upper(), out_dir, precision)

    # -------------------------------------------------------------------------
    # 1. Config & Rules
    # -------------------------------------------------------------------------
    config_dict = {
        "platform": platform.lower(),
        "segment": {
            "idle_gap_seconds": float(scorer.cfg.segment.idle_gap_seconds),
            "root_return_min_events": int(scorer.cfg.segment.root_return_min_events),
            "max_journey_length": int(scorer.cfg.segment.max_journey_length),
            "min_journey_length": int(scorer.cfg.segment.min_journey_length),
            "cut_on_root_return": bool(scorer.cfg.segment.cut_on_root_return),
            "cut_on_auth_change": bool(scorer.cfg.segment.cut_on_auth_change),
        },
        "postprocess": {
            "collapse_consecutive": bool(scorer.cfg.post.collapse_consecutive),
            "collapse_cycles": bool(scorer.cfg.post.collapse_cycles),
            "max_cycle_period": int(scorer.cfg.post.max_cycle_period),
            "min_cycle_repeats": int(scorer.cfg.post.min_cycle_repeats),
            "drop_chrome": bool(scorer.cfg.post.drop_chrome),
            "drop_boot": bool(scorer.cfg.post.drop_boot),
        },
        "tokens": {
            "level": str(scorer.cfg.tokens.level),
            "backoff_level": str(scorer.cfg.tokens.backoff_level),
            "min_journey_df": int(scorer.cfg.tokens.min_journey_df),
        },
        "canonize": {
            "namespace_by_os": bool(scorer.cfg.canonize.namespace_by_os),
            "canonize_urls": bool(scorer.cfg.canonize.canonize_urls),
            "mask_id_segments": bool(scorer.cfg.canonize.mask_id_segments),
        }
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # -------------------------------------------------------------------------
    # 2. Calibrated Thresholds
    # -------------------------------------------------------------------------
    t = scorer.thresholds
    thresholds_dict = {
        "distance_p95": float(t.distance_p95),
        "markov_p05": float(t.markov_p05),
        "markov_p01": float(t.markov_p01),
        "back_rate_p90": float(t.back_rate_p90),
        "loops_p90": float(t.loops_p90),
        "revisit_p90": float(t.revisit_p90),
        "span_p95": float(t.span_p95),
    }
    with (out_dir / "thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)

    # -------------------------------------------------------------------------
    # 3. Centroids
    # -------------------------------------------------------------------------
    sorted_cluster_ids = sorted(int(k) for k in scorer.centroids.keys())
    centroids_dict = {
        str(cid): [round(float(x), 6) for x in scorer.centroids[cid]]
        for cid in sorted_cluster_ids
    }
    with (out_dir / "centroids.json").open("w", encoding="utf-8") as f:
        json.dump(centroids_dict, f)

    centroids_mat = np.vstack([scorer.centroids[cid] for cid in sorted_cluster_ids]).astype(dtype)
    with (out_dir / "centroids_matrix.bin").open("wb") as f:
        f.write(centroids_mat.tobytes())

    # -------------------------------------------------------------------------
    # 4. JourneyVectorizer (TF-IDF + SVD for each channel + PCA + Scaler)
    # -------------------------------------------------------------------------
    vec = scorer.vectorizer
    active_channel_names = sorted(getattr(vec, "channels", {}).keys())
    channel_weights = {
        "primary": 1.0,
        **{name: float(getattr(vec, "channel_weights", {}).get(name, 1.0)) for name in active_channel_names}
    }

    # Vocabularies
    vocabularies = {
        "primary": {str(k): int(v) for k, v in vec.primary.tfidf.vocabulary_.items()},
        **{name: {str(k): int(v) for k, v in vec.channels[name].tfidf.vocabulary_.items()} for name in active_channel_names}
    }
    with (out_dir / "vocabularies.json").open("w", encoding="utf-8") as f:
        json.dump(vocabularies, f)

    # Numeric Scaler
    log_skewed_cols = [
        c for c in (
            "n_events_final", "n_unique_tokens", "n_loop_removed", "n_dedup_removed",
            "span_seconds", "median_gap_s", "p90_gap_s", "max_gap_s"
        ) if c in vec.numeric_columns
    ]
    vectorizer_metadata = {
        "channel_weights": channel_weights,
        "numeric_block_weight": float(vec.cfg.numeric_block_weight),
        "numeric_columns": vec.numeric_columns,
        "numeric_log1p_columns": log_skewed_cols,
        "numeric_mean": [float(x) for x in vec.scaler.mean_],
        "numeric_scale": [float(x) for x in vec.scaler.scale_],
        "svd_components_count": int(vec.primary.svd.n_components) if vec.primary.svd is not None else 48,
        "has_global_pca": vec.global_pca is not None,
        "global_pca_components_count": int(vec.global_pca.n_components_) if vec.global_pca is not None else None,
        "final_embedding_dim": int(centroids_mat.shape[1]),
    }
    with (out_dir / "vectorizer_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(vectorizer_metadata, f, indent=2)

    # Export binary projection weights
    weights_binary_path = out_dir / f"projection_weights.{precision}.bin"
    with weights_binary_path.open("wb") as f:
        # Primary folded matrix (48, vocab_len)
        W_primary = (vec.primary.svd.components_ * vec.primary.tfidf.idf_).astype(dtype)
        f.write(W_primary.tobytes())

        # Channels folded matrices
        for name in active_channel_names:
            ch_enc = vec.channels[name]
            W_ch = (ch_enc.svd.components_ * ch_enc.tfidf.idf_).astype(dtype)
            f.write(W_ch.tobytes())

        # Global PCA
        if vec.global_pca is not None:
            f.write(vec.global_pca.components_.astype(dtype).tobytes())
            f.write(vec.global_pca.mean_.astype(np.float32).tobytes())

        # Scaler
        f.write(vec.scaler.mean_.astype(np.float32).tobytes())
        f.write(vec.scaler.scale_.astype(np.float32).tobytes())

    # -------------------------------------------------------------------------
    # 5. Markov Bank
    # -------------------------------------------------------------------------
    markov = scorer.markov
    cluster_keys = sorted(int(k) for k in markov.counts.keys())
    cluster_idx_map = {cid: idx for idx, cid in enumerate(cluster_keys)}

    # JSON export of Markov transitions
    markov_json_dict = {
        "smoothing": float(markov.smoothing),
        "vocab_size": len(markov.vocab),
        "vocabulary": markov.vocab,
        "clusters": {}
    }
    for cid in cluster_keys:
        trans = markov.counts[cid]
        totals = markov.row_totals[cid]
        transitions_by_src: dict[str, dict[str, float]] = {}
        totals_by_src: dict[str, float] = {}
        default_total = markov.smoothing * len(markov.vocab)

        for (src_idx, dst_idx), cnt in trans.items():
            src_tok = markov.vocab[src_idx]
            dst_tok = markov.vocab[dst_idx]
            if src_tok not in transitions_by_src:
                transitions_by_src[src_tok] = {}
            transitions_by_src[src_tok][dst_tok] = float(cnt)

        for src_idx, tot in enumerate(totals):
            if abs(tot - default_total) > 1e-4:
                totals_by_src[markov.vocab[src_idx]] = float(tot)

        markov_json_dict["clusters"][str(cid)] = {
            "transitions": transitions_by_src,
            "totals": totals_by_src,
        }

    with (out_dir / "markov.json").open("w", encoding="utf-8") as f:
        json.dump(markov_json_dict, f)

    # Compact Binary Markov Graph for high-performance zero-copy mmap
    all_transitions = []
    all_row_totals = []
    default_total = markov.smoothing * len(markov.vocab)

    for cid in cluster_keys:
        c_idx = cluster_idx_map[cid]
        for (src_idx, dst_idx), cnt in markov.counts[cid].items():
            all_transitions.append((c_idx, src_idx, dst_idx, float(cnt)))
        for src_idx, tot in enumerate(markov.row_totals[cid]):
            if abs(tot - default_total) > 1e-4:
                all_row_totals.append((c_idx, src_idx, float(tot)))

    with (out_dir / "markov_graph.bin").open("wb") as f:
        f.write(struct.pack("<IIff", len(all_transitions), len(all_row_totals), float(markov.smoothing), float(len(markov.vocab))))
        for c_idx, src, dst, cnt in all_transitions:
            f.write(struct.pack("<HHHf", c_idx, src, dst, cnt))
        for c_idx, src, tot in all_row_totals:
            f.write(struct.pack("<HHf", c_idx, src, tot))

    # -------------------------------------------------------------------------
    # 6. Cluster Names / Business Mapping
    # -------------------------------------------------------------------------
    cluster_mapping = parse_cluster_mapping(mapping_path, platform)
    if cluster_mapping:
        with (out_dir / "cluster_mapping.json").open("w", encoding="utf-8") as f:
            json.dump(cluster_mapping, f, indent=2)

    # -------------------------------------------------------------------------
    # 7. Master Manifest
    # -------------------------------------------------------------------------
    manifest = {
        "bundle_version": "2.1.0",
        "platform": platform.lower(),
        "precision": precision,
        "n_clusters": len(sorted_cluster_ids),
        "cluster_ids": sorted_cluster_ids,
        "embedding_dim": int(centroids_mat.shape[1]),
        "channel_weights": channel_weights,
        "has_cluster_mapping": bool(cluster_mapping),
        "files": {
            "manifest": "manifest.json",
            "config": "config.json",
            "thresholds": "thresholds.json",
            "centroids_json": "centroids.json",
            "centroids_binary": "centroids_matrix.bin",
            "vocabularies": "vocabularies.json",
            "vectorizer_metadata": "vectorizer_metadata.json",
            "weights_binary": f"projection_weights.{precision}.bin",
            "markov_json": "markov.json",
            "markov_binary": "markov_graph.bin",
            "cluster_mapping": "cluster_mapping.json" if cluster_mapping else None,
        }
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_bytes = sum(p.stat().st_size for p in out_dir.glob("*") if p.is_file())
    LOGGER.info(
        "✅ [%s] Export complete! Bundle size: %.2f MB (%d files in %s)",
        platform.upper(), total_bytes / (1024 * 1024), len(list(out_dir.glob("*"))), out_dir
    )
    return manifest


def verify_exported_bundle(scorer: JourneyScorer, export_dir: str | Path, tolerance: float | None = None) -> bool:
    """Verify that exported binary weights faithfully reproduce original Scorer transform."""
    exp = Path(export_dir)
    manifest = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
    precision = manifest["precision"]
    if tolerance is None:
        tolerance = 2e-4 if precision == "float16" else 1e-5
    dtype = np.float16 if precision == "float16" else np.float32

    vocabs = json.loads((exp / "vocabularies.json").read_text(encoding="utf-8"))
    vec_meta = json.loads((exp / "vectorizer_metadata.json").read_text(encoding="utf-8"))

    weights_path = exp / f"projection_weights.{precision}.bin"
    with weights_path.open("rb") as f:
        raw_bytes = f.read()

    offset = 0
    itemsize = np.dtype(dtype).itemsize

    # Primary W
    p_len = 48 * len(vocabs["primary"])
    W_p = np.frombuffer(raw_bytes, dtype=dtype, count=p_len, offset=offset).reshape(48, len(vocabs["primary"]))
    offset += p_len * itemsize

    # Channels W
    W_channels = {}
    for ch_name in sorted(k for k in vocabs if k != "primary"):
        ch_len = 48 * len(vocabs[ch_name])
        W_channels[ch_name] = np.frombuffer(raw_bytes, dtype=dtype, count=ch_len, offset=offset).reshape(48, len(vocabs[ch_name]))
        offset += ch_len * itemsize

    # Global PCA
    if vec_meta["has_global_pca"]:
        pca_dim = vec_meta["global_pca_components_count"]
        pca_in_dim = 48 * (1 + len(W_channels)) + len(vec_meta["numeric_columns"])
        pca_comp_len = pca_dim * pca_in_dim
        pca_comp = np.frombuffer(raw_bytes, dtype=dtype, count=pca_comp_len, offset=offset).reshape(pca_dim, pca_in_dim)
        offset += pca_comp_len * itemsize
        pca_mean = np.frombuffer(raw_bytes, dtype=np.float32, count=pca_in_dim, offset=offset)
        offset += pca_in_dim * 4
    else:
        pca_comp, pca_mean = None, None

    # Scaler
    num_dim = len(vec_meta["numeric_columns"])
    scaler_mean = np.frombuffer(raw_bytes, dtype=np.float32, count=num_dim, offset=offset)
    offset += num_dim * 4
    scaler_scale = np.frombuffer(raw_bytes, dtype=np.float32, count=num_dim, offset=offset)
    offset += num_dim * 4

    test_seq = ["view@HomeVC", "action@HomeVC#btn_pay", "view@PaymentVC", "action@PaymentVC#confirm"]
    test_channels = {
        "coarse": ["internet/general", "internet/payment", "internet/payment", "internet/payment"],
        "intent": ["internet/general/view", "internet/payment/bill/pay", "internet/payment/view", "internet/payment/bill/confirm"],
        "operation": ["entry:view", "step:pay", "entry:view", "commit:confirm"]
    }
    dummy_journey = pd.DataFrame([{
        "n_events_final": 4,
        "n_unique_tokens": 4,
        "action_ratio": 0.5,
        "back_rate": 0.0,
        "revisit_ratio": 0.0,
        "n_loop_removed": 0,
        "n_dedup_removed": 0,
        "span_seconds": 25.0,
        "median_gap_s": 5.0,
        "p90_gap_s": 8.0,
        "max_gap_s": 10.0,
    }])

    expected_matrix = scorer.vectorizer.transform(dummy_journey, [test_seq], {k: [v] for k, v in test_channels.items()})

    def _encode_channel(seq: list[str], vocab: dict[str, int], W: np.ndarray) -> np.ndarray:
        bounded = ["<bos>"] + seq + ["<eos>"]
        ngrams = []
        for n in (1, 2):
            for i in range(len(bounded) - n + 1):
                ngrams.append(" ".join(bounded[i:i + n]))
        from collections import Counter
        counts = Counter(ngrams)
        z = np.zeros(48, dtype=np.float32)
        for ng, cnt in counts.items():
            if ng in vocab:
                idx = vocab[ng]
                tf = 1.0 + np.log(cnt)
                z += tf * W[:, idx].astype(np.float32)
        norm = np.linalg.norm(z)
        return (z / norm) if norm > 0 else z

    blocks = [_encode_channel(test_seq, vocabs["primary"], W_p) * vec_meta["channel_weights"]["primary"]]
    for ch_name in sorted(W_channels):
        w_ch = vec_meta["channel_weights"].get(ch_name, 1.0)
        blocks.append(_encode_channel(test_channels[ch_name], vocabs[ch_name], W_channels[ch_name]) * w_ch)

    raw_num = dummy_journey[vec_meta["numeric_columns"]].to_numpy(dtype=float)[0]
    for i, col in enumerate(vec_meta["numeric_columns"]):
        if col in vec_meta["numeric_log1p_columns"]:
            raw_num[i] = np.log1p(max(0.0, raw_num[i]))
    scaled_num = (raw_num - scaler_mean) / scaler_scale
    norm_num = np.linalg.norm(scaled_num)
    if norm_num > 0:
        scaled_num /= norm_num
    blocks.append(scaled_num * vec_meta["numeric_block_weight"])

    concatenated = np.concatenate(blocks)
    if pca_comp is not None:
        projected = (concatenated - pca_mean) @ pca_comp.T.astype(np.float32)
    else:
        projected = concatenated
    norm_final = np.linalg.norm(projected)
    if norm_final > 0:
        projected /= norm_final

    diff = float(np.max(np.abs(expected_matrix[0] - projected)))
    LOGGER.info("Max numerical difference on [%s]: %.6e (Tolerance: %.6e)", manifest["platform"], diff, tolerance)
    assert diff <= tolerance, f"Verification failed: max difference {diff} exceeds tolerance {tolerance}"
    LOGGER.info("🎉 Verification PASSED for [%s]!", manifest["platform"].upper())
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export JourneyScorer for Android and iOS into mobile bundles.")
    parser.add_argument(
        "--platform",
        choices=["android", "ios", "both"],
        default="both",
        help="Target platform to export (android, ios, or both). Default: both",
    )
    parser.add_argument(
        "--scorer-dir",
        type=Path,
        default=Path("output/partitioned_runs/pca48_svd48_ngrams12_500k/latest"),
        help="Base run directory containing {platform}/{platform}_journey_scorer.pkl",
    )
    parser.add_argument(
        "--scorer",
        type=Path,
        default=None,
        help="Explicit path to a single scorer .pkl file (used when --platform is android or ios)",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("output/scores/pca48_ngrams12_500/cluster_mapping.csv"),
        help="Path to cluster_mapping.csv for mapping cluster IDs to business names",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/mobile"),
        help="Root output directory (will create {output-dir}/{platform}/)",
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Precision for projection weights and centroids (default: float16)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify exported weights against original Python Scorer transform",
    )
    args = parser.parse_args()

    platforms = ["android", "ios"] if args.platform == "both" else [args.platform]

    for plat in platforms:
        if args.scorer is not None and args.platform != "both":
            scorer_path = args.scorer
        else:
            scorer_path = args.scorer_dir / plat / f"{plat}_journey_scorer.pkl"

        if not scorer_path.exists():
            LOGGER.error("Scorer file not found: %s", scorer_path)
            continue

        plat_out = args.output_dir / plat
        scorer = JourneyScorer.load(scorer_path)
        export_single_platform(scorer, plat, plat_out, mapping_path=args.mapping, precision=args.precision)

        if args.verify:
            verify_exported_bundle(scorer, plat_out)


if __name__ == "__main__":
    main()
