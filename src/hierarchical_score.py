"""C-first, B-fallback inference for fitted journey clustering runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cluster_postprocess import nearest_two_centroids
from .score import JourneyScorer


class HierarchicalJourneyScorer:
    """Run strict C inference first and consult B only for C-noise journeys."""

    def __init__(
        self,
        c_scorer: JourneyScorer,
        b_scorer: JourneyScorer,
        c_distance_limits: dict[int, float],
        b_distance_limits: dict[int, float],
        b_markov_limits: dict[int, float],
        *,
        min_distance_margin: float = 0.10,
        recurring_support: int = 10,
        name_mapping: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.c_scorer = c_scorer
        self.b_scorer = b_scorer
        self.c_distance_limits = {
            int(key): float(value) for key, value in c_distance_limits.items()
        }
        self.b_distance_limits = {
            int(key): float(value) for key, value in b_distance_limits.items()
        }
        self.b_markov_limits = {
            int(key): float(value) for key, value in b_markov_limits.items()
        }
        self.min_distance_margin = float(min_distance_margin)
        self.recurring_support = int(recurring_support)
        self.name_mapping = name_mapping or {}
        self._validate()

    def _validate(self) -> None:
        c_cfg = self.c_scorer.cfg.to_dict()
        b_cfg = self.b_scorer.cfg.to_dict()
        for section in ("canonize", "tokens", "post", "segment"):
            if c_cfg.get(section) != b_cfg.get(section):
                raise ValueError(f"B/C preprocessing mismatch in config section {section!r}")
        b_clusters = set(int(label) for label in self.b_scorer.centroids)
        c_clusters = set(int(label) for label in self.c_scorer.centroids)
        if not c_clusters <= set(self.c_distance_limits):
            missing = sorted(c_clusters - set(self.c_distance_limits))[:10]
            raise ValueError(f"missing C distance limits for clusters: {missing}")
        if not b_clusters <= set(self.b_distance_limits):
            missing = sorted(b_clusters - set(self.b_distance_limits))[:10]
            raise ValueError(f"missing B distance limits for clusters: {missing}")
        if not b_clusters <= set(self.b_markov_limits):
            missing = sorted(b_clusters - set(self.b_markov_limits))[:10]
            raise ValueError(f"missing B Markov limits for clusters: {missing}")

    @classmethod
    def load(cls, postprocess_dir: str | Path, platform: str) -> "HierarchicalJourneyScorer":
        root = Path(postprocess_dir)
        config = json.loads((root / "postprocess_config.json").read_text(encoding="utf-8"))
        b_threshold_payload = json.loads(
            (root / f"{platform}_B_cluster_thresholds.json").read_text(encoding="utf-8")
        )
        c_threshold_payload = json.loads(
            (root / f"{platform}_C_cluster_thresholds.json").read_text(encoding="utf-8")
        )
        name_path = root / f"{platform}_hierarchical_cluster_names.json"
        names: dict[str, dict[str, Any]] = {}
        if name_path.exists():
            payload = json.loads(name_path.read_text(encoding="utf-8"))
            names = {str(item["effective_cluster_key"]): item for item in payload["clusters"]}
        c_run = Path(config["C_run"])
        b_run = Path(config["B_run"])
        return cls(
            JourneyScorer.load(c_run / f"{platform}_journey_scorer.pkl"),
            JourneyScorer.load(b_run / f"{platform}_journey_scorer.pkl"),
            {int(key): value for key, value in c_threshold_payload["distance_limits"].items()},
            {int(key): value for key, value in b_threshold_payload["distance_limits"].items()},
            {int(key): value for key, value in b_threshold_payload["markov_limits"].items()},
            min_distance_margin=float(config["thresholds"]["min_distance_margin"]),
            recurring_support=int(config["thresholds"]["recurring_support"]),
            name_mapping=names,
        )

    def score(self, raw_events: pd.DataFrame) -> pd.DataFrame:
        journeys, sequences, channels = self.c_scorer.prepare(raw_events)
        return self.score_prepared(journeys, sequences, channels=channels)

    def score_prepared(
        self,
        journeys: pd.DataFrame,
        sequences: list[list[str]],
        *,
        channels: dict[str, list[list[str]]] | None = None,
    ) -> pd.DataFrame:
        c_scored = self.c_scorer.score_prepared(
            journeys, sequences, channels=channels, distance_limits=self.c_distance_limits
        )
        if c_scored.empty:
            return c_scored

        out = c_scored.rename(
            columns={
                "cluster": "c_cluster",
                "nearest_cluster": "c_nearest_cluster",
                "distance_to_centroid": "c_distance_to_centroid",
                "distance_limit": "c_distance_limit",
                "markov_logprob": "c_markov_logprob",
                "geometric_anomaly": "c_geometric_anomaly",
                "generative_anomaly": "c_generative_anomaly",
                "severe_anomaly": "c_severe_anomaly",
                "next_action": "c_next_action",
                "next_action_share": "c_next_action_share",
            }
        )
        n = len(out)
        c_noise = out["c_cluster"].to_numpy(dtype=int) == -1
        secondary = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["nearest_b_cluster"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["second_b_cluster"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        for column in (
            "nearest_b_distance", "second_b_distance", "nearest_b_distance_limit",
            "nearest_b_distance_ratio", "nearest_b_distance_margin",
            "nearest_b_markov_logprob", "nearest_b_markov_limit",
        ):
            out[column] = np.nan
        b_pass = np.zeros(n, dtype=bool)

        if c_noise.any():
            idx = np.flatnonzero(c_noise)
            noise_journeys = journeys.iloc[idx].reset_index(drop=True)
            noise_sequences = [sequences[position] for position in idx]
            # B and C share preprocessing (checked in `_validate`), so the same
            # channel sequences describe both embeddings.
            noise_channels = {
                name: [rows[position] for position in idx] for name, rows in (channels or {}).items()
            }
            matrix = self.b_scorer.vectorizer.transform(
                noise_journeys, noise_sequences, noise_channels
            )
            first, first_distance, second, second_distance = nearest_two_centroids(
                matrix, self.b_scorer.centroids
            )
            markov = self.b_scorer.markov.score_all(noise_sequences, first)
            distance_limit = np.array([self.b_distance_limits[int(label)] for label in first])
            markov_limit = np.array([self.b_markov_limits[int(label)] for label in first])
            margin = (second_distance - first_distance) / np.maximum(first_distance, 1e-12)
            passed = (
                (first_distance <= distance_limit)
                & (margin >= self.min_distance_margin)
                & np.isfinite(markov)
                & (markov >= markov_limit)
            )
            b_pass[idx] = passed
            secondary.iloc[idx[passed]] = first[passed]
            out.loc[c_noise, "nearest_b_cluster"] = first
            out.loc[c_noise, "second_b_cluster"] = second
            out.loc[c_noise, "nearest_b_distance"] = first_distance
            out.loc[c_noise, "second_b_distance"] = second_distance
            out.loc[c_noise, "nearest_b_distance_limit"] = distance_limit
            out.loc[c_noise, "nearest_b_distance_ratio"] = first_distance / distance_limit
            out.loc[c_noise, "nearest_b_distance_margin"] = margin
            out.loc[c_noise, "nearest_b_markov_logprob"] = markov
            out.loc[c_noise, "nearest_b_markov_limit"] = markov_limit

        out["primary_cluster"] = out["c_cluster"].astype("Int64")
        out["secondary_cluster"] = secondary
        assignment = np.full(n, "unassigned_novel", dtype=object)
        assignment[~c_noise] = "C_primary"
        assignment[b_pass] = "B_secondary_inference"

        behavioural_flags = out["friction_flags"].fillna("").map(
            lambda value: "|".join(
                flag
                for flag in str(value).split("|")
                if flag and flag not in {"unknown_archetype", "improbable_transitions"}
            )
        )
        out["behavioral_friction_flags"] = behavioural_flags
        noise_support = out.loc[c_noise, "sequence"].value_counts(dropna=False).to_dict()
        out["exact_sequence_batch_noise_support"] = np.where(
            c_noise, out["sequence"].map(noise_support).fillna(0), 0
        ).astype(int)
        unresolved = c_noise & (~b_pass)
        recurring = unresolved & (
            out["exact_sequence_batch_noise_support"].to_numpy(dtype=int)
            >= self.recurring_support
        )
        friction = unresolved & (~recurring) & behavioural_flags.ne("").to_numpy()
        assignment[recurring] = "unassigned_recurring"
        assignment[friction] = "unassigned_friction"
        out["assignment_type"] = assignment

        keys = np.full(n, "UNKNOWN", dtype=object)
        keys[~c_noise] = [f"C:{int(value)}" for value in out.loc[~c_noise, "c_cluster"]]
        keys[b_pass] = [f"B:{int(value)}" for value in secondary.loc[b_pass]]
        out["effective_cluster_key"] = keys
        out["effective_markov_logprob"] = out["c_markov_logprob"]
        out.loc[b_pass, "effective_markov_logprob"] = out.loc[
            b_pass, "nearest_b_markov_logprob"
        ]
        out["effective_next_action"] = out["c_next_action"]
        out["effective_next_action_share"] = out["c_next_action_share"]
        if b_pass.any():
            b_idx = np.flatnonzero(b_pass)
            predictions = [
                self.b_scorer.markov.predict_next(sequences[position], int(secondary.iloc[position]), 1)
                for position in b_idx
            ]
            out.loc[b_pass, "effective_next_action"] = [
                prediction[0][0] if prediction else None for prediction in predictions
            ]
            out.loc[b_pass, "effective_next_action_share"] = [
                prediction[0][2] if prediction else np.nan for prediction in predictions
            ]
        out.loc[unresolved, ["effective_next_action", "effective_next_action_share"]] = [None, np.nan]
        return self._attach_names(out)

    def _attach_names(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        unknown = {
            "cluster_name": "Hành trình chưa phân loại / hỗn hợp",
            "cluster_name_en": "Unclassified / mixed journeys",
            "business_family": "chưa phân loại",
            "business_family_code": "unknown",
            "naming_confidence": "not_applicable",
        }
        rows = [self.name_mapping.get(str(key), unknown) for key in out["effective_cluster_key"]]
        out["cluster_name"] = [row.get("cluster_name", unknown["cluster_name"]) for row in rows]
        out["cluster_name_en"] = [
            row.get("cluster_name_en", unknown["cluster_name_en"]) for row in rows
        ]
        out["business_family"] = [
            row.get("business_family", unknown["business_family"]) for row in rows
        ]
        out["business_family_code"] = [
            row.get("business_family_code", unknown["business_family_code"]) for row in rows
        ]
        out["naming_confidence"] = [
            row.get("naming_confidence", unknown["naming_confidence"]) for row in rows
        ]
        return out

    def predict_next(
        self,
        sequence: list[str],
        effective_cluster_key: str,
        *,
        top_k: int = 5,
    ) -> pd.DataFrame:
        if effective_cluster_key == "UNKNOWN":
            return pd.DataFrame(columns=["next_token", "smoothed_probability", "observed_share"])
        namespace, raw_cluster = effective_cluster_key.split(":", 1)
        scorer = self.c_scorer if namespace == "C" else self.b_scorer
        return scorer.predict_next(sequence, int(raw_cluster), top_k=top_k)
