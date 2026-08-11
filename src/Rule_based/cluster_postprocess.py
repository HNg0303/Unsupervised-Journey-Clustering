"""Hierarchical post-processing for a strict C run and a lenient B run.

The strict run remains authoritative.  Its noise is enriched from the lenient
run without overwriting either fitted HDBSCAN label:

* a journey already clustered by B receives that B cluster as a secondary
  archetype;
* a journey that is noise in both runs may receive a soft B assignment only
  when centroid distance, distance margin, and Markov likelihood all pass
  thresholds calibrated on B's fitted members;
* unresolved journeys are separated into recurring, friction-like, and novel
  groups so that ``-1`` is no longer treated as one homogeneous bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PostprocessThresholds:
    """Thresholds used to turn B into a secondary assignment layer."""

    distance_quantile: float = 0.95
    markov_quantile: float = 0.05
    min_distance_margin: float = 0.10
    recurring_support: int = 10

    def validate(self) -> None:
        if not 0.0 < self.distance_quantile < 1.0:
            raise ValueError("distance_quantile must be strictly between 0 and 1")
        if not 0.0 < self.markov_quantile < 1.0:
            raise ValueError("markov_quantile must be strictly between 0 and 1")
        if self.min_distance_margin < 0.0:
            raise ValueError("min_distance_margin cannot be negative")
        if self.recurring_support < 2:
            raise ValueError("recurring_support must be at least 2")


def align_journeys(c_journeys: pd.DataFrame, b_journeys: pd.DataFrame) -> pd.DataFrame:
    """Return C rows enriched with B labels, validating one-to-one identity."""

    required = {"journey_id", "cluster", "sequence"}
    for name, frame in (("C", c_journeys), ("B", b_journeys)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} journeys missing required columns: {sorted(missing)}")
        duplicates = frame["journey_id"].duplicated(keep=False)
        if duplicates.any():
            examples = frame.loc[duplicates, "journey_id"].astype(str).head(5).tolist()
            raise ValueError(f"{name} contains duplicate journey_id values: {examples}")

    c_ids = set(c_journeys["journey_id"])
    b_ids = set(b_journeys["journey_id"])
    if c_ids != b_ids:
        raise ValueError(
            "B and C must contain the same journey_id set; "
            f"only_in_C={len(c_ids - b_ids)}, only_in_B={len(b_ids - c_ids)}"
        )

    b_lookup = b_journeys[["journey_id", "cluster", "sequence"]].rename(
        columns={"cluster": "b_cluster", "sequence": "b_sequence"}
    )
    aligned = c_journeys.rename(columns={"cluster": "c_cluster"}).merge(
        b_lookup, on="journey_id", how="left", validate="one_to_one"
    )
    mismatched = aligned["sequence"].fillna("") != aligned["b_sequence"].fillna("")
    if mismatched.any():
        examples = aligned.loc[mismatched, "journey_id"].astype(str).head(5).tolist()
        raise ValueError(f"B and C sequence mismatch for journey_id values: {examples}")
    return aligned.drop(columns=["b_sequence"])


def nearest_two_centroids(
    matrix: np.ndarray,
    centroids: Mapping[int, np.ndarray],
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest/second-nearest labels and distances with bounded memory."""

    if not centroids:
        n = matrix.shape[0]
        return (
            np.full(n, -1, dtype=int),
            np.full(n, np.nan),
            np.full(n, -1, dtype=int),
            np.full(n, np.nan),
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    labels = np.array(sorted(int(label) for label in centroids), dtype=int)
    centers = np.vstack([centroids[int(label)] for label in labels])
    if matrix.shape[1] != centers.shape[1]:
        raise ValueError(
            f"matrix dimension {matrix.shape[1]} does not match centroids {centers.shape[1]}"
        )

    n = matrix.shape[0]
    first_label = np.empty(n, dtype=int)
    first_distance = np.empty(n, dtype=float)
    second_label = np.full(n, -1, dtype=int)
    second_distance = np.full(n, np.nan, dtype=float)
    center_norm = np.sum(centers * centers, axis=1)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = matrix[start:stop]
        squared = (
            np.sum(block * block, axis=1)[:, None]
            + center_norm[None, :]
            - 2.0 * block @ centers.T
        )
        np.maximum(squared, 0.0, out=squared)
        if len(labels) == 1:
            best = np.zeros(stop - start, dtype=int)
            first_label[start:stop] = labels[best]
            first_distance[start:stop] = np.sqrt(squared[:, 0])
            continue

        pair = np.argpartition(squared, kth=1, axis=1)[:, :2]
        pair_dist = np.take_along_axis(squared, pair, axis=1)
        order = np.argsort(pair_dist, axis=1)
        pair = np.take_along_axis(pair, order, axis=1)
        pair_dist = np.take_along_axis(pair_dist, order, axis=1)
        first_label[start:stop] = labels[pair[:, 0]]
        second_label[start:stop] = labels[pair[:, 1]]
        first_distance[start:stop] = np.sqrt(pair_dist[:, 0])
        second_distance[start:stop] = np.sqrt(pair_dist[:, 1])

    return first_label, first_distance, second_label, second_distance


def cluster_quantiles(
    labels: np.ndarray,
    values: np.ndarray,
    quantile: float,
) -> dict[int, float]:
    """Calculate one finite-value quantile per non-noise cluster."""

    frame = pd.DataFrame({"cluster": labels, "value": values})
    frame = frame[(frame["cluster"] != -1) & np.isfinite(frame["value"])]
    return {
        int(cluster): float(group["value"].quantile(quantile))
        for cluster, group in frame.groupby("cluster", sort=True)
    }


def friction_flags(frame: pd.DataFrame, thresholds: object) -> pd.Series:
    """Explain behavioural friction using thresholds stored in JourneyScorer."""

    flags: list[str] = []
    for row in frame.itertuples(index=False):
        marks: list[str] = []
        if getattr(row, "back_rate", 0.0) > thresholds.back_rate_p90:
            marks.append("excessive_back")
        if getattr(row, "n_loop_removed", 0.0) > thresholds.loops_p90:
            marks.append("navigation_loop")
        if getattr(row, "revisit_ratio", 0.0) > thresholds.revisit_p90:
            marks.append("screen_thrash")
        if getattr(row, "span_seconds", 0.0) > thresholds.span_p95:
            marks.append("slow_journey")
        flags.append("|".join(marks))
    return pd.Series(flags, index=frame.index, dtype="string")


def apply_hierarchy(
    frame: pd.DataFrame,
    *,
    soft_pass: np.ndarray,
    recurring_support: int,
) -> pd.DataFrame:
    """Apply primary/secondary/unassigned labels without replacing fitted IDs."""

    if len(soft_pass) != len(frame):
        raise ValueError("soft_pass length must match frame length")
    out = frame.copy()
    primary = out["c_cluster"].to_numpy(dtype=int) != -1
    b_existing = (~primary) & (out["b_cluster"].to_numpy(dtype=int) != -1)
    borderline = (
        (~primary)
        & (~b_existing)
        & np.asarray(soft_pass, dtype=bool)
    )
    unresolved = (~primary) & (~b_existing) & (~borderline)

    out["primary_cluster"] = out["c_cluster"].astype("Int64")
    secondary = pd.Series(pd.NA, index=out.index, dtype="Int64")
    secondary.loc[b_existing] = out.loc[b_existing, "b_cluster"].astype("Int64")
    secondary.loc[borderline] = out.loc[borderline, "nearest_b_cluster"].astype("Int64")
    out["secondary_cluster"] = secondary

    assignment = np.full(len(out), "unassigned_novel", dtype=object)
    assignment[primary] = "C_primary"
    assignment[b_existing] = "B_existing_secondary"
    assignment[borderline] = "B_borderline_secondary"
    recurring = unresolved & (
        out["exact_sequence_noise_support"].fillna(0).to_numpy(dtype=int)
        >= recurring_support
    )
    friction = unresolved & (~recurring) & out["friction_flags"].fillna("").ne("").to_numpy()
    assignment[recurring] = "unassigned_recurring"
    assignment[friction] = "unassigned_friction"
    out["assignment_type"] = assignment

    confidence = np.full(len(out), "none", dtype=object)
    confidence[primary] = "primary"
    confidence[b_existing] = "secondary_fitted"
    confidence[borderline] = "secondary_soft"
    out["assignment_confidence"] = confidence

    reason = np.full(len(out), "novel_singleton", dtype=object)
    reason[primary] = "known_C_archetype"
    reason[b_existing] = "rare_archetype_from_B"
    reason[borderline] = "borderline_known_B"
    reason[recurring] = "rare_recurring_pattern"
    reason[friction] = "friction_candidate"
    out["noise_reason"] = reason

    keys = np.full(len(out), "UNKNOWN", dtype=object)
    keys[primary] = [f"C:{value}" for value in out.loc[primary, "c_cluster"]]
    keys[b_existing | borderline] = [
        f"B:{int(value)}" for value in secondary.loc[b_existing | borderline]
    ]
    out["effective_cluster_key"] = keys
    return out


def build_secondary_catalog(
    enriched: pd.DataFrame,
    b_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize B archetypes used to rescue C-noise journeys."""

    used = enriched[enriched["secondary_cluster"].notna()].copy()
    if used.empty:
        return pd.DataFrame(
            columns=[
                "secondary_cluster", "assigned_journeys", "fitted_B_assignments",
                "soft_B_assignments", "n_sessions", "n_devices", "share_of_C_noise",
            ]
        )
    used["secondary_cluster"] = used["secondary_cluster"].astype(int)
    c_noise_count = int((enriched["c_cluster"] == -1).sum())
    rows: list[dict[str, object]] = []
    for cluster, group in used.groupby("secondary_cluster", sort=True):
        rows.append(
            {
                "secondary_cluster": int(cluster),
                "assigned_journeys": int(len(group)),
                "fitted_B_assignments": int(
                    (group["assignment_type"] == "B_existing_secondary").sum()
                ),
                "soft_B_assignments": int(
                    (group["assignment_type"] == "B_borderline_secondary").sum()
                ),
                "n_sessions": int(group["session_id"].nunique()),
                "n_devices": int(group["device_id"].nunique()) if "device_id" in group else None,
                "share_of_C_noise": round(len(group) / max(c_noise_count, 1), 6),
            }
        )
    result = pd.DataFrame(rows)
    catalog = b_catalog[b_catalog["cluster"] != -1].rename(
        columns={
            "cluster": "secondary_cluster",
            "size": "B_catalog_size",
            "share": "B_catalog_share",
        }
    )
    keep = [
        column for column in (
            "secondary_cluster", "B_catalog_size", "B_catalog_share", "top_entry_token",
            "top_exit_token", "medoid_journey_id", "medoid_length", "medoid_path",
        ) if column in catalog.columns
    ]
    return (
        result.merge(catalog[keep], on="secondary_cluster", how="left", validate="one_to_one")
        .sort_values("assigned_journeys", ascending=False)
        .reset_index(drop=True)
    )
