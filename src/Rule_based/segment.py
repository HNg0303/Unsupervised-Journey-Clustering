"""Stage 3 - Journey segmentation (session -> journey).

Why this stage is mandatory here, not optional:

`session_id` in this dataset is *not* a user session. Its wall-clock span runs
from 0 to 1832 minutes (median 3.2 min, p75 18.9 min, max 30 hours) and a single
session reaches 1524 events. Clustering whole sessions would mean clustering a
mixture of unrelated goals.

Two segmentation families are provided.

L0 - RULE-BASED (default, interpretable, no fitting)
    A boundary is placed when any of these fire:
      * session change
      * idle gap > tau           (data: p97.5 = 73s, p99 = 306s -> tau = 90s)
      * return to a root/hub screen after >= k events
      * auth state change (login / logout)
      * hard length cap

L1 - BRANCHING ENTROPY (statistical refinement, opt-in)
    Classic unsupervised word-segmentation signal, applied to token streams.
    For each position, measure H(next token | preceding n-gram). A goal
    boundary is where the next step becomes unpredictable: inside a journey the
    next screen is nearly determined; at the end of one, the user may go
    anywhere. Cut where entropy exceeds a high percentile. Purely data-driven,
    so it finds boundaries the rule list does not know about.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import ROOT_SCREENS, SegmentConfig

# Auth transitions are detected on the ACTION target, never on the screen name.
# Matching screens would fire on every `View LoginVC` / `View android/login`,
# which happens throughout guest browsing and is not a goal boundary.
AUTH_ACTION_MARKERS: tuple[str, ...] = (
    "continue_login",
    "login_with",
    "click_login",
    "log_out",
    "logout",
    "sign_out",
    "signin_success",
)


def _is_root(screen_bare: str) -> bool:
    return screen_bare in ROOT_SCREENS


def _is_auth_action(event_type: str, target: str) -> bool:
    if event_type != "Action":
        return False
    low = target.lower()
    return any(marker in low for marker in AUTH_ACTION_MARKERS)


# --------------------------------------------------------------------------
# L1 - branching entropy
# --------------------------------------------------------------------------
def branching_entropy(
    sequences: list[list[str]], order: int = 3
) -> dict[tuple[str, ...], float]:
    """H(next | context) for every observed context up to `order` tokens.

    Returns a mapping context-tuple -> entropy in nats. Contexts seen fewer than
    3 times are omitted (their entropy estimate is meaningless).
    """
    counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        for i in range(len(seq) - 1):
            for k in range(1, order + 1):
                if i - k + 1 < 0:
                    continue
                ctx = tuple(seq[i - k + 1 : i + 1])
                counts[ctx][seq[i + 1]] += 1

    entropy: dict[tuple[str, ...], float] = {}
    for ctx, nxt in counts.items():
        total = sum(nxt.values())
        if total < 3:
            continue
        entropy[ctx] = -sum(
            (c / total) * math.log(c / total) for c in nxt.values() if c > 0
        )
    return entropy


def entropy_boundary_scores(
    seq: list[str], entropy: dict[tuple[str, ...], float], order: int
) -> np.ndarray:
    """Per-position boundary score: entropy of the longest matching context."""
    scores = np.zeros(len(seq), dtype=float)
    for i in range(len(seq)):
        for k in range(min(order, i + 1), 0, -1):
            ctx = tuple(seq[i - k + 1 : i + 1])
            if ctx in entropy:
                scores[i] = entropy[ctx]
                break
    return scores


# --------------------------------------------------------------------------
# L0 - rule-based segmentation
# --------------------------------------------------------------------------
def assign_journeys(tokens: pd.DataFrame, cfg: SegmentConfig) -> pd.DataFrame:
    """Add `journey_id`, `journey_pos` and `boundary_reason` columns.

    Input  : canonical+tokenised events sorted by (session_id, ts) with L1/L2/L3 token columns.
    Output : same frame with journey assignment; one journey never spans
             two sessions.
    """
    df = tokens.reset_index(drop=True)

    session = df["session_id"].to_numpy()
    gap = df["gap_prev_s"].to_numpy()
    screen_bare = df["screen_bare"].to_numpy()
    event_type = df["key"].to_numpy()
    target = df["target"].to_numpy()

    journey_ids = np.empty(len(df), dtype=object)
    reasons = np.empty(len(df), dtype=object)
    positions = np.zeros(len(df), dtype=int)

    counter = 0
    since_cut = 0
    current = None

    for i in range(len(df)):
        new_session = i == 0 or session[i] != session[i - 1]
        reason = None

        # Check the rules in data production.

        if new_session:
            reason = "session_start"
        elif gap[i] > cfg.idle_gap_seconds:
            reason = "idle_gap"
        elif since_cut >= cfg.max_journey_length:
            reason = "length_cap"
        elif (
            cfg.cut_on_root_return
            and _is_root(screen_bare[i])
            and not _is_root(screen_bare[i - 1])
            and since_cut >= cfg.root_return_min_events
        ):
            reason = "root_return"
        elif (
            cfg.cut_on_auth_change
            and _is_auth_action(event_type[i], target[i])
            and not _is_auth_action(event_type[i - 1], target[i - 1])
        ):
            reason = "auth_change"

        if reason is not None:
            counter += 1
            current = f"J{counter:06d}"
            since_cut = 0

        journey_ids[i] = current
        reasons[i] = reason or ""
        positions[i] = since_cut
        since_cut += 1

    df["journey_id"] = journey_ids
    df["journey_pos"] = positions
    df["boundary_reason"] = reasons
    return df


def refine_with_entropy(
    df: pd.DataFrame, cfg: SegmentConfig, token_col: str = "token_l2"
) -> pd.DataFrame:
    """Add entropy-driven cuts on top of the rule-based assignment."""
    sequences = [g[token_col].tolist() for _, g in df.groupby("journey_id", sort=False)]
    entropy = branching_entropy(sequences, order=cfg.entropy_ngram_order)

    all_scores: list[np.ndarray] = []
    for _, g in df.groupby("journey_id", sort=False):
        all_scores.append(entropy_boundary_scores(g[token_col].tolist(), entropy, cfg.entropy_ngram_order))
    flat = np.concatenate(all_scores) if all_scores else np.zeros(0)
    df = df.copy()
    df["entropy_score"] = flat

    if not cfg.use_entropy_boundaries or len(flat) == 0:
        return df

    threshold = float(np.percentile(flat[flat > 0], cfg.entropy_percentile)) if (flat > 0).any() else np.inf

    new_ids: list[str] = []
    counter = 0
    current = None
    prev_journey = None
    since_cut = 0
    for journey, score, reason in zip(df["journey_id"], df["entropy_score"], df["boundary_reason"]):
        cut = journey != prev_journey
        # entropy spike marks the END of a unit, so cut on the following event
        if not cut and score >= threshold and since_cut >= cfg.min_journey_length:
            cut = True
        if cut:
            counter += 1
            current = f"J{counter:06d}"
            since_cut = 0
        new_ids.append(current)
        prev_journey = journey
        since_cut += 1

    df["journey_id_rule"] = df["journey_id"]
    df["journey_id"] = new_ids
    df["entropy_threshold"] = threshold
    return df


def segmentation_report(df: pd.DataFrame, cfg: SegmentConfig) -> pd.DataFrame:
    """Diagnostics for the chosen segmentation."""
    sizes = df.groupby("journey_id", sort=False).size()
    reasons = df.loc[df["boundary_reason"].ne(""), "boundary_reason"].value_counts()
    rows = [
        {
            "metric": "sessions",
            "value": int(df["session_id"].nunique()),
        },
        {"metric": "journeys", "value": int(sizes.size)},
        {"metric": "events", "value": int(len(df))},
        {"metric": "mean_journey_length", "value": round(float(sizes.mean()), 2)},
        {"metric": "median_journey_length", "value": float(sizes.median())},
        {"metric": "p90_journey_length", "value": float(sizes.quantile(0.90))},
        {"metric": "max_journey_length", "value": int(sizes.max())},
        {
            "metric": f"journeys_below_min_len_{cfg.min_journey_length}",
            "value": int((sizes < cfg.min_journey_length).sum()),
        },
    ]
    for reason, count in reasons.items():
        rows.append({"metric": f"cut_reason::{reason}", "value": int(count)})
    return pd.DataFrame(rows)


def sweep_idle_gap(
    tokens: pd.DataFrame, cfg: SegmentConfig, grid: tuple[float, ...] = (30, 60, 90, 120, 300, 900)
) -> pd.DataFrame:
    """Sensitivity of journey count/length to the idle-gap threshold."""
    rows = []
    for tau in grid:
        trial = SegmentConfig(**{**cfg.__dict__, "idle_gap_seconds": float(tau)})
        assigned = assign_journeys(tokens, trial)
        sizes = assigned.groupby("journey_id", sort=False).size()
        rows.append(
            {
                "idle_gap_seconds": tau,
                "journeys": int(sizes.size),
                "mean_length": round(float(sizes.mean()), 2),
                "median_length": float(sizes.median()),
                "share_len_lt_3": round(float((sizes < 3).mean()), 4),
                "share_len_gt_50": round(float((sizes > 50).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    segment_config = SegmentConfig() #use default.
    tokenized_android_path = "outputs/journeys/tokenized_events_android.csv"
    tokenized_ios_path = "outputs/journeys/tokenized_events_ios.csv"
    tokenized_android_df = pd.read_csv(tokenized_android_path)
    tokenized_ios_df = pd.read_csv(tokenized_ios_path)
    android_journeys_df = assign_journeys(tokenized_android_df, segment_config)
    ios_journeys_df = assign_journeys(tokenized_ios_df, segment_config)
    output_path = "outputs/journeys"
    os.makedirs(output_path, exist_ok=True)
    android_journeys_df.to_csv(os.path.join(output_path, "android_journeys.csv"), index=False)
    ios_journeys_df.to_csv(os.path.join(output_path, "ios_journeys.csv"), index=False)