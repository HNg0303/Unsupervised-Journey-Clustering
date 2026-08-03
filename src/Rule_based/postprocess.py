"""Stage 4 - Post-tokenisation sequence cleanup.

Turns tokenised events into one clean clickstream sequence per journey, plus the
small set of numeric features that clustering actually consumes. Nothing else.

Three kinds of redundancy live in this log; two are noise, one is *signal*:

1. Consecutive exact repeats (A A A) - navigation double-fires. Noise.
   Collapsed.
2. Cyclic repeats (A B A B A B) - user bouncing between two screens. Friction
   signal, not noise. Collapsed to one cycle, count kept as `n_loop_removed`.
3. OS chrome / boot screens (MainTabBarController, Splash, ...) - structural,
   not chosen by the user. Dropped from the sequence, counted as a feature.

Everything removed is recorded as a number, so no information is destroyed - it
moves from the sequence channel to the numeric channel.

Implementation note
-------------------
Collapsing works on *row positions*, never on token strings. One journey yields
one list of surviving row indices, and every token level (L1/L2/L3) is then read
off those same indices. That is why L1 and L2 sequences stay position-aligned
for free, and why this module no longer needs the run-mirroring helpers it used
to carry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PostProcessConfig


# --------------------------------------------------------------------------
# collapse primitives - all return POSITIONS to keep
# --------------------------------------------------------------------------
def keep_after_runs(seq: list[str]) -> list[int]:
    """A A A B B -> positions of the first A and the first B."""
    return [0] + [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]] if seq else []


def keep_after_cycles(
    seq: list[str], max_period: int = 4, min_repeats: int = 2
) -> tuple[list[int], int]:
    """A B A B A B -> positions of one A B. Returns (positions, events removed).

    Scans left to right preferring the shortest period, so a 2-cycle is never
    mistaken for a degenerate 4-cycle.
    """
    keep: list[int] = []
    removed = 0
    i, n = 0, len(seq)
    while i < n:
        matched = False
        for period in range(2, max_period + 1):
            if i + period * min_repeats > n:
                continue
            block = seq[i : i + period]
            repeats, j = 1, i + period
            while j + period <= n and seq[j : j + period] == block:
                repeats += 1
                j += period
            if repeats >= min_repeats:
                keep.extend(range(i, i + period))
                removed += (repeats - 1) * period
                i = j
                matched = True
                break
        if not matched:
            keep.append(i)
            i += 1
    return keep, removed


def clean_positions(
    seq: list[str], cfg: PostProcessConfig
) -> tuple[list[int], int, int]:
    """Apply both collapses. Returns (kept positions, n_dedup, n_loop)."""
    pos = list(range(len(seq)))

    if cfg.collapse_consecutive:
        pos = [pos[i] for i in keep_after_runs(seq)]
    n_dedup = len(seq) - len(pos)

    n_loop = 0
    if cfg.collapse_cycles:
        kept, n_loop = keep_after_cycles(
            [seq[i] for i in pos], cfg.max_cycle_period, cfg.min_cycle_repeats
        )
        pos = [pos[i] for i in kept]
    return pos, n_dedup, n_loop


def _as_bool(series: pd.Series) -> np.ndarray:
    """CSV round-trips booleans as the strings 'True'/'False'; astype(bool) on
    those returns True for both."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).to_numpy()


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def build_journey_sequences(
    df: pd.DataFrame,
    cfg: PostProcessConfig,
    token_col: str = "token_l2",
    extra_token_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, list[list[str]], dict[str, list[list[str]]]]:
    """Collapse each journey into one clean token sequence + numeric features.

    Input : event-level frame carrying `journey_id` and token columns
    Output: (journey-level frame, sequences, {level: sequences})

    `extra_token_cols` yields the same journeys at another resolution, index
    aligned with the primary one - needed by `tokens.fold_rare_tokens` for the
    rare-token backoff.
    """
    df = df.reset_index(drop=True)
    journey = df["journey_id"].to_numpy()
    # journeys are contiguous by construction (assigned over a sorted frame)
    starts = np.flatnonzero(np.r_[True, journey[1:] != journey[:-1]])
    ends = np.r_[starts[1:], len(df)]

    tok = df[token_col].to_numpy()
    extras = {c: df[c].to_numpy() for c in extra_token_cols}
    screen_class = df["screen_class"].to_numpy()
    is_action = (df["key"].to_numpy() == "Action")
    is_back = _as_bool(df["is_back"])
    gap = df["gap_prev_s"].to_numpy(dtype=float)
    dwell = df["duration_clip"].to_numpy(dtype=float)
    screen = df["screen"].to_numpy()
    # read back from CSV `ts` is a string, and only some rows carry fractional
    # seconds - a single inferred format silently NaTs the rest
    ts = pd.to_datetime(df["ts"], utc=True, format="mixed").to_numpy()
    session = df["session_id"].to_numpy()
    reason = df["boundary_reason"].to_numpy()

    def _col(name: str) -> np.ndarray:
        return df[name].to_numpy() if name in df.columns else np.full(len(df), None)

    device = _col("device_id")
    customer = _col("customer_id")
    os_col = _col("segmentation.segment")

    records: list[dict[str, object]] = []
    sequences: list[list[str]] = []
    extra_sequences: dict[str, list[list[str]]] = {c: [] for c in extra_token_cols}

    for lo, hi in zip(starts, ends):
        raw_len = int(hi - lo)
        cls = screen_class[lo:hi]

        keep = np.ones(raw_len, dtype=bool)
        if cfg.drop_chrome:
            keep &= cls != "chrome"
        if cfg.drop_boot:
            keep &= cls != "boot"
        rows = np.flatnonzero(keep) + lo

        pos, n_dedup, n_loop = clean_positions(tok[rows].tolist(), cfg)
        rows = rows[pos]
        seq = tok[rows].tolist()
        sequences.append(seq)
        for col, arr in extras.items():
            extra_sequences[col].append(arr[rows].tolist())

        n = len(seq)
        act = is_action[rows]
        gaps = gap[rows][1:] if n > 1 else np.zeros(0)
        dwells = dwell[rows][~act] if n else np.zeros(0)

        records.append(
            {
                # --- identity ------------------------------------------------
                "journey_id": journey[lo],
                "session_id": session[lo],
                "device_id": device[lo],
                "customer_id": customer[lo],
                "os": os_col[lo],
                "start_ts": ts[lo],
                "end_ts": ts[hi - 1],
                "boundary_reason": reason[lo],
                # --- size / what cleanup removed -----------------------------
                "n_events_raw": raw_len,
                "n_events_final": n,
                "n_unique_tokens": len(set(seq)),
                "n_dropped_screens": int(raw_len - keep.sum()),
                "n_dedup_removed": n_dedup,
                "n_loop_removed": n_loop,
                # --- behaviour -----------------------------------------------
                "action_ratio": round(float(act.mean()), 4) if n else 0.0,
                "back_rate": round(float(is_back[rows].mean()), 4) if n else 0.0,
                "revisit_ratio": round(1 - len(set(seq)) / n, 4) if n else 0.0,
                # --- time ----------------------------------------------------
                "span_seconds": round(
                    float((ts[hi - 1] - ts[lo]) / np.timedelta64(1, "s")), 3
                ),
                "total_dwell_s": round(float(dwells.sum()), 2),
                "median_gap_s": round(float(np.median(gaps)), 3) if gaps.size else 0.0,
                # --- endpoints -------------------------------------------------
                "entry_screen": screen[rows][0] if n else None,
                "exit_screen": screen[rows][-1] if n else None,
            }
        )

    return pd.DataFrame(records), sequences, extra_sequences


def postprocess_report(journeys: pd.DataFrame) -> pd.DataFrame:
    """Aggregate evidence of what cleanup removed."""
    raw = int(journeys["n_events_raw"].sum())
    final = int(journeys["n_events_final"].sum())
    return pd.DataFrame(
        [
            {"metric": "journeys", "value": len(journeys)},
            {"metric": "events_before_cleanup", "value": raw},
            {"metric": "chrome_boot_dropped", "value": int(journeys["n_dropped_screens"].sum())},
            {"metric": "consecutive_dupes_removed", "value": int(journeys["n_dedup_removed"].sum())},
            {"metric": "loop_events_removed", "value": int(journeys["n_loop_removed"].sum())},
            {"metric": "events_after_cleanup", "value": final},
            {"metric": "compression_ratio", "value": round(final / raw, 4) if raw else 0.0},
            {"metric": "journeys_with_loops", "value": int((journeys["n_loop_removed"] > 0).sum())},
            {"metric": "median_journey_length", "value": float(journeys["n_events_final"].median())},
        ]
    )


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os

    from .config import PostProcessConfig, SegmentConfig

    post_cfg = PostProcessConfig()
    seg_cfg = SegmentConfig()
    out_dir = "outputs/journeys"
    os.makedirs(out_dir, exist_ok=True)

    for platform in ("android", "ios"):
        events = pd.read_csv(f"{out_dir}/{platform}_journeys.csv", low_memory=False)
        journeys, sequences, _ = build_journey_sequences(events, post_cfg)

        # journeys too short to carry order information are useless for
        # clustering; drop them here so every downstream artefact is aligned
        keep = journeys["n_events_final"] >= seg_cfg.min_journey_length
        journeys = journeys.loc[keep].reset_index(drop=True)
        sequences = [s for s, k in zip(sequences, keep) if k]

        journeys["sequence"] = [" -> ".join(s) for s in sequences]
        journeys.to_csv(f"{out_dir}/{platform}_journey_features.csv", index=False)
        with open(f"{out_dir}/{platform}_journey_sequences.json", "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {"journey_id": jid, "tokens": seq}
                    for jid, seq in zip(journeys["journey_id"], sequences)
                ],
                fh,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n=== {platform} ===")
        print(postprocess_report(journeys).to_string(index=False))
