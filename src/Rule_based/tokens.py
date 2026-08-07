"""Stage 2 - Multi-resolution tokenisation.

One token per canonical event, emitted at three resolutions so that the same
pipeline can be clustered at a robust granularity and *explained* at a coarse
one:

    L1  screen only                V@iOS::HomeVC          A@iOS::HomeVC
    L2  screen + shallow action    A@iOS::HomeVC#Home/Nav_profile
    L3  screen + full action       A@iOS::HomeVC#home/home_service_management/
                                     internet_service_tab/click_modem_control

L3 is faithful but sparse (67% of exact triples occur fewer than 5 times).
L2 is the modelling default. L1 is the reporting/interpretation view.
"""

from __future__ import annotations

import pandas as pd

from .canonize import MISSING
from .config import CanonizeConfig, TokenConfig

RARE = "<rare>"


def _shallow_path(target: str, depth: int) -> str:
    """
        For a slash-delimited action path, return the first `depth` segments.
    """
    if target == MISSING:
        return MISSING # <none>
    parts = [p for p in target.split("/") if p]
    if len(parts) <= depth:
        return "/".join(parts) if parts else target
    return "/".join(parts[:depth]) + "/*"


def build_tokens(canon: pd.DataFrame, cfg: CanonizeConfig) -> pd.DataFrame:
    """Attach the production token (``event_type@segment_name``).

    Legacy level columns remain aliases so persisted scorers and reporting code
    have one migration path, but they no longer carry different semantics.
    """
    out = canon.copy()
    for column in ("token_l1", "token_l2", "token_l3"):
        out[column] = out["event_token"]
    return out


def select_level(tokens: pd.DataFrame, level: str) -> pd.Series:
    column = {"L1": "token_l1", "L2": "token_l2", "L3": "token_l3"}.get(level.upper())
    if column is None:
        raise ValueError(f"unknown token level {level!r}; expected L1, L2 or L3")
    return tokens[column]


def fold_rare_tokens(
    sequences: list[list[str]],
    coarse_sequences: list[list[str]],
    cfg: TokenConfig,
) -> tuple[list[list[str]], pd.DataFrame]:
    """Back off tokens that are rare *across journeys* to their coarser form.

    Document frequency is computed over journeys rather than events on purpose:
    a token that fires 200 times inside one journey is still a single piece of
    evidence and must not earn its own vocabulary entry.
    """
    df_counts: dict[str, int] = {}
    for seq in sequences:
        for tok in set(seq):
            df_counts[tok] = df_counts.get(tok, 0) + 1

    coarse_df: dict[str, int] = {}
    for seq in coarse_sequences:
        for tok in set(seq):
            coarse_df[tok] = coarse_df.get(tok, 0) + 1

    folded: list[list[str]] = []
    n_kept = n_backoff = n_rare = 0
    for seq, coarse in zip(sequences, coarse_sequences):
        row: list[str] = []
        for tok, ctok in zip(seq, coarse):
            if df_counts.get(tok, 0) >= cfg.min_journey_df:
                row.append(tok)
                n_kept += 1
            elif coarse_df.get(ctok, 0) >= cfg.min_journey_df:
                row.append(ctok)
                n_backoff += 1
            else:
                row.append(RARE)
                n_rare += 1
        folded.append(row)

    total = max(n_kept + n_backoff + n_rare, 1)
    stats = pd.DataFrame(
        [
            {
                "vocabulary_before": len(df_counts),
                "vocabulary_after": len({t for s in folded for t in s}),
                "tokens_kept": n_kept,
                "tokens_backed_off": n_backoff,
                "tokens_to_rare": n_rare,
                "backoff_share": round(n_backoff / total, 4),
                "rare_share": round(n_rare / total, 4),
            }
        ]
    )
    return folded, stats


def token_dictionary(tokens: pd.DataFrame, level: str) -> pd.DataFrame:
    """Frequency table for the chosen level, with provenance columns."""
    col = {"L1": "token_l1", "L2": "token_l2", "L3": "token_l3"}[level.upper()]
    grouped = (
        tokens.groupby(col, sort=False)
        .agg(
            event_frequency=("record_id", "size"),
            session_frequency=("session_id", "nunique"),
            event_type=("event_type", "first"),
            segment_name=("segment_name", "first"),
            platform=("platform", "first"),
            first_seen=("event_time", "min"),
            last_seen=("event_time", "max"),
        )
        .reset_index()
        .rename(columns={col: "token"})
        .sort_values("event_frequency", ascending=False)
        .reset_index(drop=True)
    )
    grouped.insert(0, "token_id", grouped.index + 1)
    return grouped


if __name__ == "__main__":
    import os
    canon_config = CanonizeConfig() #use default.
    token_config = TokenConfig() #use default.
    output_path = "outputs/journeys"
    os.makedirs(output_path, exist_ok=True)

    canonized_android_path = "outputs/journeys/canonized_events_android.csv"
    canonized_ios_path = "outputs/journeys/canonized_events_ios.csv"

    #tokenize android + ios canonized events
    canon_android_df = pd.read_csv(canonized_android_path)
    canon_ios_df = pd.read_csv(canonized_ios_path)
    tokens_android_df = build_tokens(canon_android_df, canon_config)
    tokens_ios_df = build_tokens(canon_ios_df, canon_config)

    #Save tokenized events to csv
    tokens_android_df.to_csv(os.path.join(output_path, "tokenized_events_android.csv"),
                            index=False)
    tokens_ios_df.to_csv(os.path.join(output_path, "tokenized_events_ios.csv"),
                        index=False)
