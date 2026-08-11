"""Stage 3 - Multi-resolution tokenisation.

One event, four tokens, each a different resolution of the same behaviour:

    exact  action@ManageModemVC#do_action/MODEM_TURN_ON_OFF
    L3     internet/modem/modem/toggle          (family/module/object/operation)
    L2     internet/modem                       (family/module)
    L1     internet                             (family)
    op     configure:toggle                     (stage:operation)

The exact token is faithful and unusable on its own: two thirds of the exact
vocabulary occurs in fewer than five journeys, and the Android and iOS spellings
of one screen never meet. The three semantic levels are computed by the
enrichment stage from the *structure* of the path, so `FsListConnectedDeviceVC`
and `internet_fprotect_screen/management_device/management_device_screen`
collapse onto the same `internet/device` - which is the whole point.

L1/L2/L3 are genuinely different strings here. They used to be three aliases of
the exact token, which made `--level L1` a no-op and the rare-token backoff
below a no-op with it.
"""

from __future__ import annotations

import pandas as pd

from .config import CanonizeConfig, TokenConfig
from .semantics import SEMANTIC_COLUMNS, annotate_semantics

RARE = "<rare>"
# Mirrors `production.TOKEN_SEPARATOR`; kept local so this module does not
# import the production adapter just to split a string.
TOKEN_SEPARATOR = "@"

# Channel name -> the column holding that resolution. `exact` is the production
# token; the other three are derived by `semantics`. Order is the resolution
# ladder, coarse last, and `fold_rare_tokens` backs off along it.
CHANNEL_COLUMNS: dict[str, str] = {
    "exact": "exact_token",
    "intent": "token_l3",
    "coarse": "token_l2",
    "family": "token_l1",
    "operation": "operation_token",
}

# `TokenConfig.level` selects the primary modelling token.
LEVEL_COLUMNS: dict[str, str] = {
    "EXACT": "exact_token",
    "L3": "token_l3",
    "L2": "token_l2",
    "L1": "token_l1",
}

TOKEN_COLUMNS: tuple[str, ...] = (
    "exact_token",
    "token_l1",
    "token_l2",
    "token_l3",
    "operation_token",
)


def level_column(level: str) -> str:
    """Column backing a `TokenConfig.level` value."""
    column = LEVEL_COLUMNS.get(str(level).upper())
    if column is None:
        raise ValueError(
            f"unknown token level {level!r}; expected one of {sorted(LEVEL_COLUMNS)}"
        )
    return column


def channel_column(channel: str) -> str:
    """Column backing a feature channel name."""
    column = CHANNEL_COLUMNS.get(str(channel).lower())
    if column is None:
        raise ValueError(
            f"unknown token channel {channel!r}; expected one of {sorted(CHANNEL_COLUMNS)}"
        )
    return column


def build_tokens(canon: pd.DataFrame, cfg: CanonizeConfig) -> pd.DataFrame:
    """Attach every token resolution to a canonical event frame.

    Runs the semantic enrichment stage first when it has not been run already,
    so callers that only want tokens (`score.JourneyScorer.prepare`) stay a
    one-liner. Source event columns are never modified.
    """
    out = canon if set(SEMANTIC_COLUMNS).issubset(canon.columns) else annotate_semantics(canon)
    out = out.copy()

    family = out["business_family"].astype(str)
    module = out["business_module"].astype(str)
    obj = out["business_object"].astype(str)
    operation = out["operation"].astype(str)
    stage = out["operation_stage"].astype(str)

    out["exact_token"] = out["event_token"].astype(str)
    out["token_l1"] = family
    out["token_l2"] = family + "/" + module
    out["token_l3"] = family + "/" + module + "/" + obj + "/" + operation
    out["operation_token"] = stage + ":" + operation
    return out


def select_level(tokens: pd.DataFrame, level: str) -> pd.Series:
    return tokens[level_column(level)]


def channel_sequences(
    sequences_by_column: dict[str, list[list[str]]], channels: tuple[str, ...]
) -> dict[str, list[list[str]]]:
    """Re-key per-column journey sequences by channel name."""
    return {name: sequences_by_column[channel_column(name)] for name in channels}


def _semantic_token(exact: str, channel: str) -> str:
    """Re-derive one semantic token from an exact `event_type@screen[#target]`.

    Anything that is not an exact token - a `<rare>` sentinel, or a token that
    the rare-token backoff already replaced with a coarser form - is returned
    unchanged. It is already at or below the requested resolution.
    """
    from .semantics import classify_event  # local: keeps import graph acyclic

    event_type, separator, remainder = exact.partition(TOKEN_SEPARATOR)
    if not separator or event_type not in ("view", "action"):
        return exact
    screen, _, target = remainder.partition("#")
    label = classify_event(event_type, screen, target)
    if channel == "coarse":
        return f"{label.business_family}/{label.business_module}"
    if channel == "intent":
        return (
            f"{label.business_family}/{label.business_module}/"
            f"{label.business_object}/{label.operation}"
        )
    if channel == "family":
        return label.business_family
    if channel == "operation":
        return f"{label.operation_stage}:{label.operation}"
    raise ValueError(f"cannot derive channel {channel!r} from an exact token")


def channels_from_exact_sequences(
    sequences: list[list[str]], channels: tuple[str, ...]
) -> dict[str, list[list[str]]]:
    """Rebuild semantic channels for journeys held only as exact sequences.

    The enrichment stage is a pure function of the event path, so a consumer
    that persisted nothing but the `sequence` column - `postprocess_cluster_runs`
    reads exactly that - can recover every coarser resolution without going back
    to the raw events.
    """
    return {
        name: [[_semantic_token(token, name) for token in seq] for seq in sequences]
        for name in channels
    }


def fold_rare_tokens(
    sequences: list[list[str]],
    coarse_sequences: list[list[str]],
    cfg: TokenConfig,
) -> tuple[list[list[str]], pd.DataFrame]:
    """Back off tokens that are rare *across journeys* to their coarser form.

    Document frequency is computed over journeys rather than events on purpose:
    a token that fires 200 times inside one journey is still a single piece of
    evidence and must not earn its own vocabulary entry.

    `coarse_sequences` is the same journey at the next resolution down the
    ladder, index-aligned. With real semantic levels the backoff now *means*
    something: a one-off deep path degrades to `internet/modem` instead of
    straight to `<rare>`.
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
    col = level_column(level)
    grouped = (
        tokens.groupby(col, sort=False)
        .agg(
            event_frequency=("record_id", "size"),
            session_frequency=("session_id", "nunique"),
            event_type=("event_type", "first"),
            segment_name=("segment_name", "first"),
            platform=("platform", "first"),
            business_family=("business_family", "first"),
            business_module=("business_module", "first"),
            mean_semantic_confidence=("semantic_confidence", "mean"),
            first_seen=("event_time", "min"),
            last_seen=("event_time", "max"),
        )
        .reset_index()
        .rename(columns={col: "token"})
        .sort_values("event_frequency", ascending=False)
        .reset_index(drop=True)
    )
    grouped["mean_semantic_confidence"] = grouped["mean_semantic_confidence"].round(3)
    grouped.insert(0, "token_id", grouped.index + 1)
    return grouped


def vocabulary_report(tokens: pd.DataFrame) -> pd.DataFrame:
    """Vocabulary size and concentration at every resolution.

    This is the evidence that the ladder is worth having: exact -> L3 -> L2 -> L1
    should show the vocabulary collapsing while top-100 coverage climbs.
    """
    rows: list[dict[str, object]] = []
    for name, column in CHANNEL_COLUMNS.items():
        counts = tokens[column].value_counts()
        rows.append(
            {
                "channel": name,
                "column": column,
                "vocabulary": int(counts.size),
                "singletons": int((counts == 1).sum()),
                "singleton_share": round(float((counts == 1).mean()), 4) if counts.size else 0.0,
                "top100_coverage": round(float(counts.head(100).sum() / counts.sum()), 4)
                if counts.size
                else 0.0,
            }
        )
    return pd.DataFrame(rows)
