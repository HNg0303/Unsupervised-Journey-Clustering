"""Stage 4b - PrefixSpan: frequent sequential patterns over journey sequences.

This is the *second route* out of postprocessing, parallel to the TF-IDF n-gram
route in `features.py`:

    route A (features.py)   sequence -> TF-IDF over contiguous 1..3-grams -> SVD
    route B (this module)   sequence -> frequent SUBsequences -> binary matrix

The difference that matters: an n-gram must be contiguous, a PrefixSpan pattern
need not be. `HOME -> PAY -> CONFIRM` is one pattern even when a user detoured
through help and back on the way, so route B recovers the *goal* while route A
recovers the *exact path*. Noisy real clickstreams break n-grams constantly, so
the two views disagree, and that is the point of running both.

Two outputs:

  pattern_features()    journey x pattern binary matrix - drop-in alternative to
                        the TF-IDF block for clustering.
  journey_candidates()  each journey labelled with the most specific frequent
                        pattern it contains - a rule-based archetype guess that
                        needs no clustering at all, and a sanity check on the
                        clusters you do fit.

Algorithm: standard PrefixSpan (Pei et al. 2001) with pseudo-projection. Each
event is a single item, so there is no itemset-extension step - only sequence
extension, which is what makes this short.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import PrefixSpanConfig

Pattern = tuple[str, ...] # Pattern 


# --------------------------------------------------------------------------
# core mining
# --------------------------------------------------------------------------
def prefixspan(
    sequences: list[list[str]], min_support: int, max_length: int = 5
) -> dict[Pattern, int]:
    """Mine frequent subsequences. Returns {pattern: number of journeys}.

    Support counts *journeys containing the pattern*, not occurrences, so a
    single user looping 200 times cannot manufacture a pattern.
    """
    patterns: dict[Pattern, int] = {}

    def grow(prefix: Pattern, projection: list[tuple[int, int]]) -> None:
        # for each candidate next item, the projection it induces
        extensions: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for seq_id, start in projection:
            seq = sequences[seq_id]
            seen: set[str] = set()
            for p in range(start, len(seq)):
                token = seq[p]
                if token not in seen:  # first occurrence only -> one hit/journey
                    seen.add(token)
                    extensions[token].append((seq_id, p + 1))

        for token, proj in extensions.items():
            support = len(proj)
            if support < min_support:
                continue
            extended = prefix + (token,)
            patterns[extended] = support
            if len(extended) < max_length:
                grow(extended, proj)

    grow((), [(i, 0) for i in range(len(sequences))])
    return patterns


def _is_subsequence(short: Pattern, long: Pattern) -> bool:
    it = iter(long)
    return all(token in it for token in short)


def close_patterns(patterns: dict[Pattern, int]) -> dict[Pattern, int]:
    """Keep only closed patterns: no longer pattern has the same support.

    Without this the output is dominated by every prefix of the same journey
    shape, which is noise in a report and redundant columns in a matrix.
    """
    by_support: dict[int, list[Pattern]] = defaultdict(list)
    for pattern, support in patterns.items():
        by_support[support].append(pattern)

    closed: dict[Pattern, int] = {}
    for support, group in by_support.items():
        group.sort(key=len, reverse=True)
        for i, pattern in enumerate(group):
            if not any(
                len(other) > len(pattern) and _is_subsequence(pattern, other)
                for other in group[:i]
            ):
                closed[pattern] = support
    return closed


def mine_patterns(
    sequences: list[list[str]], cfg: PrefixSpanConfig
) -> pd.DataFrame:
    """Mine, filter and rank patterns. One row per pattern."""
    n = len(sequences)
    min_support = max(cfg.min_support_count, int(round(cfg.min_support * n)))

    patterns = prefixspan(sequences, min_support, cfg.max_pattern_length)
    if cfg.closed_only:
        patterns = close_patterns(patterns)

    # single-token patterns describe popularity, not behaviour
    patterns = {p: s for p, s in patterns.items() if len(p) >= cfg.min_pattern_length}

    rows = [
        {
            "pattern": " -> ".join(p),
            "tokens": list(p),
            "length": len(p),
            "support_count": s,
            "support": round(s / n, 4),
        }
        for p, s in patterns.items()
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # The cap is spent per length, not globally. Ranking purely by length would
    # keep only max-length patterns and throw away the short general ones that
    # most journeys actually match - route B would then cover almost nothing.
    lengths = table["length"].nunique()
    quota = max(1, -(-cfg.max_patterns // lengths))  # ceil
    table = (
        table.sort_values(["length", "support_count"], ascending=[True, False])
        .groupby("length", sort=True)
        .head(quota)
        .sort_values(["length", "support_count"], ascending=[False, False])
        .head(cfg.max_patterns)
        .reset_index(drop=True)
    )
    table.insert(0, "pattern_id", [f"P{i:04d}" for i in range(1, len(table) + 1)])
    return table


# --------------------------------------------------------------------------
# route B-1: patterns as features
# --------------------------------------------------------------------------
def contains(pattern: list[str], seq: list[str]) -> bool:
    it = iter(seq)
    return all(token in it for token in pattern)


def pattern_features(
    sequences: list[list[str]], table: pd.DataFrame
) -> pd.DataFrame:
    """journey x pattern binary matrix - the alternative to the TF-IDF block."""
    if table.empty:
        return pd.DataFrame(index=range(len(sequences)))
    matrix = np.zeros((len(sequences), len(table)), dtype=np.int8)
    for j, tokens in enumerate(table["tokens"]):
        for i, seq in enumerate(sequences):
            if contains(tokens, seq):
                matrix[i, j] = 1
    return pd.DataFrame(matrix, columns=table["pattern_id"].tolist())


# --------------------------------------------------------------------------
# route B-2: patterns as journey candidates
# --------------------------------------------------------------------------
def journey_candidates(
    sequences: list[list[str]], table: pd.DataFrame
) -> pd.DataFrame:
    """Label each journey with the most specific pattern it contains.

    "Most specific" = longest, ties broken by rarest, because a long rare
    pattern says far more about what the user was doing than a short common one
    every journey matches.
    """
    if table.empty:
        return pd.DataFrame(
            {"candidate_id": [None] * len(sequences), "candidate": None, "n_patterns_matched": 0}
        )

    order = table.sort_values(
        ["length", "support_count"], ascending=[False, True]
    ).reset_index(drop=True)

    rows = []
    for seq in sequences:
        matched = [
            (row.pattern_id, row.pattern)
            for row in order.itertuples()
            if contains(row.tokens, seq)
        ]
        best = matched[0] if matched else (None, None)
        rows.append(
            {
                "candidate_id": best[0],
                "candidate": best[1],
                "n_patterns_matched": len(matched),
            }
        )
    return pd.DataFrame(rows)


def candidate_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate archetype: how many journeys it claims."""
    total = len(candidates)
    summary = (
        candidates.groupby(["candidate_id", "candidate"], dropna=False)
        .size()
        .reset_index(name="n_journeys")
        .sort_values("n_journeys", ascending=False)
        .reset_index(drop=True)
    )
    summary["share"] = (summary["n_journeys"] / total).round(4)
    return summary


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    cfg = PrefixSpanConfig()
    out_dir = "outputs/journeys"
    os.makedirs(out_dir, exist_ok=True)

    for platform in ("android", "ios"):
        path = f"{out_dir}/{platform}_journey_sequences.json"
        with open(path, encoding="utf-8") as fh:
            records = json.load(fh)
        journey_ids = [r["journey_id"] for r in records]
        sequences = [r["tokens"] for r in records]

        table = mine_patterns(sequences, cfg)
        features = pattern_features(sequences, table)
        candidates = journey_candidates(sequences, table)
        candidates.insert(0, "journey_id", journey_ids)
        features.insert(0, "journey_id", journey_ids)

        table.drop(columns=["tokens"]).to_csv(
            f"{out_dir}/{platform}_patterns.csv", index=False
        )
        features.to_csv(f"{out_dir}/{platform}_pattern_features.csv", index=False)
        candidates.to_csv(f"{out_dir}/{platform}_journey_candidates.csv", index=False)

        print(f"\n=== {platform} ===")
        print(f"journeys={len(sequences):,}  patterns={len(table):,}")
        matched = candidates["candidate_id"].notna().mean() if len(candidates) else 0.0
        print(f"journeys matching >=1 pattern: {matched:.1%}")
        if not table.empty:
            print("\ntop patterns by support:")
            print(
                table.nlargest(10, "support_count")[
                    ["pattern_id", "length", "support_count", "support", "pattern"]
                ].to_string(index=False)
            )
            print("\ntop candidate archetypes:")
            print(candidate_summary(candidates).head(10).to_string(index=False))
