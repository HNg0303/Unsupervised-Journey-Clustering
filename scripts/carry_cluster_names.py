"""Carry the business names of the last named model onto a newly trained model.

HDBSCAN cluster ids are local to one run, so a retrain cannot reuse names by
cluster_id. Instead the old scorer re-assigns the new model's training
journeys: every journey then has a new cluster and an old cluster, and each new
cluster takes the name its journeys mostly had before.

Per new cluster, counting only journeys the old model put in a named cluster
(old cluster -1 and unnamed old clusters are left out):

    coverage  covered journeys / sampled journeys. Below --min-coverage the
              cluster is new behaviour: step "new", needs_review.
    1 old_cluster  one old cluster holds >= --threshold: copy its name as is.
    2 taxonomy_id  old clusters summed by taxonomy_id; one id >= --threshold.
    3 submodule    ids summed by submodule (M05.06), then
    4 module       by module (M05); the first level >= --threshold wins.
    5 review       no level reaches it: the taxonomy_id leader is suggested
                   with needs_review = true.

A level is decided only when its leader is strictly ahead of the runner-up.
New cluster -1 keeps "Chưa phân loại | Journey hỗn hợp/nhiễu".

Writes, per platform, into --output-dir/<platform>/:

    <platform>_named_clusters.csv  every cluster that got a name or a suggestion,
                                   readable by scripts/map_named_clusters.py
    <platform>_carry_report.csv    every new cluster: step, shares, top candidates

Example:

    python scripts/carry_cluster_names.py \\
        --old-run output/partitioned_runs/678/678_20260929_090904 \\
        --old-names output/scores/678/678_20260929_090904 \\
        --new-run output/partitioned_runs/678/678_20261101_090000

At most --max-per-cluster journeys per new cluster are re-scored (default 2,000),
so a run of millions of journeys finishes in minutes; 0 re-scores all of them.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from map_named_clusters import (  # noqa: E402
    DEFAULT_SITEMAP, LEVELS, NOISE, PLATFORMS, load_names, load_sitemap, read_chunks,
)
from journey_clustering.score import JourneyScorer, _assign  # noqa: E402

STEPS = ("old_cluster", "taxonomy_id", "submodule", "module", "review", "new", "noise")
CONFIDENCE = {"old_cluster": "high", "taxonomy_id": "high", "submodule": "medium", "module": "low"}
# Token lists written by pandas are numpy reprs: ['view@HOME' 'view@android/Home'\n ...]
QUOTED = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")


def tokens(value: object) -> list[str]:
    """A token list from a parquet list, a numpy repr string or an "a -> b" string."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    if text.startswith("["):
        return [single or double for single, double in QUOTED.findall(text)]
    return text.split(" -> ") if text else []


def old_clusters(scorer: JourneyScorer, journeys: pd.DataFrame) -> np.ndarray:
    """The old model's cluster per journey, exactly as its production scoring assigns it."""
    sequences = [tokens(value) for value in journeys["sequence"]]
    names = sorted(getattr(scorer.vectorizer, "channel_weights", {}))
    channels = {
        name: [tokens(value) for value in journeys[f"channel_{name}"]]
        if f"channel_{name}" in journeys else sequences
        for name in names
    }
    matrix = scorer.vectorizer.transform(journeys, sequences, channels or None)
    labels, distance = _assign(matrix, scorer.centroids)
    return np.where(distance > scorer.thresholds.distance_p95, -1, labels)


def crosstab(
    journeys: Path, scorer: JourneyScorer, max_per_cluster: int, chunksize: int, seed: int
) -> tuple[Counter, dict[int, Counter]]:
    """Journey count per new cluster and (new cluster -> old cluster counts) on a sample."""
    sizes: Counter = Counter()
    pairs: dict[int, Counter] = defaultdict(Counter)
    sampled: Counter = Counter()
    rng = np.random.default_rng(seed)
    started = time.time()
    for chunk in read_chunks(journeys, chunksize):
        column = "cluster_id" if "cluster_id" in chunk else "cluster"
        chunk = chunk.iloc[rng.permutation(len(chunk))].reset_index(drop=True)
        clusters = chunk[column].astype(int).to_numpy()
        sizes.update(clusters.tolist())
        rank = chunk.groupby(column).cumcount().to_numpy()
        take = clusters != -1
        if max_per_cluster:
            already = np.array([sampled[c] for c in clusters])
            take &= already + rank < max_per_cluster
        if take.any():
            labels = old_clusters(scorer, chunk.loc[take].reset_index(drop=True))
            for new, old in zip(clusters[take].tolist(), labels.tolist()):
                pairs[new][int(old)] += 1
            sampled.update(clusters[take].tolist())
        print(f"  {sum(sizes.values()):,} journeys read, {sum(sampled.values()):,} re-scored "
              f"({time.time() - started:,.0f}s)", flush=True)
    return sizes, pairs


def leader(counts: Counter) -> tuple[str, float, bool]:
    """(key, share, strictly ahead of the runner-up)."""
    total = sum(counts.values())
    ranked = counts.most_common(2)
    if not ranked or total == 0:
        return "", 0.0, False
    ahead = len(ranked) == 1 or ranked[0][1] > ranked[1][1]
    return ranked[0][0], ranked[0][1] / total, ahead


def sitemap_name(taxonomy_id: str, sitemap: dict[str, tuple[str, str, str]]) -> dict[str, str]:
    levels = sitemap[taxonomy_id]
    return {"taxonomy_id": taxonomy_id, "cluster_name": " | ".join(level for level in levels if level),
            **dict(zip(LEVELS, levels))}


def decide(
    old_counts: Counter,
    names: dict[int, dict[str, str]],
    sitemap: dict[str, tuple[str, str, str]],
    threshold: float,
    min_coverage: float,
) -> dict[str, object]:
    sampled = sum(old_counts.values())
    covered = Counter({old: n for old, n in old_counts.items() if old != -1 and old in names})
    coverage = sum(covered.values()) / sampled if sampled else 0.0
    by_id: Counter = Counter()
    for old, n in covered.items():
        by_id[names[old]["taxonomy_id"]] += n
    total = sum(covered.values())
    candidates = " || ".join(
        f"{taxonomy_id} {n / total:.0%}" for taxonomy_id, n in by_id.most_common(3)
    ) if total else ""
    result = {"carry_coverage": round(coverage, 4), "carry_candidates": candidates,
              "carried_from_cluster": "", "carry_share": 0.0}

    def named(step: str, share: float, name: dict[str, str], source: int | str = "") -> dict[str, object]:
        return {**result, **name, "carry_step": step, "carry_share": round(share, 4),
                "carried_from_cluster": source}

    if coverage >= min_coverage:
        old, share, ahead = leader(covered)
        if ahead and share >= threshold:
            return named("old_cluster", share, {k: names[old][k] for k in ("taxonomy_id", "cluster_name", *LEVELS)}, old)
        submodules, modules = Counter(), Counter()
        for taxonomy_id, n in by_id.items():
            if len(taxonomy_id) >= 6:
                submodules[taxonomy_id[:6]] += n
            modules[taxonomy_id[:3]] += n
        for step, counts in (("taxonomy_id", by_id), ("submodule", submodules), ("module", modules)):
            taxonomy_id, _, ahead = leader(counts)
            # every covered journey is the denominator, also at submodule level
            share = counts[taxonomy_id] / total
            if ahead and share >= threshold and taxonomy_id in sitemap:
                return named(step, share, sitemap_name(taxonomy_id, sitemap))
    step = "review" if coverage >= min_coverage else "new"
    taxonomy_id, share, _ = leader(by_id)
    if taxonomy_id in sitemap:
        return named(step, share, sitemap_name(taxonomy_id, sitemap))
    return {**result, "carry_step": step, "taxonomy_id": "", "cluster_name": "", **{k: "" for k in LEVELS}}


def carry_platform(args: argparse.Namespace, platform: str, sitemap) -> None:
    scorer_path = args.old_run / platform / f"{platform}_journey_scorer.pkl"
    names_path = args.old_names / platform / f"{platform}_named_clusters.csv"
    journeys_path = args.new_run / platform / f"{platform}_journeys.csv"
    output = args.output_dir / platform
    named_path = output / f"{platform}_named_clusters.csv"
    report_path = output / f"{platform}_carry_report.csv"
    for path in (scorer_path, names_path, journeys_path):
        if not path.exists():
            raise SystemExit(f"{platform}: missing {path}")
    if not args.overwrite and (named_path.exists() or report_path.exists()):
        raise SystemExit(f"{platform}: {named_path} already exists; pass --overwrite")

    names = load_names(names_path, sitemap)
    names.pop(-1, None)
    named_ids = {old for old, name in names.items() if name["taxonomy_id"]}
    names = {old: names[old] for old in named_ids}
    print(f"{platform}: {len(names):,} named old clusters from {names_path}")
    scorer = JourneyScorer.load(scorer_path)
    sizes, pairs = crosstab(journeys_path, scorer, args.max_per_cluster, args.chunksize, args.seed)

    total = sum(sizes.values())
    rows = []
    for cluster in sorted(sizes):
        if cluster == -1:
            row = {**NOISE, "carry_step": "noise", "carry_share": 0.0, "carry_coverage": 0.0,
                   "carry_candidates": "", "carried_from_cluster": ""}
        else:
            row = decide(pairs[cluster], names, sitemap, args.threshold, args.min_coverage)
        step = row["carry_step"]
        rows.append({
            "platform": platform, "cluster_id": cluster,
            **{k: row[k] for k in ("taxonomy_id", "cluster_name", *LEVELS)},
            "naming_confidence": CONFIDENCE.get(step, "low"),
            "naming_source": f"carried:{args.old_run.name}:{step}",
            "needs_review": step in ("review", "new"),
            "size": sizes[cluster], "share": round(sizes[cluster] / total, 8),
            "sampled": sum(pairs[cluster].values()),
            **{k: row[k] for k in ("carry_step", "carry_share", "carry_coverage",
                                    "carry_candidates", "carried_from_cluster")},
        })
    report = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False, lineterminator="\n")
    report.loc[report["cluster_name"].ne("")].drop(
        columns=["sampled", "carry_share", "carry_coverage", "carry_candidates"]
    ).to_csv(named_path, index=False, lineterminator="\n")

    print(f"{platform}: {len(report):,} new clusters, {total:,} journeys")
    for step in STEPS:
        part = report[report["carry_step"] == step]
        if len(part):
            print(f"  {step:<12} {len(part):>6,} clusters  {part['size'].sum() / total:6.1%} journeys")
    lost = set(name["taxonomy_id"] for name in names.values()) - set(report["taxonomy_id"])
    print(f"  {len(lost):,} old taxonomy_id no longer named by any new cluster")
    print(f"  -> {named_path}\n  -> {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--old-run", type=Path, required=True,
                        help="training run of the named model: <platform>/<platform>_journey_scorer.pkl")
    parser.add_argument("--old-names", type=Path, required=True,
                        help="folder with <platform>/<platform>_named_clusters.csv of that model")
    parser.add_argument("--new-run", type=Path, required=True,
                        help="training run of the new model: <platform>/<platform>_journeys.csv")
    parser.add_argument("--output-dir", type=Path, help="default: --new-run")
    parser.add_argument("--platform", choices=PLATFORMS, help="one platform; default: both")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--max-per-cluster", type=int, default=2000, help="0 = re-score every journey")
    parser.add_argument("--sitemap", type=Path, default=DEFAULT_SITEMAP)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.new_run
    return args


def main() -> int:
    args = parse_args()
    sitemap = load_sitemap(args.sitemap)
    for platform in [args.platform] if args.platform else PLATFORMS:
        carry_platform(args, platform, sitemap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
