"""Map the reviewed names in <platform>_named_clusters.csv onto <platform> journeys.

Reads the cluster labeling app export, checks every name against the HiFPT
sitemap taxonomy (taxonomy/hifpt_sitemap_taxonomy.csv) and writes the journeys
with these columns right after the cluster column:

    taxonomy_id, cluster_name, business_family, business_submodule, business_detail

A cluster named only to level 1 or 2 gets the module or submodule id. Cluster -1
and clusters missing from the export get "Chưa phân loại | Journey hỗn hợp/nhiễu"
and no id. Name columns left by older naming runs (cluster_name_en,
business_family_code, naming_confidence, ...) are replaced. The journeys file is
read 200,000 rows at a time. The output holds customer and device ids: keep it
local.

Examples:

    # both platforms of one run, default file names inside the run folder
    python scripts/map_named_clusters.py --scores-run output/scores/678/678_20260929_090904

    # explicit files for one platform
    python scripts/map_named_clusters.py --platform android \
        --named android_named_clusters.csv --journeys android_journeys.csv \
        --output android_journeys_named.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITEMAP = ROOT / "taxonomy" / "hifpt_sitemap_taxonomy.csv"
PLATFORMS = ("android", "ios")
LEVELS = ("business_family", "business_submodule", "business_detail")
NAME_COLUMNS = ("taxonomy_id", "cluster_name", *LEVELS)
# Name columns older naming runs wrote on every journey; they are replaced.
STALE_COLUMNS = (
    *NAME_COLUMNS, "cluster_name_en", "business_family_code", "naming_confidence",
    "naming_source", "needs_review",
)
# Cluster -1 holds the journeys no cluster accepted, a mix of every business area.
NOISE = {
    "taxonomy_id": "",
    "cluster_name": "Chưa phân loại | Journey hỗn hợp/nhiễu",
    "business_family": "Chưa phân loại",
    "business_submodule": "Journey hỗn hợp/nhiễu",
    "business_detail": "",
}
# Identifiers pandas would otherwise read as floats (6039276 -> 6039276.0).
ID_COLUMNS = {"journey_id": "string", "session_id": "string", "device_id": "string", "customer_id": "string"}


def load_sitemap(path: Path) -> dict[str, tuple[str, str, str]]:
    """taxonomy_id -> (module, submodule, feature) for every sitemap level."""
    if path.suffix.lower() == ".json":
        sitemap = {}
        for item in json.loads(path.read_text(encoding="utf-8"))["features"]:
            sitemap[item["module_id"]] = (item["module"].strip(), "", "")
            sitemap[item["submodule_id"]] = (item["module"].strip(), item["submodule"].strip(), "")
            sitemap[item["feature_id"]] = (item["module"].strip(), item["submodule"].strip(), item["feature"].strip())
        return sitemap
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["taxonomy_id"]: tuple(row[c] or "" for c in LEVELS) for row in csv.DictReader(handle)}


def load_names(path: Path, sitemap: dict[str, tuple[str, str, str]]) -> dict[int, dict[str, str]]:
    """cluster_id -> name columns, with the sitemap id of every named cluster."""
    by_names = {levels: taxonomy_id for taxonomy_id, levels in sitemap.items()}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names, errors = {}, []
    for row in rows:
        cluster_id = int(row["cluster_id"])
        if cluster_id in names:
            errors.append(f"cluster {cluster_id} is named twice")
            continue
        if cluster_id == -1:
            names[-1] = dict(NOISE)
            continue
        levels = tuple((row.get(c) or "").strip() for c in LEVELS)
        taxonomy_id = (row.get("taxonomy_id") or "").strip()
        if taxonomy_id and sitemap.get(taxonomy_id) != levels:
            errors.append(f"cluster {cluster_id}: {taxonomy_id} is {sitemap.get(taxonomy_id)}, not {levels}")
        taxonomy_id = taxonomy_id or by_names.get(levels, "")
        if not taxonomy_id:
            errors.append(f"cluster {cluster_id}: {' | '.join(levels)} is not in the sitemap")
        names[cluster_id] = {"taxonomy_id": taxonomy_id, "cluster_name": row["cluster_name"],
                             **dict(zip(LEVELS, levels))}
    if errors:
        raise SystemExit(f"{path.name}: names do not match the sitemap:\n  " + "\n  ".join(errors[:20]))
    names.setdefault(-1, dict(NOISE))
    return names


def read_chunks(path: Path, chunksize: int):
    if path.is_dir() or path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for file in sorted(path.rglob("*.parquet")) if path.is_dir() else [path]:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=chunksize):
                yield batch.to_pandas()
        return
    header = pd.read_csv(path, nrows=0).columns
    dtype = {column: kind for column, kind in ID_COLUMNS.items() if column in header}
    yield from pd.read_csv(path, chunksize=chunksize, dtype=dtype, low_memory=False)


def name_chunk(chunk: pd.DataFrame, names: dict[int, dict[str, str]]) -> tuple[pd.DataFrame, int]:
    cluster_column = next((c for c in ("cluster_id", "cluster") if c in chunk.columns), None)
    if cluster_column is None:
        raise SystemExit("journeys have no cluster_id or cluster column")
    out = chunk.drop(columns=[c for c in STALE_COLUMNS if c in chunk.columns])
    clusters = out[cluster_column].astype(int)
    rows = [names.get(cluster, names[-1]) for cluster in clusters]
    position = out.columns.get_loc(cluster_column) + 1
    for column in reversed(NAME_COLUMNS):
        out.insert(position, column, [row[column] for row in rows])
    return out, int((~clusters.isin(names)).sum())


def map_file(journeys: Path, output: Path, names: dict[int, dict[str, str]], chunksize: int) -> dict[str, int]:
    temporary = output.with_name(f".{output.name}.tmp")
    total = unknown = without_id = 0
    started = time.time()
    try:
        for index, chunk in enumerate(read_chunks(journeys, chunksize)):
            named, missing = name_chunk(chunk, names)
            named.to_csv(temporary, mode="w" if index == 0 else "a", header=index == 0, index=False)
            total += len(named)
            unknown += missing
            without_id += int(named["taxonomy_id"].eq("").sum())
            print(f"  {total:,} journeys ({time.time() - started:,.0f}s)", flush=True)
        if total == 0:
            raise SystemExit(f"{journeys} has no journeys")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"rows": total, "unknown": unknown, "without_id": without_id}


def default_journeys(folder: Path, platform: str) -> Path:
    for name in (f"{platform}_journeys.csv", f"{platform}_scores.csv"):
        if (folder / name).exists():
            return folder / name
    raise SystemExit(f"no {platform}_journeys.csv or {platform}_scores.csv in {folder}; pass --journeys")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scores-run", type=Path, help="run folder holding <platform>/ subfolders")
    parser.add_argument("--platform", choices=PLATFORMS, help="one platform; default: both")
    parser.add_argument("--named", type=Path, help="default: <run>/<platform>/<platform>_named_clusters.csv")
    parser.add_argument("--journeys", type=Path, help="default: <run>/<platform>/<platform>_journeys.csv or _scores.csv")
    parser.add_argument("--output", type=Path, help="default: next to the journeys, <platform>_journeys_named.csv")
    parser.add_argument("--sitemap", type=Path, default=DEFAULT_SITEMAP, help="sitemap taxonomy CSV or JSON")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    if not args.scores_run and not (args.platform and args.named and args.journeys):
        parser.error("pass --scores-run, or --platform with --named and --journeys")
    if (args.named or args.journeys or args.output) and not args.platform:
        parser.error("--named, --journeys and --output need --platform")
    return args


def main() -> int:
    args = parse_args()
    sitemap = load_sitemap(args.sitemap)
    for platform in [args.platform] if args.platform else PLATFORMS:
        folder = args.scores_run / platform if args.scores_run else None
        named_path = args.named or folder / f"{platform}_named_clusters.csv"
        journeys = args.journeys or default_journeys(folder, platform)
        output = args.output or journeys.with_name(f"{platform}_journeys_named.csv")
        names = load_names(named_path, sitemap)
        print(f"{platform}: {len(names):,} named clusters from {named_path.name}, all in the sitemap")
        result = map_file(journeys, output, names, args.chunksize)
        print(
            f"{platform}: {result['rows']:,} journeys -> {output} "
            f"({result['without_id']:,} without taxonomy_id, "
            f"{result['unknown']:,} in clusters missing from {named_path.name})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
