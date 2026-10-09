"""Map the reviewed names in <platform>_named_clusters.csv onto <platform> journeys.

Reads the labeling app export directly (no journey_cluster_taxonomy.csv
needed), checks every name against the HiFPT sitemap, and writes the journeys
with these columns right after the cluster column:

    taxonomy_id, cluster_name, business_family, business_submodule, business_detail

A cluster named only to level 1 or 2 gets the module or submodule id. Cluster -1
and clusters missing from the export get "Chưa phân loại | Journey hỗn hợp/nhiễu"
and no id. The journeys file is read 200,000 rows at a time (CSV, parquet file
or parquet folder). The output holds customer and device ids: keep it local.

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from journey_clustering.cli.cluster_taxonomy import attach_sitemap_ids, read_rows, taxonomy_rows  # noqa: E402
from journey_clustering.cluster_mapping import DEFAULT_SITEMAP, load_sitemap, map_scores_file  # noqa: E402

PLATFORMS = ("android", "ios")
NAME_COLUMNS = ("taxonomy_id", "cluster_name", "business_family", "business_submodule", "business_detail")


def default_journeys(folder: Path, platform: str) -> Path:
    for name in (f"{platform}_journeys.csv", f"{platform}_scores.csv"):
        if (folder / name).exists():
            return folder / name
    if (folder / f"model_version={folder.parent.name}").exists():
        return folder / f"model_version={folder.parent.name}"
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
    parser.add_argument("--model-version", help="default: the run folder name, or the journeys' own value")
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
    run = args.scores_run.resolve() if args.scores_run else None
    model_version = args.model_version or (run.name if run else "")
    for platform in [args.platform] if args.platform else PLATFORMS:
        folder = run / platform if run else None
        named_path = args.named or folder / f"{platform}_named_clusters.csv"
        journeys = args.journeys or default_journeys(folder, platform)
        output = args.output or journeys.with_name(f"{platform}_journeys_named.csv")
        if journeys.is_dir():
            output = args.output or journeys.parent / f"{platform}_journeys_named.csv"

        rows = taxonomy_rows(read_rows(named_path), model_version, platform)
        errors = attach_sitemap_ids(rows, sitemap, platform)
        if errors:
            raise SystemExit("names not in the sitemap:\n  " + "\n  ".join(errors[:20]))
        taxonomy = {int(row["cluster_id"]): row for row in rows}
        if -1 not in taxonomy:
            taxonomy[-1] = taxonomy_rows([{"cluster_id": "-1", "cluster_name": "", "business_family": "",
                                           "naming_confidence": "", "naming_source": "", "needs_review": "0"}],
                                         model_version, platform)[0]
        result = map_scores_file(journeys, output, taxonomy, columns=NAME_COLUMNS, chunksize=args.chunksize)
        print(
            f"{platform}: {len(rows):,} named clusters -> {result['rows']:,} journeys "
            f"({result['rows_without_id']:,} without taxonomy_id, {result['unknown_cluster_rows']:,} "
            f"in clusters missing from {named_path.name}) -> {output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
