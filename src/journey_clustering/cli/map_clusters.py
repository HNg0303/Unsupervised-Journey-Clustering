"""Put the HiFPT sitemap taxonomy_id on scored journeys of android and ios.

Reads ``journey_cluster_taxonomy.csv`` (built by ``journey-cluster-taxonomy``,
one sitemap id per cluster) and each platform's scores, then writes a copy per
platform with ``taxonomy_id`` right after the cluster column, the
``journey_summary`` layout. ``--with-names`` also adds business_family,
business_submodule and business_detail from the sitemap, for reading the file
without a join. Journeys of cluster -1 or of a cluster the file does not know
get no id. The output contains customer and device identifiers, so keep it
next to the scores.

Example:

    journey-map --scores-run output/scores/678/678_20260929_090904

    journey-map --scores-run output/scores/678/678_20260929_090904 --with-names \\
        --platforms android --input output/scores/678/678_20260929_090904/android \\
        --output /tmp/android_journeys_named.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from journey_clustering.cluster_mapping import DEFAULT_SITEMAP, load_cluster_taxonomy, load_sitemap, map_scores_file

PLATFORMS = ("android", "ios")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scores-run", type=Path, required=True, help="scores folder of one model run")
    parser.add_argument("--model-version", help="default: the scores run folder name")
    parser.add_argument("--taxonomy", type=Path, help="default: <scores-run>/journey_cluster_taxonomy.csv")
    parser.add_argument("--platforms", nargs="+", default=list(PLATFORMS), choices=PLATFORMS)
    parser.add_argument(
        "--input", type=Path,
        help="scores CSV, parquet file or folder (one platform only); "
        "default: <scores-run>/<platform>/<platform>_scores.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        help="named CSV (one platform only); default: <scores-run>/<platform>/<platform>_journeys_named.csv",
    )
    parser.add_argument("--with-names", action="store_true", help="also write the three sitemap names")
    parser.add_argument("--sitemap", type=Path, default=DEFAULT_SITEMAP, help="sitemap taxonomy CSV or JSON")
    parser.add_argument("--chunksize", type=int, default=200_000, help="rows read at a time")
    args = parser.parse_args()
    if (args.input or args.output) and len(args.platforms) != 1:
        parser.error("--input and --output need exactly one --platforms value")
    return args


def main() -> int:
    args = parse_args()
    run = args.scores_run.resolve()
    model_version = args.model_version or run.name
    taxonomy_path = (args.taxonomy or run / "journey_cluster_taxonomy.csv").resolve()
    sitemap = load_sitemap(args.sitemap)
    for platform in args.platforms:
        taxonomy = load_cluster_taxonomy(taxonomy_path, platform=platform, model_version=model_version)
        source = args.input or run / platform / f"{platform}_scores.csv"
        target = args.output or run / platform / f"{platform}_journeys_named.csv"
        missing = sorted({row["taxonomy_id"] for row in taxonomy.values()} - set(sitemap) - {""})
        if missing:
            raise SystemExit(f"{platform}: taxonomy_id not in the sitemap: {missing[:10]}")
        result = map_scores_file(
            source, target, taxonomy, sitemap=sitemap if args.with_names else None, chunksize=args.chunksize
        )
        print(
            f"{platform}: {result['rows']:,} journeys, {result['rows_without_id']:,} without taxonomy_id "
            f"(cluster -1 or {result['unknown_cluster_rows']:,} in unknown clusters) -> {target}"
        )
        for module, count in sorted(result["per_module"].items(), key=lambda item: -item[1])[:5]:
            print(f"  {module} {sitemap[module]['business_family']}: {count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
