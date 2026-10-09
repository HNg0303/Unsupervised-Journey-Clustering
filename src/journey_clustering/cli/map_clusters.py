"""Attach the reviewed business names to scored journeys of android and ios.

Reads ``journey_cluster_taxonomy.csv`` (built by ``journey-cluster-taxonomy``)
and each platform's scores, then writes one named copy per platform with
taxonomy_id, cluster_name, business_family, business_submodule,
business_detail, naming_confidence, naming_source and needs_review right after
the cluster column. Journeys in a cluster the taxonomy does not know get the
cluster -1 noise name. The output contains customer and device identifiers,
so keep it next to the scores.

Example:

    journey-map --scores-run output/scores/678/678_20260929_090904

    journey-map --scores-run output/scores/678/678_20260929_090904 \\
        --platforms android --input output/scores/678/678_20260929_090904/android \\
        --output /tmp/android_journeys_named.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from journey_clustering.cluster_mapping import load_cluster_taxonomy, map_scores_file

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
    for platform in args.platforms:
        taxonomy = load_cluster_taxonomy(taxonomy_path, platform=platform, model_version=model_version)
        source = args.input or run / platform / f"{platform}_scores.csv"
        target = args.output or run / platform / f"{platform}_journeys_named.csv"
        result = map_scores_file(source, target, taxonomy, chunksize=args.chunksize)
        print(
            f"{platform}: {result['rows']:,} journeys named with {len(taxonomy):,} clusters "
            f"({result['unknown_cluster_rows']:,} in clusters without a name) -> {target}"
        )
        for family, count in sorted(result["per_business_family"].items(), key=lambda item: -item[1])[:5]:
            print(f"  {family}: {count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
