"""Export the HiFPT sitemap taxonomy JSON as the `journey_sitemap_taxonomy` table.

One row per module, submodule and feature: taxonomy_id, business_family
(module), business_submodule (submodule), business_detail (feature).

Example:

    journey-sitemap --input hifpt-journey-taxonomy.json --output taxonomy/hifpt_sitemap_taxonomy.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from journey_clustering.cluster_mapping import DEFAULT_SITEMAP, load_sitemap, write_sitemap


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="sitemap taxonomy JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_SITEMAP)
    args = parser.parse_args()
    sitemap = load_sitemap(args.input)
    write_sitemap(sitemap, args.output)
    levels = [sum(1 for key in sitemap if key.count(".") == depth) for depth in range(3)]
    print(f"wrote {len(sitemap):,} ids ({levels[0]} modules, {levels[1]} submodules, {levels[2]} features) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
