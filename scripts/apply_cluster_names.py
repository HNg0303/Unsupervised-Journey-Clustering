"""Attach human names to every cluster artefact, and audit for drift.

    python scripts/apply_cluster_names.py
    python scripts/apply_cluster_names.py --audit

Writes `*_catalog_named.csv` and `*_labels_named.csv` next to the originals, plus
one family roll-up per platform/route.

`--audit` checks the names still describe the run: any cluster id present in the
catalog but missing from `cluster_names.py` (or vice versa) is reported. Refitting
the model reshuffles ids, so run this after every refit before quoting a name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Rule_based.cluster_names import NAMES, apply_names, family_summary  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "clusters"
COMBOS = [("android", "a_tfidf"), ("android", "b_prefixspan"),
          ("ios", "a_tfidf"), ("ios", "b_prefixspan")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    drift = 0
    for platform, route in COMBOS:
        catalog = pd.read_csv(OUT_DIR / f"{platform}_{route}_catalog.csv")
        known = set(NAMES[(platform, route)])
        found = set(catalog["cluster"].astype(int))

        missing, extra = sorted(found - known), sorted(known - found)
        if missing or extra:
            drift += 1
            print(f"[DRIFT] {platform}/{route}: unnamed={missing} stale={extra}")
        elif args.audit:
            print(f"[ok] {platform}/{route}: all {len(found)} clusters named")

        if args.audit:
            continue

        named = apply_names(catalog, platform, route)
        named.to_csv(OUT_DIR / f"{platform}_{route}_catalog_named.csv", index=False)

        labels = pd.read_csv(OUT_DIR / f"{platform}_{route}_labels.csv")
        apply_names(labels, platform, route).to_csv(
            OUT_DIR / f"{platform}_{route}_labels_named.csv", index=False
        )

        fam = family_summary(catalog, platform, route)
        fam.to_csv(OUT_DIR / f"{platform}_{route}_families.csv", index=False)
        print(f"\n=== {platform} / {route} ===")
        print(fam.to_string(index=False))

    if args.audit:
        print("\nno drift" if not drift else f"\n{drift} catalog(s) out of sync with cluster_names.py")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
