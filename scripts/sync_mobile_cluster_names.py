"""Sync shareholder catalog names into an existing mobile class mapping.

    python scripts/sync_mobile_cluster_names.py --platform android
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def enrich_cluster_names(mapping: dict, platform: str, catalog_dir: Path) -> dict:
    catalog_path = catalog_dir / "shareholder_cluster_catalog_vi.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    platform_entry = next(
        (item for item in catalog.get("platforms", []) if item.get("platform") == platform),
        None,
    )
    if platform_entry is None:
        raise ValueError(f"platform {platform!r} not found in shareholder catalog")
    english_catalog_path = catalog_path.with_name("shareholder_cluster_catalog.json")
    english_catalog = json.loads(english_catalog_path.read_text(encoding="utf-8"))
    english_platform = next(
        (item for item in english_catalog.get("platforms", []) if item.get("platform") == platform),
        None,
    )
    if english_platform is None:
        raise ValueError(f"platform {platform!r} not found in English shareholder catalog")
    family_codes = {
        int(item["cluster_id"]): item.get("business_family", "other")
        for item in english_platform.get("clusters", [])
    }
    classes = []
    for item in platform_entry.get("clusters", []):
        cluster = int(item["cluster_id"])
        name = item.get("cluster_name", f"Cluster {cluster}")
        classes.append({
            "cluster": cluster,
            "class_group_code": family_codes.get(cluster, "other"),
            "class_group": item.get("business_family", "khác"),
            "class_code": "unknown_journey" if cluster == -1 else f"cluster_{cluster}",
            "class_name": name,
            "cluster_name": name,
            "class_description": "Evidence-backed name from the current shareholder cluster catalog.",
            "naming_confidence": item.get("naming_confidence", "unknown"),
        })
    mapping["classes"] = sorted(classes, key=lambda row: int(row["cluster"]))
    mapping["cluster_name_source"] = catalog_path.relative_to(REPO_ROOT).as_posix()
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=["android", "ios"])
    parser.add_argument(
        "--cluster-dir",
        default="output/clusters_with_screen",
        help="directory containing the named shareholder catalogs",
    )
    args = parser.parse_args()

    catalog_dir = Path(args.cluster_dir)
    if not catalog_dir.is_absolute():
        catalog_dir = REPO_ROOT / catalog_dir
    source = REPO_ROOT / "output" / "mobile" / args.platform / "class_mapping.json"
    app_asset = REPO_ROOT / "mobile" / "android" / "app" / "src" / "main" / "assets" / "class_mapping.json"
    mapping = enrich_cluster_names(
        json.loads(source.read_text(encoding="utf-8")),
        args.platform,
        catalog_dir,
    )
    source.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.platform == "android":
        app_asset.parent.mkdir(parents=True, exist_ok=True)
        app_asset.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"synced {len(mapping.get('classes', []))} class names -> {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
