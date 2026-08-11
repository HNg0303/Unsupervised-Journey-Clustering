"""Name C-primary and B-secondary clusters from medoids and ranked n-grams."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from scripts.build_shareholder_cluster_catalog import (  # noqa: E402
    BUSINESS_FAMILY_VI,
    SIGNAL_VI,
    cluster_entry,
    translate_cluster_name,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ngrams(path: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(float(row["cluster"]))].append(row)
    return grouped


def reference_name_map(reference_catalog: Path) -> dict[str, str]:
    """Map English classifier labels onto the supplied Vietnamese vocabulary."""

    vietnamese = read_json(reference_catalog)
    english_path = reference_catalog.with_name("shareholder_cluster_catalog.json")
    mapping: dict[str, str] = {}
    if english_path.exists():
        english = read_json(english_path)
        vi_platforms = {item["platform"]: item for item in vietnamese["platforms"]}
        for platform in english["platforms"]:
            vi_by_id = {
                int(item["cluster_id"]): item
                for item in vi_platforms[platform["platform"]]["clusters"]
            }
            for item in platform["clusters"]:
                cluster_id = int(item["cluster_id"])
                mapping[item["cluster_name"]] = vi_by_id[cluster_id]["cluster_name"]
    return mapping


def translate_signal(signal: str) -> str:
    if signal in SIGNAL_VI:
        return SIGNAL_VI[signal]
    prefix = "single unique action signal suppressed: "
    if signal.startswith(prefix):
        return "đã loại tín hiệu chỉ dựa trên một action duy nhất: " + signal[len(prefix):]
    return signal


def name_run(
    run_dir: Path,
    platform: str,
    namespace: str,
    canonical_names: dict[str, str],
) -> list[dict[str, Any]]:
    catalog = read_json(run_dir / f"{platform}_cluster_catalog.json")
    ngrams = read_ngrams(run_dir / f"{platform}_cluster_ngrams.csv")
    rows: list[dict[str, Any]] = []
    for record in catalog:
        cluster = int(record["cluster"])
        named = cluster_entry(record, ngrams.get(cluster, []), platform)
        evidence = named["evidence"]["top_ngrams"]
        english_name = named["cluster_name"]
        translated_name = translate_cluster_name(english_name)
        reference_name = canonical_names.get(english_name)
        # Repair an untranslated legacy value while retaining the reference
        # vocabulary whenever it already contains a Vietnamese display name.
        vietnamese_name = (
            translated_name
            if reference_name in (None, english_name) and translated_name != english_name
            else reference_name or translated_name
        )
        family_code = named["business_family"]
        english_signals = named["naming_basis"]["signals"]
        rows.append(
            {
                "namespace": namespace,
                "cluster": cluster,
                "effective_cluster_key": f"{namespace}:{cluster}",
                "cluster_name": vietnamese_name,
                "cluster_name_vi": vietnamese_name,
                "cluster_name_en": english_name,
                "business_family": BUSINESS_FAMILY_VI.get(family_code, family_code),
                "business_family_code": family_code,
                "naming_confidence": named["naming_confidence"],
                "naming_signals": [translate_signal(signal) for signal in english_signals],
                "naming_signals_en": english_signals,
                "single_action_guard_applied": any(
                    "single unique action signal suppressed" in signal
                    for signal in english_signals
                ),
                "medoid_journey_id": record.get("medoid_journey_id"),
                "medoid_path": record.get("medoid_path"),
                "top_entry_token": record.get("top_entry_token"),
                "top_exit_token": record.get("top_exit_token"),
                "top_ngrams": evidence,
                "source_catalog_size": int(record.get("size", 0)),
                "source_catalog_share": float(record.get("share", 0.0)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["naming_signals"] = " | ".join(item["naming_signals"])
        item["naming_signals_en"] = " | ".join(item["naming_signals_en"])
        item["top_ngrams"] = " | ".join(
            str(ngram.get("ngram", "")) for ngram in item["top_ngrams"]
        )
        flat.append(item)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def write_source_run_catalog(
    run_dir: Path,
    platform: str,
    namespace: str,
    names: list[dict[str, Any]],
) -> None:
    raw = read_json(run_dir / f"{platform}_cluster_catalog.json")
    by_cluster = {int(row["cluster"]): row for row in names}
    named = []
    for record in raw:
        naming = by_cluster[int(record["cluster"])]
        named.append(
            {
                **record,
                "namespace": namespace,
                "effective_cluster_key": naming["effective_cluster_key"],
                "cluster_name": naming["cluster_name"],
                "cluster_name_vi": naming["cluster_name_vi"],
                "cluster_name_en": naming["cluster_name_en"],
                "business_family": naming["business_family"],
                "business_family_code": naming["business_family_code"],
                "naming_confidence": naming["naming_confidence"],
                "naming_signals": naming["naming_signals"],
                "single_action_guard_applied": naming["single_action_guard_applied"],
                "top_ngrams": naming["top_ngrams"],
            }
        )
    (run_dir / f"{platform}_cluster_catalog_named.json").write_text(
        json.dumps(named, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(run_dir / f"{platform}_cluster_name_mapping.csv", names)
    (run_dir / f"{platform}_cluster_name_mapping.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postprocess-run",
        type=Path,
        default=Path(
            "output/journey_runs/"
            "L2_ng1-3_C-mcs100-ms5_B-mcs50-ms3_postprocessed"
        ),
    )
    parser.add_argument(
        "--platforms", nargs="+", default=["android", "ios"], choices=["android", "ios"]
    )
    parser.add_argument(
        "--reference-catalog",
        type=Path,
        default=Path("output/clusters_with_screen/shareholder_cluster_catalog_vi.json"),
        help="canonical vocabulary and naming convention",
    )
    args = parser.parse_args()
    root = args.postprocess_run.resolve()
    config = read_json(root / "postprocess_config.json")
    b_run, c_run = Path(config["B_run"]), Path(config["C_run"])
    reference_catalog = args.reference_catalog.resolve()
    canonical_names = reference_name_map(reference_catalog)

    for platform in args.platforms:
        c_names = name_run(c_run, platform, "C", canonical_names)
        b_names = name_run(b_run, platform, "B", canonical_names)
        all_names = c_names + b_names
        payload = {
            "schema_version": "1.0",
            "platform": platform,
            "naming_basis": "medoid path + top three ranked cluster n-grams",
            "reference_catalog": str(reference_catalog),
            "single_action_policy": (
                "a sole action without a matching destination view is suppressed "
                "in favor of the coarser medoid surface"
            ),
            "sources": {"C_run": str(c_run), "B_run": str(b_run)},
            "clusters": all_names,
        }
        (root / f"{platform}_hierarchical_cluster_names.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_csv(root / f"{platform}_hierarchical_cluster_names.csv", all_names)
        write_source_run_catalog(c_run, platform, "C", c_names)
        write_source_run_catalog(b_run, platform, "B", b_names)

        c_by_id = {row["cluster"]: row for row in c_names if row["cluster"] != -1}
        primary = [c_by_id[key] for key in sorted(c_by_id)]
        (root / f"{platform}_primary_cluster_catalog_named.json").write_text(
            json.dumps(primary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        secondary_stats = read_json(root / f"{platform}_secondary_cluster_catalog.json")
        b_by_id = {row["cluster"]: row for row in b_names if row["cluster"] != -1}
        secondary_named: list[dict[str, Any]] = []
        for stats in secondary_stats:
            cluster = int(stats["secondary_cluster"])
            if cluster not in b_by_id:
                continue
            secondary_named.append({**stats, **b_by_id[cluster]})
        (root / f"{platform}_secondary_cluster_catalog_named.json").write_text(
            json.dumps(secondary_named, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        guarded = sum(bool(row["single_action_guard_applied"]) for row in all_names)
        print(
            f"{platform}: named {len(c_names) - 1} C clusters and {len(b_names) - 1} B clusters; "
            f"single-action guard applied to {guarded}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
