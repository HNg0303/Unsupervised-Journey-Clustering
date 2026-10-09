"""Build the `journey_cluster_taxonomy` table rows from reviewed cluster names.

The reviewed names come from the cluster labeling app export
(``<platform>_named_clusters.csv``). Before they are published they must belong
to the same model as the scored journeys, otherwise every join on
(model_version, platform, cluster_id) attaches a name to the wrong cluster.
The check compares each reviewed cluster with the shareholder catalog the
naming step wrote from the scores:

* the cluster IDs are the same set;
* the journey count of every cluster matches (when the export carries ``size``);
* the n-gram evidence the reviewer saw is the model's evidence (``top_ngrams``).

Example:

    journey-cluster-taxonomy \
        --scores-run output/scores/678/678_20260929_090904 \
        --output output/scores/678/678_20260929_090904/journey_cluster_taxonomy.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PLATFORMS = ("android", "ios")

# Column order of the `journey_cluster_taxonomy` database table.
TAXONOMY_COLUMNS = (
    "model_version", "platform", "cluster_id", "taxonomy_id", "cluster_name",
    "business_family", "business_submodule", "business_detail",
    "naming_confidence", "naming_source", "needs_review", "named_at",
)
# Cluster -1 holds the journeys no cluster accepted, a mix of every business
# area, so it always carries the noise name the naming step gives it.
NOISE_NAME = {
    "taxonomy_id": "",
    "cluster_name": "Chưa phân loại | Journey hỗn hợp/nhiễu",
    "business_family": "Chưa phân loại",
    "business_submodule": "Journey hỗn hợp/nhiễu",
    "business_detail": "",
    "naming_confidence": "low",
    "naming_source": "noise_cluster",
    "needs_review": 0,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def catalog_clusters(path: Path, platform: str) -> dict[int, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for block in payload.get("platforms", []):
        if block.get("platform") == platform:
            return {int(row["cluster_id"]): row for row in block["clusters"]}
    raise ValueError(f"{path} has no clusters for platform {platform!r}")


def alignment_errors(
    named: list[dict[str, str]], catalog: dict[int, dict[str, object]], platform: str
) -> list[str]:
    """Explain every way the reviewed names disagree with the scored model."""
    errors: list[str] = []
    ids = [int(row["cluster_id"]) for row in named]
    duplicated = sorted({value for value in ids if ids.count(value) > 1})
    if duplicated:
        errors.append(f"{platform}: duplicated cluster_id {duplicated[:10]}")
    missing = sorted(set(catalog) - set(ids))
    extra = sorted(set(ids) - set(catalog))
    if missing:
        errors.append(f"{platform}: {len(missing)} scored clusters have no name, e.g. {missing[:10]}")
    if extra:
        errors.append(f"{platform}: {len(extra)} named clusters are not in the model, e.g. {extra[:10]}")
    size_mismatch, evidence_mismatch = [], []
    for row in named:
        expected = catalog.get(int(row["cluster_id"]))
        if expected is None:
            continue
        if row.get("size") not in (None, "") and int(float(row["size"])) != int(expected["size"]):
            size_mismatch.append(int(row["cluster_id"]))
        if "top_ngrams" in row and row["top_ngrams"] != str(expected.get("top_mass_ngrams", "")):
            evidence_mismatch.append(int(row["cluster_id"]))
    if size_mismatch:
        errors.append(f"{platform}: journey count differs for clusters {size_mismatch[:10]}")
    if evidence_mismatch:
        errors.append(f"{platform}: n-gram evidence differs for clusters {evidence_mismatch[:10]}")
    return errors


def utc_datetime(value: str) -> str:
    """ISO timestamp from the labeling app -> UTC DATETIME text for MariaDB."""
    if not value:
        return ""
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def taxonomy_rows(
    named: list[dict[str, str]], model_version: str, platform: str
) -> list[dict[str, object]]:
    rows = []
    for row in sorted(named, key=lambda item: int(item["cluster_id"])):
        rows.append({
            "model_version": model_version,
            "platform": platform,
            "cluster_id": int(row["cluster_id"]),
            "taxonomy_id": row.get("taxonomy_id", ""),
            "cluster_name": row["cluster_name"],
            "business_family": row["business_family"],
            "business_submodule": row.get("business_submodule", ""),
            "business_detail": row.get("business_detail", ""),
            "naming_confidence": row["naming_confidence"],
            "naming_source": row["naming_source"],
            "needs_review": int(str(row["needs_review"]).strip().lower() in {"1", "true"}),
            "named_at": utc_datetime(row.get("updated_at", "")),
        })
        if rows[-1]["cluster_id"] == -1:
            if rows[-1]["cluster_name"] != NOISE_NAME["cluster_name"]:
                print(f"{platform}: cluster -1 named {rows[-1]['cluster_name']!r}; using the noise name")
            rows[-1].update(NOISE_NAME)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scores-run", type=Path, required=True, help="scores folder of one model run")
    parser.add_argument("--model-version", help="default: the scores run folder name")
    parser.add_argument("--platforms", nargs="+", default=list(PLATFORMS), choices=PLATFORMS)
    parser.add_argument(
        "--named", type=Path, nargs="*", default=[],
        help="reviewed name exports; default: <scores-run>/<platform>/<platform>_named_clusters.csv",
    )
    parser.add_argument("--output", type=Path, help="default: <scores-run>/journey_cluster_taxonomy.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = args.scores_run.resolve()
    model_version = args.model_version or run.name
    named_paths = {path.name.split("_", 1)[0]: path for path in args.named}
    output = (args.output or run / "journey_cluster_taxonomy.csv").resolve()

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for platform in args.platforms:
        named_path = named_paths.get(platform, run / platform / f"{platform}_named_clusters.csv")
        catalog_path = run / platform / f"{platform}_taxonomy_shareholder_catalog.json"
        named = read_rows(named_path)
        errors += alignment_errors(named, catalog_clusters(catalog_path, platform), platform)
        rows += taxonomy_rows(named, model_version, platform)
        print(f"{platform}: {len(named):,} named clusters checked against {catalog_path.name}")
    if errors:
        raise SystemExit("reviewed names do not match the scored model:\n  " + "\n  ".join(errors))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TAXONOMY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    print(f"wrote {len(rows):,} rows for model {model_version} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
