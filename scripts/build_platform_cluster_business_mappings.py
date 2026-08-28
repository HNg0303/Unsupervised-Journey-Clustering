#!/usr/bin/env python3
"""Create one reference-format business mapping CSV per platform."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/scores/pca48_ngrams12_500"

# Reuse the repository's detailed function crosswalk and function-tree parser.
spec = importlib.util.spec_from_file_location(
    "business_mapping_rules", ROOT / "scripts/build_cluster_business_mapping.py"
)
rules_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rules_module)
DETAILED_RULES = rules_module.RULES
FUNCTIONS = rules_module.function_tree()


def canonical_for(text: str) -> str | None:
    import re
    low = text.lower()
    for pattern, code in DETAILED_RULES:
        if re.search(pattern, low):
            return code
    return None


def read_level2_labels() -> dict[tuple[str, int], dict[str, str]]:
    path = BASE / "cluster_naming_audit.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row["platform"], int(row["cluster"])): row
            for row in csv.DictReader(handle)
        }


def detail_decision(level2_code: str, confidence: str, ngrams: list[dict], medoid: str):
    """Return a level-3 code only with recurring, parent-consistent evidence."""
    if confidence != "high" or level2_code.startswith("unclassified"):
        return None, 0.0, [], []
    max_mass = max([float(row["cluster_mass"]) for row in ngrams] or [1.0])
    scores = Counter()
    hits = Counter()
    evidence = defaultdict(list)
    rare_warnings = []
    for row in ngrams:
        code = canonical_for(row["ngram"])
        if not code or ".".join(code.split(".")[:2]) != level2_code:
            continue
        rank = int(row["rank"])
        mass = float(row["cluster_mass"])
        weight = (1 / math.sqrt(rank)) * (0.30 + 0.70 * mass / max_mass)
        scores[code] += weight
        hits[code] += 1
        evidence[code].append(row["ngram"])
        if mass < max_mass * 0.05:
            rare_warnings.append(row["ngram"])
    medoid_code = canonical_for(medoid)
    medoid_support = None
    if medoid_code and ".".join(medoid_code.split(".")[:2]) == level2_code:
        medoid_support = medoid_code
        scores[medoid_code] += 0.35

    if not scores:
        return None, 0.0, [], rare_warnings
    code, top = scores.most_common(1)[0]
    total = sum(scores.values())
    share = top / total if total else 0.0
    # Two common n-grams, or one common n-gram corroborated by the medoid.
    enough_sources = hits[code] >= 2 or (hits[code] >= 1 and medoid_support == code)
    # Require clear dominance among candidate third-level functions.
    if not enough_sources or share < 0.58:
        return None, share, evidence[code], rare_warnings
    return code, share, evidence[code], rare_warnings


def build(platform: str, labels: dict[tuple[str, int], dict[str, str]]) -> Path:
    model = BASE / platform / "model_version=latest" / f"platform={platform}" / "model_output"
    catalog = {
        int(row["cluster"]): row
        for row in json.loads((model / f"{platform}_cluster_catalog.json").read_text())
    }
    ngrams = defaultdict(list)
    with (model / f"{platform}_cluster_ngrams.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["rank"]) <= 8:
                ngrams[int(row["cluster"])].append(row)

    fields = [
        "cluster", "size", "share", "mapping_name", "function_code",
        "business_family", "business_submodule", "business_detail", "confidence",
        "evidence_share", "secondary_function_codes", "ngram_primary_function",
        "ngram_evidence_share", "all_8_ngrams_used", "dominant_cluster_intents",
        "top_mass_ngrams", "rare_token_caution", "medoid_path_support_only",
    ]
    rows = []
    for cluster in sorted(catalog):
        cat = catalog[cluster]
        label = labels[(platform, cluster)]
        family = label["cluster_name_level_1"]
        submodule = label["cluster_name_level_2"]
        level2_code = label["canonical_level_2_code"]
        confidence = label["naming_confidence"]
        detail_code, detail_share, detail_evidence, rare = detail_decision(
            level2_code, confidence, ngrams[cluster], cat.get("medoid_path", "")
        )
        detail = FUNCTIONS.get(detail_code, ("", "", ""))[2] if detail_code else ""
        function_code = detail_code or level2_code
        mapping_name = " | ".join(value for value in (family, submodule, detail) if value)

        code_scores = Counter()
        for ng in ngrams[cluster]:
            code = canonical_for(ng["ngram"])
            if code:
                code_scores[code] += math.sqrt(float(ng["cluster_mass"])) / math.sqrt(int(ng["rank"]))
        ranked_codes = [code for code, _ in code_scores.most_common()]
        ngram_primary = ranked_codes[0] if ranked_codes else ""
        ngram_share = (code_scores[ngram_primary] / sum(code_scores.values())) if code_scores else 0.0
        top_mass = sorted(
            ngrams[cluster], key=lambda row: (-float(row["cluster_mass"]), int(row["rank"]))
        )[:4]
        rows.append({
            "cluster": cluster,
            "size": cat["size"],
            "share": cat["share"],
            "mapping_name": mapping_name,
            "function_code": function_code,
            "business_family": family,
            "business_submodule": submodule,
            "business_detail": detail,
            "confidence": confidence,
            "evidence_share": round(detail_share, 4) if detail_code else "",
            "secondary_function_codes": "; ".join(code for code in ranked_codes if code != function_code)[:500],
            "ngram_primary_function": ngram_primary,
            "ngram_evidence_share": round(ngram_share, 4),
            "all_8_ngrams_used": len(ngrams[cluster]),
            # The current artifacts do not contain a cluster-level intent summary;
            # keep the reference column without fabricating evidence.
            "dominant_cluster_intents": "",
            "top_mass_ngrams": " || ".join(row["ngram"] for row in top_mass),
            "rare_token_caution": " || ".join(rare[:3]),
            "medoid_path_support_only": cat.get("medoid_path", ""),
        })

    output = model / f"{platform}_cluster_business_mapping.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "platform": platform,
        "output": str(output),
        "rows": len(rows),
        "with_level_3": sum(bool(row["business_detail"]) for row in rows),
        "two_levels_only": sum(not bool(row["business_detail"]) for row in rows),
    }, ensure_ascii=False))
    return output


def main():
    labels = read_level2_labels()
    for platform in ("ios", "android"):
        build(platform, labels)


if __name__ == "__main__":
    main()
