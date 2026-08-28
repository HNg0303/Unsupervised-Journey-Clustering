#!/usr/bin/env python3
"""Analyse what customers do after successfully signing an e-contract with VNeID.

The input is the named journey CSV.  The first successful VNeID e-contract
journey per customer is the anchor.  Later journeys are deduplicated at
customer × time-bucket × behaviour before percentages are calculated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="one named journey CSV (legacy single-file mode)")
    p.add_argument("--android", type=Path, help="named Android journey CSV")
    p.add_argument("--ios", type=Path, help="named iOS journey CSV")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--day-max", type=int, default=30)
    p.add_argument("--week-max", type=int, default=12)
    p.add_argument("--month-max", type=int, default=6)
    return p.parse_args()


def dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def is_signature(row: dict[str, str]) -> bool: # Journey Sequence : A@vneid@econtract 
    text = " ".join((row.get(k) or "") for k in ("sequence", "entry_token", "exit_token", "cluster_name", "mapping_function_code")).lower()
    vneid = "vneid" in text
    success = bool(re.search(r"signed_successfully|success_sign_econtract|sign_success|sign.*successful", text))
    return vneid and success


def behaviour(row: dict[str, str]) -> tuple[str, str, str, str] | None:
    code = (row.get("mapping_function_code") or "").strip()
    family = (row.get("business_family") or "").strip()
    sub = (row.get("business_submodule") or "").strip()
    detail = (row.get("business_detail") or "").strip()
    # Technical screens and noise are not treated as a customer business action.
    if not code or code.startswith("unclassified.") or family in {"Noise", "Hệ thống kỹ thuật", "Chưa phân loại"}:
        return None
    label = " | ".join(x for x in (family, sub, detail) if x)
    return code, family, sub, label


def bucket(delta_days: float, unit: str, max_bucket: int) -> int | None:
    if delta_days < 0:
        return None
    if unit == "day":
        value = int(delta_days)
    elif unit == "week":
        value = int(delta_days // 7)
    else:
        # Calendar-independent month approximation is explicit and stable:
        # month 0 = days 0–29, month 1 = days 30–59, etc.
        value = int(delta_days // 30)
    return value if value <= max_bucket else None


def main() -> int:
    a = parse_args()
    sources = [path.resolve() for path in (a.input, a.android, a.ios) if path is not None]
    if not sources:
        raise SystemExit("provide --input or at least one of --android/--ios")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    missing = [str(source) for source in sources if not source.exists()]
    if missing:
        raise SystemExit(f"input not found: {', '.join(missing)}")

    anchors: dict[str, tuple[datetime, str]] = {}
    signature_journeys = 0
    total_rows = 0
    # Pass 1: find the earliest successful VNeID signature per customer.
    for source in sources:
        with source.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                total_rows += 1
                if not row.get("customer_id") or not is_signature(row):
                    continue
                end = dt(row.get("end_ts", "")) or dt(row.get("start_ts", ""))
                if end is None:
                    continue
                signature_journeys += 1
                customer = row["customer_id"]
                old = anchors.get(customer)
                if old is None or end < old[0]:
                    anchors[customer] = (end, row.get("journey_id", ""))

    # customer × unit × bucket -> distinct behaviours seen for that user.
    user_bucket_behaviours: dict[tuple[str, str, int], set[tuple[str, str, str, str]]] = defaultdict(set)
    first_action: dict[str, datetime] = {}
    post_journeys = 0
    for source in sources:
        with source.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                customer = row.get("customer_id", "")
                anchor = anchors.get(customer)
                if anchor is None or row.get("journey_id", "") == anchor[1]:
                    continue
                start = dt(row.get("start_ts", ""))
                if start is None or start <= anchor[0]:
                    continue
                item = behaviour(row)
                if item is None:
                    continue
                delta_days = (start - anchor[0]).total_seconds() / 86400.0
                post_journeys += 1
                first_action[customer] = min(first_action.get(customer, start), start)
                for unit, limit in (("day", a.day_max), ("week", a.week_max), ("month", a.month_max)):
                    b = bucket(delta_days, unit, limit)
                    if b is not None:
                        user_bucket_behaviours[(customer, unit, b)].add(item)

    # User-level file: auditable grouping by customer_id.
    user_path = out / "vneid_econtract_post_signature_customer_summary.csv"
    with user_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["customer_id", "signature_anchor_ts", "signature_anchor_journey_id", "first_post_action_ts", "day_0_30_behaviours", "week_0_12_behaviours", "month_0_6_behaviours", "post_signature_behaviour_count"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        by_customer: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for (customer, unit, b), items in user_bucket_behaviours.items():
            by_customer[customer][unit].append(f"{b}: " + "; ".join(sorted({x[3] for x in items})))
        for customer in sorted(anchors):
            anchor, journey = anchors[customer]
            values = {u: " || ".join(sorted(v)) for u, v in by_customer.get(customer, {}).items()}
            w.writerow({"customer_id": customer, "signature_anchor_ts": anchor.isoformat(), "signature_anchor_journey_id": journey, "first_post_action_ts": first_action.get(customer, ""), "day_0_30_behaviours": values.get("day", ""), "week_0_12_behaviours": values.get("week", ""), "month_0_6_behaviours": values.get("month", ""), "post_signature_behaviour_count": len({x[3] for x in sum((list(s) for (c, _, _), s in user_bucket_behaviours.items() if c == customer), [])})})

    # Aggregate file: one row per time unit × elapsed bucket × behaviour.
    users = Counter()
    labels: dict[tuple[str, int, str], tuple[str, str, str]] = {}
    for (customer, unit, b), items in user_bucket_behaviours.items():
        for code, family, sub, label in items:
            users[(unit, b, code)] += 1
            labels[(unit, b, code)] = (family, sub, label)
    summary_path = out / "vneid_econtract_post_signature_behavior_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["time_unit", "elapsed_bucket", "bucket_label", "function_code", "business_family", "business_submodule", "behavior_name", "unique_users", "cohort_users", "pct_of_signature_cohort", "pct_of_active_users_in_bucket"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        active = Counter((unit, b) for (customer, unit, b) in user_bucket_behaviours)
        for (unit, b, code), count in sorted(users.items(), key=lambda x: (x[0][0], x[0][1], -x[1], x[0][2])):
            family, sub, label = labels[(unit, b, code)]
            w.writerow({"time_unit": unit, "elapsed_bucket": b, "bucket_label": f"{unit}_{b}", "function_code": code, "business_family": family, "business_submodule": sub, "behavior_name": label, "unique_users": count, "cohort_users": len(anchors), "pct_of_signature_cohort": round(100 * count / len(anchors), 4), "pct_of_active_users_in_bucket": round(100 * count / active[(unit, b)], 4)})

    # A compact family-level rollup is easier to read for the executive answer.
    family_users: Counter[tuple[str, int, str]] = Counter()
    family_seen: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for (customer, unit, b), items in user_bucket_behaviours.items():
        for _, family, _, _ in items:
            family_seen[(unit, b)].add((customer, family))
    for (unit, b), pairs in family_seen.items():
        for customer, family in pairs:
            family_users[(unit, b, family)] += 1
    family_path = out / "vneid_econtract_post_signature_family_summary.csv"
    with family_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["time_unit", "elapsed_bucket", "bucket_label", "business_family", "unique_users", "cohort_users", "pct_of_signature_cohort", "pct_of_active_users_in_bucket"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        active = Counter((unit, b) for (customer, unit, b) in user_bucket_behaviours)
        for (unit, b, family), count in sorted(family_users.items(), key=lambda x: (x[0][0], x[0][1], -x[1], x[0][2])):
            w.writerow({"time_unit": unit, "elapsed_bucket": b, "bucket_label": f"{unit}_{b}", "business_family": family, "unique_users": count, "cohort_users": len(anchors), "pct_of_signature_cohort": round(100 * count / len(anchors), 4), "pct_of_active_users_in_bucket": round(100 * count / active[(unit, b)], 4)})

    meta = {"sources": [str(source) for source in sources], "definition": "Earliest per-customer journey containing VNeID and successful e-contract signing markers; later journeys exclude technical/noise clusters and deduplicate behaviour per customer/time bucket.", "total_input_rows": total_rows, "signature_journeys": signature_journeys, "signature_customers": len(anchors), "post_signature_business_journeys": post_journeys, "day_max": a.day_max, "week_max": a.week_max, "month_max": a.month_max, "outputs": [str(user_path), str(summary_path), str(family_path)]}
    (out / "vneid_econtract_post_signature_analysis_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
