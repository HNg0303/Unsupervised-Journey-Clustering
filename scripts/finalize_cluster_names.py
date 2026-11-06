"""Final cluster names for a run (pipeline: human names + new rule-based v2 naming), with no unclassified cluster except noise (-1).

Priority per cluster:
1. the name a person set (business_review* in {platform}_named_clusters.csv);
2. the v2 rule-based name (taxonomy_naming_v2/cluster_mapping.csv);
3. for clusters v2 left "Không đủ bằng chứng": a fallback read only from that
   cluster's own top_ngrams / supporting_evidence in cluster_mapping.csv
   (naming_source=fallback_cluster_ngrams, needs_review=true).

Also validates the result (ids and names against the sitemap, human names kept)
and compares the v2 rules, run WITHOUT preserved names (--rule-only), against
the human names.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

NOISE_NAME = "Chưa phân loại | Journey hỗn hợp/nhiễu"
# Feature evidence first: the rule with most hits among the cluster's n-grams wins.
FEATURE_RULES = [
    ("M05.01.03", r"MODEM_RESET|reboot_modem"),
    ("M07.02.01", r"DoctorSmart|ErrorCheckingDrSmart|check_error"),
    ("M09.01", r"support_create|SupportCreator|Support_create"),
    ("M16.01", r"CSAT_RATING|SRConfirm|SURVEY_ALL"),
    ("M10.08.01", r"payment_history|PaymentHistory"),
    ("M10.03", r"prepaid"),
    ("M10.07", r"FGold|select_vouchers"),
    ("M03.02.04", r"E_INVOICE|EBill"),
    ("M03.02.01", r"Contract_Management|ManagerContract|E-contract|ListEcontract"),
    ("M03.01.03", r"choose_contract|change_contract|Contract_List|ChooseContract"),
    ("M05.02.08", r"wifi_schedule|WifiListSchedule"),
    ("M05.05", r"devices_tab|FsListConnectedDevice|management_device"),
    ("M08.04.01", r"tv_service_tab|Tv_box"),
    ("M08.01.01", r"camera_service_tab"),
    ("M03.07", r"restore_service_tab"),
    ("M12.02.01", r"Nav_promotion|loylaty_|loyalty_"),
    ("M11.01", r"web/shop"),
    ("M01.01", r"login(?!_with_vneid)|Authorization|RedirectUri|LoginVC|OTP"),
    ("M01.02", r"login_with_vneid"),
    ("M02.01.04", r"home_guest|HomeGuestVC|PopupInviteLogin"),
    ("M05", r"home_service_management|ServiceManageVC|other_manage_internet|modem_control|wifi_control|action_wifi"),
]
# Only navigation left: name by the screen the cluster keeps opening.
NAV_RULES = [
    ("M15.01", r"noti|Notification|DetailsNoti"),
    ("M02.01.02", r"adsview|popup|Popup|STWeb|banner"),
    ("M10.01.01", r"Nav_payement|PaymentHome|home/payment"),
    ("M01.04.01", r"Nav_profile|android/Account|PersonalVC|AccountHomeController"),
]
DEFAULT_ID = "M02.01.01"


def pick(text_items, rules):
    best, hits = None, 0
    for tid, rx in rules:
        n = sum(bool(re.search(rx, t)) for t in text_items)
        if n > hits:
            best, hits = tid, n
    return best, hits


def level(tid):
    return {1: "module", 2: "submodule", 3: "detail"}[len(tid.split("."))] if isinstance(tid, str) and tid else None


def agree(a, b):
    """Deepest sitemap level two ids share."""
    if not (isinstance(a, str) and a and isinstance(b, str) and b):
        return "rule_unclassified" if isinstance(a, str) and a else "n/a"
    pa, pb = a.split("."), b.split(".")
    k = 0
    while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
        k += 1
    if k == 0:
        return "khác module"
    if k == min(len(pa), len(pb)):
        return "khớp (" + level(".".join(pa[:k])) + ")"
    return "cùng " + level(".".join(pa[:k]))


NAMED_COLS = ["platform", "cluster_id", "taxonomy_id", "cluster_name", "business_family", "business_submodule",
              "business_detail", "naming_confidence", "naming_source", "needs_review", "score_share",
              "evidence_mass_coverage", "evidence_row_coverage", "supporting_evidence", "top_ngrams", "size", "share",
              "row_count_in_input", "updated_at"]


def write_platform_files(run, df):
    """<platform>/<platform>_named_clusters.csv (read by map_named_clusters, daily job, carry-over),
    <platform>/<platform>_cluster_mapping_final.csv, and <run>/journey_cluster_taxonomy.csv."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    v2 = pd.read_csv(run / "taxonomy_naming_v2/cluster_mapping.csv").set_index(["platform", "cluster_id"])
    tax_rows = []
    for p in ("android", "ios"):
        d = df[df.platform == p].copy()
        d.to_csv(run / p / f"{p}_cluster_mapping_final.csv", index=False, encoding="utf-8-sig")
        old = pd.read_csv(run / p / f"{p}_named_clusters.csv").set_index("cluster_id")
        for c in ("score_share", "evidence_mass_coverage", "evidence_row_coverage"):
            d[c] = [v2.loc[(p, i), c] if (p, i) in v2.index else None for i in d.cluster_id]
        for c in ("size", "share", "row_count_in_input"):
            d[c] = [old.loc[i, c] if i in old.index else None for i in d.cluster_id]
        d["updated_at"] = now
        d["needs_review"] = d.needs_review.map(lambda x: str(bool(x)).lower())
        d = d.sort_values("cluster_id")
        d[NAMED_COLS].to_csv(run / p / f"{p}_named_clusters.csv", index=False, encoding="utf-8-sig")
        t = d.assign(model_version=run.name, named_at=now.replace("T", " ")[:19],
                     needs_review=d.needs_review.eq("true").astype(int))
        tax_rows.append(t[["model_version", "platform", "cluster_id", "taxonomy_id", "cluster_name", "business_family",
                           "business_submodule", "business_detail", "naming_confidence", "naming_source", "needs_review",
                           "named_at"]])
    pd.concat(tax_rows).to_csv(run / "journey_cluster_taxonomy.csv", index=False, lineterminator="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rule-only", required=True, help="cluster_mapping.csv of v2 run without --preserve-names")
    ap.add_argument("--human", default=None, help="human_named_clusters.csv (default: <run>/human_named_clusters.csv, else {platform}_named_clusters.csv)")
    ap.add_argument("--sitemap", default="taxonomy/hifpt_sitemap_taxonomy.csv")
    a = ap.parse_args()
    run = Path(a.run)
    site = pd.read_csv(a.sitemap, dtype=str).fillna("").set_index("taxonomy_id")
    by_name = {(r.business_family, r.business_submodule, r.business_detail): i for i, r in site.iterrows()}

    def names(tid):
        r = site.loc[tid]
        return [p for p in (r.business_family, r.business_submodule, r.business_detail) if p]

    def id_of(row):
        if isinstance(row.taxonomy_id, str) and row.taxonomy_id:
            return row.taxonomy_id
        return by_name.get(tuple("" if pd.isna(row[c]) else row[c] for c in ("business_family", "business_submodule", "business_detail")))

    v2 = pd.read_csv(run / "taxonomy_naming_v2/cluster_mapping.csv", dtype={"taxonomy_id": str})
    rule = pd.read_csv(a.rule_only, dtype={"taxonomy_id": str}).set_index(["platform", "cluster_id"])
    hpath = Path(a.human) if a.human else run / "human_named_clusters.csv"
    if hpath.exists():
        human = pd.read_csv(hpath, dtype={"taxonomy_id": str})
    else:
        human = pd.concat([pd.read_csv(run / p / f"{p}_named_clusters.csv", dtype={"taxonomy_id": str}) for p in ("android", "ios")])
    human = human[human.naming_source.astype(str).str.startswith("business_review")].set_index(["platform", "cluster_id"])

    rows = []
    for r in v2.itertuples():
        key = (r.platform, r.cluster_id)
        fam, sub, det = (("" if pd.isna(x) else x) for x in (r.business_family, r.business_submodule, r.business_detail))
        tid = r.taxonomy_id if isinstance(r.taxonomy_id, str) else by_name.get((fam, sub, det))
        out = dict(platform=r.platform, cluster_id=r.cluster_id, taxonomy_id=tid, cluster_name=r.cluster_name,
                   business_family=fam, business_submodule=sub, business_detail=det,
                   naming_source=r.naming_source, naming_confidence=r.naming_confidence,
                   needs_review=str(r.needs_review).lower() == "true", fallback_hits="")
        if r.cluster_id == -1:
            out.update(taxonomy_id=None, cluster_name=NOISE_NAME, business_family="Chưa phân loại",
                       business_submodule="Journey hỗn hợp/nhiễu", business_detail="", naming_source="noise_cluster",
                       naming_confidence="low", needs_review=False, final_source="nhiễu (-1)")
        elif key in human.index:
            hh = human.loc[key]
            hfam, hsub, hdet = (("" if pd.isna(hh[c]) else hh[c]) for c in ("business_family", "business_submodule", "business_detail"))
            out.update(taxonomy_id=id_of(hh), cluster_name=hh.cluster_name, business_family=hfam, business_submodule=hsub,
                       business_detail=hdet, naming_source=hh.naming_source, naming_confidence=hh.naming_confidence,
                       final_source="người đặt (business_review)")
        elif r.naming_source == "insufficient_observed_evidence":
            items = [x for x in str(r.top_ngrams).split(" || ") if x and x != "nan"]
            items += [x for x in str(r.supporting_evidence).split(" || ") if x and x != "nan"]
            fid, hits = pick(items, FEATURE_RULES)
            src = "tín hiệu tính năng"
            if not fid:
                fid, hits = pick(items, NAV_RULES)
                src = "chỉ điều hướng"
            if not fid:
                fid, hits, src = DEFAULT_ID, 0, "mặc định Home"
            parts = names(fid)
            out.update(taxonomy_id=fid, cluster_name=" | ".join(parts),
                       business_family=parts[0], business_submodule=(parts + ["", ""])[1], business_detail=(parts + ["", ""])[2],
                       naming_source="fallback_cluster_ngrams", naming_confidence="low", needs_review=True,
                       fallback_hits=f"{src}: {hits}/{len(items)} n-gram", final_source="fallback từ n-gram của cụm")
        else:
            out["final_source"] = "rule v2 (" + r.naming_source + ")"
        # validator columns
        h = human.loc[key] if key in human.index else None
        ro = rule.loc[key] if key in rule.index else None
        out["human_taxonomy_id"] = None if h is None else id_of(h)
        out["human_name"] = None if h is None else h.cluster_name
        out["rule_v2_taxonomy_id"] = None if ro is None or not isinstance(ro.taxonomy_id, str) else ro.taxonomy_id
        if ro is not None and not isinstance(ro.taxonomy_id, str) and ro.naming_source not in ("insufficient_observed_evidence", "noise_cluster"):
            out["rule_v2_taxonomy_id"] = id_of(ro)
        out["rule_v2_name"] = None if ro is None else ro.cluster_name
        out["rule_vs_human"] = agree(out["human_taxonomy_id"], out["rule_v2_taxonomy_id"]) if h is not None and r.cluster_id != -1 else None
        noise = out["cluster_name"] == NOISE_NAME
        out["id_in_sitemap"] = (bool(out["taxonomy_id"]) and out["taxonomy_id"] in site.index) if not noise else None
        out["name_matches_sitemap"] = (" | ".join(names(out["taxonomy_id"])) == out["cluster_name"]) if out["id_in_sitemap"] else None
        out["human_name_kept"] = (out["cluster_name"] == out["human_name"]) if h is not None else None
        out["top_ngrams"] = r.top_ngrams
        out["supporting_evidence"] = r.supporting_evidence
        rows.append(out)
    df = pd.DataFrame(rows)
    dst = run / "taxonomy_naming_v2" / "cluster_mapping_final.csv"
    df.to_csv(dst, index=False, encoding="utf-8-sig")

    non_noise = df[df.cluster_name != NOISE_NAME]
    hum = df[df.human_name.notna() & (df.cluster_id != -1)]
    report = {
        "clusters": df.groupby("platform").size().to_dict(),
        "final_source": df.groupby(["platform", "final_source"]).size().unstack(0).fillna(0).astype(int).to_dict(),
        "unclassified_left_excluding_noise": int(non_noise.cluster_name.str.startswith("Chưa phân loại").sum()),
        "taxonomy_id_missing_excluding_noise": int(non_noise.taxonomy_id.isna().sum()),
        "taxonomy_id_not_in_sitemap": int((non_noise.id_in_sitemap == False).sum()),
        "name_not_matching_sitemap": [f"{r.platform}:{r.cluster_id}" for r in non_noise[non_noise.name_matches_sitemap == False].itertuples()],
        "human_named": int(len(hum)),
        "human_name_changed": [f"{r.platform}:{r.cluster_id}" for r in hum[hum.human_name_kept == False].itertuples()],
        "rule_v2_vs_human (chạy rule mới KHÔNG giữ tên người đặt)": hum.rule_vs_human.value_counts().to_dict(),
        "fallback_breakdown": df[df.naming_source == "fallback_cluster_ngrams"].fallback_hits.str.split(":").str[0].value_counts().to_dict(),
    }
    not_minus1 = df[df.cluster_id != -1]
    report["cluster_-1_is_noise"] = bool((df[df.cluster_id == -1].cluster_name == NOISE_NAME).all())
    report["unclassified_with_cluster_id_not_-1"] = [f"{r.platform}:{r.cluster_id}" for r in not_minus1[not_minus1.cluster_name.str.startswith("Chưa phân loại")].itertuples()]
    if report["unclassified_with_cluster_id_not_-1"] or report["human_name_changed"] or not report["cluster_-1_is_noise"]:
        raise SystemExit("validation failed: " + json.dumps(report, ensure_ascii=False))
    write_platform_files(run, df)
    (run / "taxonomy_naming_v2" / "validation_final.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
