#!/usr/bin/env python3
"""Build evidence-weighted business names for Android journey clusters."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/score_400K_noPCA_minCluster100_minSample5/model_version=latest/platform=android"
MODEL = BASE / "model_output"
TREE = Path("/Users/hoangnguyen/Downloads/hifpt_function_tree.md")
OUT = MODEL / "android_cluster_business_mapping.csv"


def function_tree() -> dict[str, tuple[str, str, str]]:
    levels: dict[int, str] = {}
    result = {}
    pat = re.compile(r"^(\s*)- (.+?) \(`([^`]+)`\)")
    for line in TREE.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if not m:
            continue
        level = len(m.group(1)) // 2
        label, code = m.group(2).replace("**", ""), m.group(3)
        levels[level] = label
        if code.count(".") == 2:
            result[code] = (levels.get(0, ""), levels.get(1, ""), label)
    return result


# Ordered from specific to broad. These are semantic crosswalks from tracking
# vocabulary to the canonical function tree, not a transcription of screen names.
RULES = [
    (r"vneid.*(sign|bbdt|document)|bbdt.*vneid", "authentication.document.vneid_signing"),
    (r"vneid.*(consent|share)", "authentication.vneid.consent"),
    (r"vneid.*(open|deeplink|browser)", "authentication.vneid.open"),
    (r"vneid", "authentication.vneid.sso_login"),
    (r"cccd|identity.*confirm", "authentication.identity.cccd_confirmation"),
    (r"otp.*owner|owner.*otp", "authentication.document.owner_otp"),
    (r"login.*(error|issue|fail)|hifpt_lite.*login", "support.application.login_issue"),
    (r"auth|login|fpt.?id", "authentication.session.login"),
    (r"acceptance|bbnt", "contract.acceptance_record.detail"),
    (r"econtract.*(sign|confirm)|electronic.*(sign|confirm)", "contract.document.electronic_record_sign"),
    (r"econtract|electronic_record|contract_appendix", "contract.document.electronic_record_view"),
    (r"owner.*(transfer|change)|ownership", "contract.ownership.transfer"),
    (r"info_change|customer.*update|contract.*update", "contract.information.customer_update"),
    (r"contract.*(switch|select|choose)", "contract.information.switch"),
    (r"contract.*(list|view)|account/contract", "contract.information.view"),
    (r"auto.?pay|autopay", "payment.utility.auto_pay"),
    (r"prepaid", "payment.utility.prepaid"),
    (r"payment.*(history|transaction_history)|history.*payment", "payment.history.detail"),
    (r"e.?invoice", "payment.history.e_invoice"),
    (r"payment.*(method|card|wallet)|method.*select", "payment.checkout.method_selection"),
    (r"payment.*(promotion|fgold|gold)|fgold.*payment", "payment.checkout.promotion"),
    (r"payment.*(status|result|success|failed|processing)", "payment.transaction.status"),
    (r"payment.*(confirm|pay|checkout|qr)|checkout", "payment.checkout.confirm"),
    (r"invoice.*(select|selection)|bill.*select", "payment.invoice.invoice_selection"),
    (r"invoice|bill|payment/general|payment/behalf", "payment.invoice.invoice_list"),
    (r"relocation.*(status|tracking)", "service_management.relocation.status"),
    (r"relocation.*(address|policy)", "service_management.relocation.new_address"),
    (r"relocation|move.*service", "service_management.relocation.eligibility"),
    (r"upgrade|restore|suspend|terminate", "service_management.lifecycle.upgrade"),
    (r"service.*(status|issue_detection)|mirrorcle|csoc", "service_management.status.overview"),
    (r"wifi.*(rename|password|credential)", "device_management.internet.wifi_credentials"),
    (r"modem.*restart|reboot", "device_management.internet.modem_reboot"),
    (r"camera.*(issue|support|error)", "device_management.service_device.camera_support"),
    (r"tv.*(issue|support|error)", "device_management.service_device.tv_support"),
    (r"internet/(device|wifi|modem|parental_control|schedule|diagnostics)|management_device|fprotect", "device_management.internet.management"),
    (r"internet.*(issue|error|diagnostic)", "device_management.internet.issue_detection"),
    (r"support.*(chat|message/create)", "support.center.chat"),
    (r"support.*(history|detail)", "support.center.request_history"),
    (r"support.*(survey|nps)|survey.*rate", "feedback.post_support.nps"),
    (r"support/request.*(submit|create)|request.*create", "support.request.create"),
    (r"support/request.*(confirm|contact)", "support.request.confirm_contact"),
    (r"support/request|support/report", "support.request.describe"),
    (r"support|help|faq", "support.center.open"),
    (r"notification.*(setting|preference|update)", "engagement.notification.preference"),
    (r"notification.*(detail|open|deeplink)", "engagement.notification.detail"),
    (r"notification", "engagement.notification.list"),
    (r"voucher.*(redeem|exchange)|offer.*redeem", "commerce.loyalty.offer_redeem"),
    (r"voucher.*(detail|view)|offer.*detail", "commerce.loyalty.offer_detail"),
    (r"fgold|loyalty.*gold", "commerce.loyalty.fgold"),
    (r"loyalty|voucher|promotion|offer", "commerce.loyalty.offer_list"),
    (r"product.*(detail|view)|shop/catalog", "commerce.catalog.product_detail"),
    (r"order|registration|subscribe", "commerce.catalog.service_registration"),
    (r"shop|catalog", "commerce.catalog.recommendation"),
    (r"tarot|sao.?may", "utility.experience.tarot"),
    (r"game|checkin", "utility.experience.game"),
    (r"smart.?rental|nha.?tro", "utility.smart_rental.home"),
    (r"qr.*(wifi|prepaid)", "utility.qr.prepaid_wifi"),
    (r"qr", "utility.qr.checkin_reward"),
    (r"favourite|favorite", "engagement.home.exclusive_feature"),
    (r"home/customize", "utility.account_setting.account"),
    (r"nav_pay|nav_invoice", "navigation.main_tab.invoice"),
    (r"nav_support", "navigation.main_tab.support"),
    (r"nav_offer", "navigation.main_tab.offer"),
    (r"nav_account", "navigation.main_tab.account"),
    (r"home/navigation|nav_home", "navigation.main_tab.home"),
    (r"banner|marketing/ads", "engagement.home.banner_interaction"),
    (r"marketing/message", "engagement.notification.detail"),
    (r"home/general", "navigation.main_tab.home"),
    (r"chrome/(boot|container|popup)|splash|mainappactivity|homevc|homeviewcontroller|homecontroller|homeactivity", "unclassified.technical.internal_screen"),
]


def quoted_items(value: str) -> list[str]:
    return re.findall(r"'([^']+)'", value)


def canonical_for(text: str) -> str:
    low = text.lower()
    for pattern, code in RULES:
        if re.search(pattern, low):
            return code
    return "unclassified.dynamic.mixed_journey"


def technical_display_name(text: str) -> tuple[str, str, str]:
    """Give technical clusters an explicit operational name, never 'unclassified'."""
    low = text.lower()
    family = "Hệ thống kỹ thuật"
    if re.search(r"homevc|homeviewcontroller|homecontroller|mainappactivity|homeactivity", low):
        return family, "Container Trang chủ", "Home controller/container tracking"
    if re.search(r"webview|hiwebview", low):
        return family, "WebView và container", "WebView container tracking"
    if re.search(r"popup|window", low):
        return family, "Popup và cửa sổ hệ thống", "Popup/window tracking"
    if re.search(r"splash|boot|launch", low):
        return family, "Khởi động ứng dụng", "Splash/boot tracking"
    if re.search(r"<missing>|unknown", low):
        return family, "Tracking thiếu định danh", "Màn hình/controller chưa định danh"
    return family, "Màn hình và controller nội bộ", "Technical screen/controller tracking"


def main() -> None:
    taxonomy = function_tree()
    catalog = {int(x["cluster"]): x for x in json.loads((MODEL / "android_cluster_catalog.json").read_text())}
    ngrams: dict[int, list[dict[str, str]]] = defaultdict(list)
    with (MODEL / "android_cluster_ngrams.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            ngrams[int(r["cluster"])].append(r)

    intent_counts: dict[int, Counter[str]] = defaultdict(Counter)
    cluster_sessions: dict[int, int] = Counter()
    with (MODEL / "android_journeys.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            c = int(row["cluster"])
            cluster_sessions[c] += 1
            for intent in quoted_items(row["channel_intent"]):
                intent_counts[c][intent] += 1
    # Document frequency is the number of clusters containing an intent, not
    # the number of journeys. This keeps IDF positive and cluster-oriented.
    intent_docs = Counter(intent for counts in intent_counts.values() for intent in counts)

    n_clusters = len(catalog)
    rows = []
    for cluster in sorted(catalog):
        cat = catalog[cluster]
        intent_scores: Counter[str] = Counter()
        ngram_scores: Counter[str] = Counter()
        ngram_hits: Counter[str] = Counter()
        evidence_by_code: dict[str, list[str]] = defaultdict(list)
        # Cluster-wide intents: discount globally ubiquitous intents using IDF.
        for intent, count in intent_counts[cluster].items():
            specificity = math.log((n_clusters + 1) / (1 + intent_docs[intent])) + 0.35
            code = canonical_for(intent)
            penalty = 0.04 if code == "unclassified.dynamic.mixed_journey" else 0.55 if code == "unclassified.technical.internal_screen" else 1.0
            intent_scores[code] += count * specificity * penalty
            evidence_by_code[code].append(intent)
        # N-grams: recurrence/mass matters; a huge lift with tiny mass cannot dominate.
        for ng in ngrams[cluster]:
            mass = float(ng["cluster_mass"])
            lift = float(ng["lift"])
            code = canonical_for(ng["ngram"])
            # An unmatched n-gram is absence of semantic evidence, not positive
            # evidence that a journey is mixed. Keep its raw text for review.
            if code == "unclassified.dynamic.mixed_journey":
                continue
            # Mass is the main reliability term. Lift is capped/logged so a
            # spectacular but rare token cannot overrule repeated context.
            weight = math.sqrt(mass) * math.log1p(min(lift, 100.0)) / (1 + 0.10 * (int(ng["rank"]) - 1))
            penalty = 0.04 if code == "unclassified.dynamic.mixed_journey" else 0.55 if code == "unclassified.technical.internal_screen" else 1.0
            ngram_scores[code] += weight * penalty
            ngram_hits[code] += 1
            evidence_by_code[code].append(ng["ngram"])
        # Reward semantic recurrence across the eight n-grams. Combine
        # normalized scores so the much larger journey-event count cannot
        # numerically swamp the n-gram evidence.
        for c, hits in ngram_hits.items():
            ngram_scores[c] *= 1 + 0.12 * max(0, hits - 1)
        scores: Counter[str] = Counter()
        itotal, ntotal = sum(intent_scores.values()) or 1, sum(ngram_scores.values()) or 1
        for c, value in intent_scores.items():
            scores[c] += 0.20 * value / itotal
        for c, value in ngram_scores.items():
            scores[c] += 0.80 * value / ntotal
        ranked = scores.most_common()
        ngram_ranked = ngram_scores.most_common()
        code = ranked[0][0] if ranked else "unclassified.dynamic.mixed_journey"
        confidence_share = scores[code]
        confidence = "high" if confidence_share >= .58 else "medium" if confidence_share >= .35 else "low"
        if cluster == -1:
            code, confidence = "unclassified.dynamic.mixed_journey", "low"
        top_intents = [x for x, _ in intent_counts[cluster].most_common(6)]
        top_ng = sorted(ngrams[cluster], key=lambda x: (-float(x["cluster_mass"]), int(x["rank"])))[:4]
        rare = [x["ngram"] for x in ngrams[cluster] if float(x["cluster_mass"]) < 2.0 and float(x["lift"]) >= 20]
        display_context = " ".join(x["ngram"] for x in ngrams[cluster]) + " " + " ".join(top_intents)
        if cluster == -1:
            family, submodule, detail = "Noise", "Hành trình hỗn hợp", "Không gán chức năng nghiệp vụ"
        elif code == "unclassified.technical.internal_screen":
            family, submodule, detail = technical_display_name(display_context)
        else:
            family, submodule, detail = taxonomy.get(code, ("Chưa phân loại", "Tính năng động chưa có mapping", "Cluster/journey hỗn hợp chưa đủ độ tin cậy"))
        secondary = [c for c, _ in ranked if c != code][:2]
        ng_code = ngram_ranked[0][0] if ngram_ranked else "unclassified.dynamic.mixed_journey"
        ng_share = (ngram_scores[ng_code] / (sum(ngram_scores.values()) or 1))
        rows.append({
            "cluster": cluster,
            "size": cat["size"],
            "share": cat["share"],
            "mapping_name": f"{family} | {submodule} | {detail}",
            "function_code": code,
            "business_family": family,
            "business_submodule": submodule,
            "business_detail": detail,
            "confidence": confidence,
            "evidence_share": round(confidence_share, 4),
            "secondary_function_codes": "; ".join(secondary),
            "ngram_primary_function": ng_code,
            "ngram_evidence_share": round(ng_share, 4),
            "all_8_ngrams_used": len(ngrams[cluster]),
            "dominant_cluster_intents": " || ".join(top_intents),
            "top_mass_ngrams": " || ".join(x["ngram"] for x in top_ng),
            "rare_token_caution": " || ".join(rare[:3]),
            "medoid_path_support_only": cat.get("medoid_path", ""),
        })
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")
    print(Counter(r["confidence"] for r in rows))
    print(Counter(r["function_code"] for r in rows).most_common(20))


if __name__ == "__main__":
    main()
