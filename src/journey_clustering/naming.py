"""Name clusters with the Hi FPT 3-level taxonomy and audit observed evidence.

Unlike the legacy naming script, this pipeline reads the taxonomy source of
truth directly.  A taxonomy node is eligible for naming only after a literal
screen/event in one of the supplied cluster n-gram files matches a conservative
bilingual evidence rule.  Parent evidence never marks all of its children as
observed.

The taxonomy input may be either the original Markdown table or an exported
CSV with the columns ``taxonomy_id``, ``business_family``,
``business_submodule`` and ``business_detail``.

Example:
  python scripts/taxonomy_cluster_naming_pipeline.py \
    --taxonomy /path/to/hifpt-journey-taxonomy.md \
    --android-ngrams /path/to/android_cluster_ngrams.csv \
    --ios-ngrams /path/to/ios_cluster_ngrams.csv \
    --output-dir output/scores/taxonomy_naming \
    --android-input /path/to/android_scores.csv \
    --android-output /path/to/android_scores_taxonomy_named.csv \
    --ios-input /path/to/ios_scores.csv \
    --ios-output /path/to/ios_scores_taxonomy_named.csv \
    --overwrite

When a score input/output pair is supplied, the pipeline also creates a
shareholder catalog beside the named score CSV. Use the corresponding
``--*-catalog-output`` argument to choose another location.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PLATFORMS = ("android", "ios")
APPLIED_COLUMNS = (
    "taxonomy_id", "cluster_name", "business_family", "business_submodule",
    "business_detail", "naming_confidence", "naming_source", "needs_review",
)
STALE_SEMANTIC_COLUMNS = {
    *APPLIED_COLUMNS,
    "module", "submodule", "detail", "cluster_name_en",
    "business_family_code", "canonical_level_2_code", "mapping_function_code",
    "function_code",
}
TAXONOMY_ROW = re.compile(
    r"^\|\s*(M\d{2}\.\d{2}\.\d{2})\s*\|\s*([^|]+?)\s*\|\s*"
    r"([^|]+?)\s*\|\s*([^|]+?)\s*\|"
)
GENERIC = re.compile(
    r"(?:^|[#/@_])(?:home|popup|btn_back|back|close|view|action|do_action|"
    r"call_api_get_action|missing)(?:$|[#/@_])",
    re.I,
)


def rx(value: str) -> re.Pattern[str]:
    return re.compile(value, re.I)


@dataclass(frozen=True)
class TaxonomyLeaf:
    taxonomy_id: str
    module: str
    submodule: str
    detail: str


@dataclass(frozen=True)
class EvidenceRule:
    required: tuple[re.Pattern[str], ...]
    specificity: int

    def matches(self, text: str) -> bool:
        # Require concepts to co-occur in one event token.  Matching an action
        # on the source and an unrelated entity on the destination would turn
        # a transition into false detail evidence.
        tokens = [token for token in text.split() if token not in {"<bos>", "<eos>"}]
        return any(all(pattern.search(token) for pattern in self.required) for token in tokens)


@dataclass(frozen=True)
class ReviewedRule:
    """Human-reviewed literal tracker evidence for conservative recovery."""

    target: str
    pattern: re.Pattern[str]
    priority: int = 1


# These rules are applied only after the generic evidence scorer declines to
# name a cluster. Each expression is a literal screen/path observed in the
# supplied Android/iOS n-gram files and maps to one existing taxonomy leaf.
REVIEWED_RULES = (
    ReviewedRule("M05.02.06", rx(r"turn_on_wifi_(?:2GHz|5GHz)|wifi_(?:2GHz|5GHz)_turn_on"), 3),
    ReviewedRule("M05.02.07", rx(r"turn_off_wifi_(?:2GHz|5GHz)|wifi_(?:2GHz|5GHz)_turn_off"), 3),
    ReviewedRule("@Internet, Wi-Fi và thiết bị mạng|Wi-Fi chính/khách", rx(r"action_wifi_(?:2GHz|5GHz|guest)|wifi_bottom_sheet_home")),
    ReviewedRule("M05.01.03", rx(r"MODEM_RESET|modem_reset|restart_modem"), 3),
    ReviewedRule("@Internet, Wi-Fi và thiết bị mạng|Modem", rx(r"MODEM_TURN_ON_OFF|modem_turn_on_off"), 2),
    ReviewedRule("M05.05.02", rx(r"devices_tab_connected"), 2),
    ReviewedRule("M05.05.03", rx(r"devices_tab_blocking"), 2),
    ReviewedRule("M05.05.01", rx(r"devices_tab_all"), 2),
    ReviewedRule("M05.05.04", rx(r"devices_searching|click_search_device"), 2),
    ReviewedRule("M05.05.06", rx(r"devices_details|device_information|DeviceDetailVC")),
    ReviewedRule("M05.05.09", rx(r"confirm_block_device|(?<!un_)locking_internet"), 3),
    ReviewedRule("M05.05.10", rx(r"un_locking_internet|unblock_device"), 3),
    ReviewedRule("M15.01.01", rx(r"HomeNotificationVC|android/home/notification(?:$|[#/])")),
    ReviewedRule("M15.01.02", rx(r"view_(?:all|payment|service|promo|app)_cate"), 2),
    ReviewedRule("M15.01.03", rx(r"DetailsNotiVC|NOTIFICATION/DETAIL|NotificationDetailActivity|NotificationButtonDetail"), 2),
    ReviewedRule("M15.01.05", rx(r"mark_read_all"), 3),
    ReviewedRule("M15.01.06", rx(r"notification/delete_noti"), 3),
    ReviewedRule("M03.01.01", rx(r"ChooseContractVC|Contract_List|contract_list")),
    ReviewedRule("M03.01.03", rx(r"choose_contract|change_contract"), 2),
    ReviewedRule("@Yêu cầu hỗ trợ và chăm sóc khách hàng|Tạo yêu cầu", rx(r"SupportCreatorDescriptionOnlyVC|home/support_create")),
    ReviewedRule("M09.06.01", rx(r"ChatBotVC|ChatConversation|support_chat")),
    ReviewedRule("M09.03.04", rx(r"DetailReportVC|ReportDetailScreen|support_status_detail"), 2),
    ReviewedRule("@Sức khỏe và chất lượng mạng|Kiểm tra toàn diện Dr.Smart", rx(r"ErrorCheckingDrSmartScreen|DoctorSmart|dr.?smart")),
    ReviewedRule("M02.01.04", rx(r"HomeGuestVC|home_guest")),
    ReviewedRule("@Hóa đơn và thanh toán|Hóa đơn/khoản thu", rx(r"PaymentHomeVC|view@android/home/payment(?:$|\s)")),
    ReviewedRule("M12.02.01", rx(r"loy(?:a|la)lty_promotion_view|loyalty/home|uu-dai-hot")),
    ReviewedRule("@Hi FPT Shop và đăng ký sản phẩm|Khám phá Shop", rx(r"hi\.fpt\.vn/web/shop/search"), 2),
    ReviewedRule("@Đánh giá, góp ý và khảo sát|CSAT dịch vụ", rx(r"CSAT_RATING|csat_view_rating"), 2),
    ReviewedRule("@Thông báo, lời nhắc và truy cập nhanh|Cài đặt thông báo", rx(r"popup_remind/setting_noti"), 2),
    ReviewedRule("M15.03.10", rx(r"sa_(?:msg|mascot|short_msg|full_msg)_close"), 2),
)


def ascii_text(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    value = "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


# Vietnamese taxonomy concepts -> literal tracker vocabulary.  These are not
# synonyms invented for naming: every right-hand expression is intended to be
# searched in the supplied n-gram artifacts and is surfaced in the audit CSV.
ENTITY_CONCEPTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vneid", rx(r"vneid|cccd|redirecturireceiver|identity")),
    ("dang nhap", rx(r"login|sign[_-]?in|fpt.?id|authentication")),
    ("otp", rx(r"otp")), ("pin", rx(r"(?:^|[_/#])pin(?:$|[_/#])")),
    ("sinh trac", rx(r"biometric|face.?id|touch.?id")),
    ("tai khoan", rx(r"account|profile|personalvc")),
    ("thong tin ca nhan", rx(r"personalvc|profile|customer_information")),
    ("ngon ngu", rx(r"language|locale")),
    ("dieu khoan", rx(r"policy|term|condition")),
    ("trang chu", rx(r"homevc|home_guest|homepage|nav_home")),
    ("banner", rx(r"banner|adsview")), ("tim kiem", rx(r"search")),
    ("fpt zone", rx(r"fpt.?zone|zone")),
    ("hop dong", rx(r"contract|econtract|e_contract|appendix")),
    ("phu luc", rx(r"appendix|contract")),
    ("nghiem thu", rx(r"acceptance|bbnt|confirmacceptance|electronic_record")),
    ("chuyen dia diem", rx(r"change.?location|relocation|move.?address")),
    ("tam ngung", rx(r"suspend|temporary.?stop")),
    ("khoi phuc", rx(r"restore_service|services_restore|ecounterrestoreservice")),
    ("nang cap", rx(r"upgrade_service|service_upgrade")),
    ("modem", rx(r"modem")), ("wi fi", rx(r"wifi|wi-fi|ssid")),
    ("thiet bi", rx(r"device|mac(?:$|[/#_])")),
    ("mo hinh mang", rx(r"networkchart|networkdiagram|network_tab|networktab")),
    ("luu luong", rx(r"traffic|upload|download")),
    ("fsafe", rx(r"fsafe|fprotect")),
    ("ho so", rx(r"profile|parental")),
    ("bao ve", rx(r"protection|fprotect|fsafe|safe_search|content_block")),
    ("website", rx(r"website|web_site|domain")),
    ("health check", rx(r"health.?net|internet.?health|scan_health")),
    ("dr smart", rx(r"doctorsmart|dr.?smart|errorchecking")),
    ("speedtest", rx(r"speed.?test|latency|download.*upload")),
    ("vung phu", rx(r"coverage|signal")),
    ("mirrorcle", rx(r"mirrorcle|proactive.?error")),
    ("camera", rx(r"camera")), ("cloud", rx(r"camera_cloud|cloud_camera|cloud")),
    ("fpt play", rx(r"fpt.?play|television|tv_service|settop|set_top")),
    ("ultra fast", rx(r"ultra[_-]?fast|ultrafast|web/uf")),
    ("fpt wi fi", rx(r"fpt.?wifi|pay.?wifi")),
    ("ho tro", rx(r"support|request|report")),
    ("yeu cau", rx(r"support|request|report")),
    ("lich hen", rx(r"appointment|set_appointment|schedule")),
    ("chat", rx(r"chatbot|chatconversation|support_chat")),
    ("faq", rx(r"faq")), ("diem giao dich", rx(r"contactlocator|store.?locator")),
    ("hoa don", rx(r"bill|payment_infor|payment_home")),
    ("thanh toan", rx(r"payment|checkout|napas|vietqr|transaction")),
    ("tra ho", rx(r"behalf.?payment|payment_on_behalf")),
    ("tra truoc", rx(r"prepaid|register_prepaid")),
    ("tra tu dong", rx(r"autopay|auto.?pay")),
    ("vi", rx(r"wallet|my_cards")), ("the", rx(r"card|wallet")),
    ("phuong thuc thanh toan", rx(r"payment_method|other_methods|choose_method")),
    ("uu dai", rx(r"voucher|promotion|offer|discount")),
    ("shop", rx(r"web/shop|shopproduct|product-management|dkol")),
    ("san pham", rx(r"product|catalog|shop")), ("don hang", rx(r"order")),
    ("dia chi", rx(r"address|location")),
    ("hoi vien", rx(r"loyalty.*membership|membership|current_rank")),
    ("diem", rx(r"loyalty|fgold|point|redeem|exchange")),
    ("gioi thieu", rx(r"refer_friend|invite_friend|friendreferral|about-gtbb")),
    ("quet qr", rx(r"scan_?qr|qrscanner")), ("qr", rx(r"qr")),
    ("sao may", rx(r"tarot|sao.?may")), ("tarot", rx(r"tarot|sao.?may")),
    ("the nhan vien", rx(r"employee.?qr|view_employee")),
    ("thong bao", rx(r"notification|noti")),
    ("smart assistant", rx(r"smart.?assistant|sa_mascot|sa_short_msg")),
    ("widget", rx(r"widget")), ("csat", rx(r"csat|rating")),
    ("nps", rx(r"(?:^|[/#_])nps(?:$|[/#_])")),
    ("gop y", rx(r"feedback|suggestion")),
    ("khao sat", rx(r"survey|form-survey|fli")),
    ("game", rx(r"game|spin-wheel|checkin")),
    ("nha tro", rx(r"nha.?tro|smart.?motel|boarding.?house")),
)

ACTION_CONCEPTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("xem", rx(r"view@|go_to_screen|detail|home|list|information|history")),
    ("mo", rx(r"view@|go_to_screen|open|click|home")),
    ("dang nhap", rx(r"login|sign[_-]?in")), ("dang xuat", rx(r"logout|sign[_-]?out")),
    ("dang ky", rx(r"register|registration|subscribe")),
    ("gui lai", rx(r"resend")), ("gui", rx(r"send|submit|confirm|create")),
    ("tao", rx(r"create|add|register")), ("them", rx(r"add|create|link")),
    ("xoa", rx(r"delete|remove|unlink")), ("huy", rx(r"cancel|delete|remove")),
    ("doi", rx(r"change|edit|update|switch")),
    ("cap nhat", rx(r"update|edit|change")), ("sua", rx(r"edit|update|change")),
    ("bat", rx(r"turn_on|enable|on_off|toggle")),
    ("tat", rx(r"turn_off|disable|on_off|toggle")),
    ("chon", rx(r"choose|select|click")), ("tim", rx(r"search|find")),
    ("loc", rx(r"filter")), ("chia se", rx(r"share")), ("sao chep", rx(r"copy")),
    ("khoi dong", rx(r"restart|reset|turn_on_off")),
    ("thanh toan", rx(r"payment|checkout|pay|napas|vietqr")),
    ("danh gia", rx(r"rating|rate|csat|nps|survey")),
    ("xac nhan", rx(r"confirm|accept|agree|completed")),
    ("ky", rx(r"sign|otp|confirmacceptance")),
    ("chan", rx(r"block|deny")), ("bo chan", rx(r"unblock|allow")),
    ("gan", rx(r"assign|link")), ("bo gan", rx(r"unassign|unlink")),
)


def parse_taxonomy(path: Path) -> list[TaxonomyLeaf]:
    leaves: list[TaxonomyLeaf] = []
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "taxonomy_id", "business_family", "business_submodule",
                "business_detail",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"missing taxonomy columns in {path}: {sorted(missing)}"
                )
            for row in reader:
                leaves.append(TaxonomyLeaf(
                    row["taxonomy_id"].strip(),
                    row["business_family"].strip(),
                    row["business_submodule"].strip(),
                    row["business_detail"].strip(),
                ))
    else:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = TAXONOMY_ROW.match(line)
            if match:
                leaves.append(TaxonomyLeaf(*(part.strip() for part in match.groups())))
    ids = [leaf.taxonomy_id for leaf in leaves]
    if not leaves or len(ids) != len(set(ids)):
        raise ValueError(f"taxonomy table missing or has duplicate IDs: {path}")
    return leaves


def concept_patterns(label: str) -> tuple[re.Pattern[str], ...]:
    plain = ascii_text(label)
    entities = [(len(key), pattern) for key, pattern in ENTITY_CONCEPTS if key in plain]
    actions = [(len(key), pattern) for key, pattern in ACTION_CONCEPTS if key in plain]
    # Longest concepts prevent generic words (e.g. 'the') from weakening a
    # specific phrase such as 'the nhan vien'. Multiple concepts are kept so a
    # generic payment token cannot validate the more specific "trả hộ" leaf.
    selected: list[re.Pattern[str]] = []
    for _, pattern in sorted(entities, key=lambda item: item[0], reverse=True)[:3]:
        if pattern.pattern not in {item.pattern for item in selected}:
            selected.append(pattern)
    for _, pattern in sorted(actions, key=lambda item: item[0], reverse=True)[:2]:
        if pattern.pattern not in {item.pattern for item in selected}:
            selected.append(pattern)
    return tuple(selected)


def leaf_rule(leaf: TaxonomyLeaf) -> EvidenceRule | None:
    # The parent is part of the evidence contract. Some leaf labels mention a
    # generic object ("thiết bị") whose meaning is only specific under its
    # parent (for example Ultra Fast versus network-device management).
    label = f"{leaf.submodule} {leaf.detail}"
    # A leaf is unusable until its parent itself has a tracker concept. This
    # prevents a generic child phrase such as "mở thông báo" from validating an
    # otherwise unknown submodule merely because any notification was seen.
    if not any(key in ascii_text(leaf.submodule) for key, _ in ENTITY_CONCEPTS):
        return None
    patterns = concept_patterns(label)
    if not patterns:
        return None
    return EvidenceRule(patterns, len(patterns))


def submodule_rule(leaf: TaxonomyLeaf) -> EvidenceRule | None:
    if not any(key in ascii_text(leaf.submodule) for key, _ in ENTITY_CONCEPTS):
        return None
    patterns = concept_patterns(leaf.submodule)
    return EvidenceRule(patterns, len(patterns)) if patterns else None


def read_ngrams(path: Path, platform: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"cluster", "rank", "ngram", "lift", "cluster_mass"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"missing columns in {path}: {sorted(required)}")
        for line, row in enumerate(reader, 2):
            try:
                rows.append({
                    "platform": platform, "cluster": int(row["cluster"]),
                    "rank": int(row["rank"]), "ngram": row["ngram"].strip(),
                    "lift": float(row["lift"]), "mass": max(float(row["cluster_mass"]), 0.0),
                })
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid n-gram row at {path}:{line}") from exc
    return rows


def best_matches(
    text: str,
    leaves: list[TaxonomyLeaf],
    leaf_rules: dict[str, EvidenceRule],
    sub_rules: dict[tuple[str, str], EvidenceRule],
) -> tuple[list[TaxonomyLeaf], list[tuple[str, str]]]:
    detail_hits = [leaf for leaf in leaves if leaf.taxonomy_id in leaf_rules and leaf_rules[leaf.taxonomy_id].matches(text)]
    if detail_hits:
        best = max(leaf_rules[leaf.taxonomy_id].specificity for leaf in detail_hits)
        detail_hits = [leaf for leaf in detail_hits if leaf_rules[leaf.taxonomy_id].specificity == best]
    sub_hits = [key for key, rule in sub_rules.items() if rule.matches(text)]
    if sub_hits:
        best = max(sub_rules[key].specificity for key in sub_hits)
        sub_hits = [key for key in sub_hits if sub_rules[key].specificity == best]
    # Several sibling leaves commonly share broad tracker vocabulary (for
    # example "view modem"). That is valid parent evidence, but not enough to
    # select one leaf arbitrarily.
    sibling_groups: dict[tuple[str, str], list[TaxonomyLeaf]] = defaultdict(list)
    for leaf in detail_hits:
        sibling_groups[(leaf.module, leaf.submodule)].append(leaf)
    ambiguous = {key for key, values in sibling_groups.items() if len(values) > 1}
    if ambiguous:
        detail_hits = [leaf for leaf in detail_hits if (leaf.module, leaf.submodule) not in ambiguous]
        sub_hits = list(dict.fromkeys([*sub_hits, *sorted(ambiguous)]))
    return detail_hits, sub_hits


def analyse(
    leaves: list[TaxonomyLeaf], rows: list[dict[str, object]], min_evidence_rows: int,
    min_mass_coverage: float, min_score_share: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    leaf_rules = {leaf.taxonomy_id: rule for leaf in leaves if (rule := leaf_rule(leaf))}
    sub_rules: dict[tuple[str, str], EvidenceRule] = {}
    for leaf in leaves:
        key = (leaf.module, leaf.submodule)
        if key not in sub_rules and (rule := submodule_rule(leaf)):
            sub_rules[key] = rule

    leaf_evidence: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    sub_evidence: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    cluster_scores: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    cluster_sub_scores: dict[tuple[str, int], Counter[tuple[str, str]]] = defaultdict(Counter)
    cluster_rows: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    leaf_by_id = {leaf.taxonomy_id: leaf for leaf in leaves}
    unknown_reviewed_ids = {
        rule.target for rule in REVIEWED_RULES if not rule.target.startswith("@")
    } - set(leaf_by_id)
    if unknown_reviewed_ids:
        raise ValueError(f"reviewed rules reference unknown taxonomy IDs: {sorted(unknown_reviewed_ids)}")
    taxonomy_paths = {(leaf.module, leaf.submodule) for leaf in leaves}
    unknown_reviewed_paths = {
        tuple(rule.target[1:].split("|", 1))
        for rule in REVIEWED_RULES if rule.target.startswith("@")
    } - taxonomy_paths
    if unknown_reviewed_paths:
        raise ValueError(f"reviewed rules reference unknown taxonomy paths: {sorted(unknown_reviewed_paths)}")

    for row in rows:
        platform, cluster, text = str(row["platform"]), int(row["cluster"]), str(row["ngram"])
        cluster_rows[(platform, cluster)].append(row)
        details, subs = best_matches(text, leaves, leaf_rules, sub_rules)
        weight = float(row["mass"]) * max(math.log1p(float(row["lift"])), 0.1)
        parent_hits = set(subs)
        for leaf in details:
            leaf_evidence[(platform, leaf.taxonomy_id)].append(row)
            cluster_scores[(platform, cluster)][leaf.taxonomy_id] += weight / len(details)
            parent_hits.add((leaf.module, leaf.submodule))
        for module, submodule in parent_hits:
            sub_evidence[(platform, module, submodule)].append(row)
            cluster_sub_scores[(platform, cluster)][(module, submodule)] += weight / len(parent_hits)

    validation: list[dict[str, object]] = []
    for leaf in leaves:
        for platform in PLATFORMS:
            evidence = sorted(
                leaf_evidence.get((platform, leaf.taxonomy_id), []),
                key=lambda row: (-float(row["mass"]), int(row["rank"])),
            )
            parent = sub_evidence.get((platform, leaf.module, leaf.submodule), [])
            validation.append({
                "platform": platform, "taxonomy_id": leaf.taxonomy_id,
                "module": leaf.module, "submodule": leaf.submodule, "detail": leaf.detail,
                "module_observed": str(bool(parent or evidence)).lower(),
                "submodule_observed": str(bool(parent or evidence)).lower(),
                "detail_observed": str(len(evidence) >= min_evidence_rows).lower(),
                "detail_evidence_rows": len(evidence),
                "detail_cluster_count": len({int(row["cluster"]) for row in evidence}),
                "evidence_pattern": " AND ".join(
                    rule.pattern for rule in leaf_rules.get(leaf.taxonomy_id, EvidenceRule((), 0)).required
                ),
                "supporting_evidence": " || ".join(str(row["ngram"]) for row in evidence[:3]),
            })

    mappings: list[dict[str, object]] = []
    for (platform, cluster), ngram_rows in sorted(cluster_rows.items()):
        total_mass = sum(float(row["mass"]) for row in ngram_rows) or 1.0
        detail_scores = cluster_scores[(platform, cluster)]
        sub_scores = cluster_sub_scores[(platform, cluster)]
        chosen = None
        evidence: list[dict[str, object]] = []
        module, submodule, detail = "Chưa phân loại", "Không đủ bằng chứng", ""
        source, confidence, score_share = "insufficient_observed_evidence", "low", 0.0
        if cluster == -1:
            module, submodule, detail = "Chưa phân loại", "Journey hỗn hợp/nhiễu", ""
            source, confidence, score_share = "noise_cluster", "low", 0.0
        elif detail_scores:
            taxonomy_id, score = detail_scores.most_common(1)[0]
            candidate = leaf_by_id[taxonomy_id]
            candidate_evidence = [
                row for row in ngram_rows
                if leaf_rules[taxonomy_id].matches(str(row["ngram"]))
            ]
            candidate_share = score / (sum(detail_scores.values()) or 1.0)
            candidate_coverage = sum(float(row["mass"]) for row in candidate_evidence) / total_mass
            if (
                len(candidate_evidence) >= min_evidence_rows
                and candidate_share >= min_score_share
                and candidate_coverage >= min_mass_coverage
            ):
                chosen = candidate
                module, submodule, detail = chosen.module, chosen.submodule, chosen.detail
                score_share, evidence = candidate_share, candidate_evidence
                confidence = "high" if candidate_share >= .70 and candidate_coverage >= .30 else "medium"
                source = "validated_taxonomy_detail"
        if cluster != -1 and chosen is None and sub_scores:
            (module, submodule), score = sub_scores.most_common(1)[0]
            candidate_share = score / (sum(sub_scores.values()) or 1.0)
            parent_ids = {
                leaf.taxonomy_id for leaf in leaves
                if (leaf.module, leaf.submodule) == (module, submodule)
            }
            evidence = [
                row for row in ngram_rows
                if (
                    (module, submodule) in sub_rules
                    and sub_rules[(module, submodule)].matches(str(row["ngram"]))
                ) or any(
                    taxonomy_id in leaf_rules
                    and leaf_rules[taxonomy_id].matches(str(row["ngram"]))
                    for taxonomy_id in parent_ids
                )
            ]
            candidate_coverage = sum(float(row["mass"]) for row in evidence) / total_mass
            if candidate_share >= min_score_share and candidate_coverage >= min_mass_coverage:
                detail, score_share = "", candidate_share
                confidence = "high" if candidate_share >= .75 and candidate_coverage >= .35 else "medium"
                source = "validated_taxonomy_submodule"
            else:
                module, submodule = "Chưa phân loại", "Không đủ bằng chứng"
                evidence, score_share = [], 0.0
        if cluster != -1 and module == "Chưa phân loại":
            reviewed_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in ngram_rows:
                text = str(row["ngram"])
                matches = [rule for rule in REVIEWED_RULES if rule.pattern.search(text)]
                if matches:
                    strongest = max(rule.priority for rule in matches)
                    for rule in matches:
                        if rule.priority == strongest:
                            reviewed_rows[rule.target].append(row)
            reviewed_mass = {
                taxonomy_id: sum(float(row["mass"]) for row in matched)
                for taxonomy_id, matched in reviewed_rows.items()
            }
            if reviewed_mass:
                target, mass = max(reviewed_mass.items(), key=lambda item: item[1])
                matched = reviewed_rows[target]
                coverage = mass / total_mass
                dominance = mass / (sum(reviewed_mass.values()) or 1.0)
                row_coverage = len(matched) / (len(ngram_rows) or 1)
                row_dominance = len(matched) / (sum(map(len, reviewed_rows.values())) or 1)
                repeated_or_dominant = len(matched) >= min_evidence_rows or coverage >= 0.35
                enough_evidence = coverage >= min_mass_coverage or row_coverage >= 0.50
                clear_winner = dominance >= min_score_share or row_dominance >= min_score_share
                if enough_evidence and clear_winner and repeated_or_dominant:
                    if target.startswith("@"):
                        module, submodule = target[1:].split("|", 1)
                        detail, chosen = "", None
                    else:
                        chosen = leaf_by_id[target]
                        module, submodule, detail = chosen.module, chosen.submodule, chosen.detail
                    evidence, score_share = matched, max(dominance, row_dominance)
                    confidence = "high" if (
                        (coverage >= .35 or row_coverage >= .75)
                        and max(dominance, row_dominance) >= .70
                    ) else "medium"
                    source = "reviewed_literal_evidence"
        evidence_mass = sum(float(row["mass"]) for row in evidence)
        evidence_coverage = min(evidence_mass / total_mass, 1.0)
        evidence_row_coverage = len(evidence) / (len(ngram_rows) or 1)
        mappings.append({
            "platform": platform, "cluster_id": cluster,
            "taxonomy_id": chosen.taxonomy_id if chosen else "",
            "cluster_name": " | ".join(part for part in (module, submodule, detail) if part),
            "module": module, "submodule": submodule, "detail": detail,
            "business_family": module, "business_submodule": submodule,
            "business_detail": detail,
            "naming_confidence": confidence, "naming_source": source,
            "needs_review": str(confidence != "high").lower(),
            "score_share": round(score_share, 4),
            "evidence_mass_coverage": round(evidence_coverage, 4),
            "evidence_row_coverage": round(evidence_row_coverage, 4),
            "supporting_evidence": " || ".join(str(row["ngram"]) for row in sorted(evidence, key=lambda row: -float(row["mass"]))[:3]),
            "top_ngrams": " || ".join(str(row["ngram"]) for row in sorted(ngram_rows, key=lambda row: int(row["rank"]))),
        })
    return validation, mappings


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_cluster_id(value: object, context: str) -> int:
    try:
        number = float(str(value).strip())
        result = int(number)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid cluster ID {value!r} at {context}") from exc
    if number != result:
        raise ValueError(f"cluster ID must be an integer at {context}: {value!r}")
    return result


def write_shareholder_catalog(
    path: Path,
    platform: str,
    mapping: dict[int, dict[str, str]],
    counts: Counter[int],
) -> None:
    total = sum(counts.values()) or 1
    clusters = []
    for cluster_id in sorted(mapping):
        row = mapping[cluster_id]
        count = counts.get(cluster_id, 0)
        clusters.append({
            "cluster_id": cluster_id,
            "taxonomy_id": row["taxonomy_id"],
            "mapping_name": row["cluster_name"],
            "business_family": row["business_family"],
            "business_submodule": row["business_submodule"],
            "business_detail": row["business_detail"],
            "confidence": row["naming_confidence"],
            "naming_source": row["naming_source"],
            "needs_review": row["needs_review"],
            "score_share": row["score_share"],
            "evidence_mass_coverage": row["evidence_mass_coverage"],
            "evidence_row_coverage": row["evidence_row_coverage"],
            "supporting_evidence": row["supporting_evidence"],
            "top_mass_ngrams": row["top_ngrams"],
            "size": count,
            "share": round(count / total, 8),
            "row_count_in_input": count,
            "platform": platform,
        })
    payload = {
        "schema_version": "shareholder-cluster-catalog-taxonomy-v1",
        "platforms": [{"platform": platform, "clusters": clusters}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def apply_mapping_to_scores(
    input_path: Path,
    output_path: Path,
    catalog_path: Path,
    platform: str,
    mapping_rows: list[dict[str, object]],
    overwrite: bool,
) -> dict[str, object]:
    """Stream a score CSV, add taxonomy names, and create its catalog."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output score paths must be different")
    for path in (output_path, catalog_path):
        if path.exists() and not overwrite:
            raise ValueError(f"output exists: {path}; pass --overwrite")
    platform_mapping = {
        int(row["cluster_id"]): {key: str(value) for key, value in row.items()}
        for row in mapping_rows if row["platform"] == platform
    }
    if not platform_mapping:
        raise ValueError(f"mapping has no rows for platform {platform!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    counts: Counter[int] = Counter()
    row_count = 0
    try:
        with input_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            input_columns = list(reader.fieldnames or ())
            cluster_column = "cluster" if "cluster" in input_columns else "cluster_id"
            if cluster_column not in input_columns:
                raise ValueError("score input must contain cluster or cluster_id")
            base_columns = [
                column for column in input_columns if column not in STALE_SEMANTIC_COLUMNS
            ]
            insert_at = base_columns.index(cluster_column) + 1
            output_columns = (
                base_columns[:insert_at] + list(APPLIED_COLUMNS) + base_columns[insert_at:]
            )
            with temporary.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(
                    target, fieldnames=output_columns, extrasaction="ignore"
                )
                writer.writeheader()
                for line_number, row in enumerate(reader, start=2):
                    source_platform = str(row.get("platform", "") or row.get("os", "")).strip().lower()
                    if source_platform and source_platform != platform:
                        raise ValueError(
                            f"score platform {source_platform!r} != {platform!r} at line {line_number}"
                        )
                    cluster_id = parse_cluster_id(row.get(cluster_column), f"input line {line_number}")
                    mapped = platform_mapping.get(cluster_id)
                    if mapped is None:
                        raise ValueError(f"cluster {(platform, cluster_id)} absent from mapping")
                    for column in APPLIED_COLUMNS:
                        row[column] = mapped[column]
                    if "platform" in output_columns:
                        row["platform"] = platform
                    writer.writerow(row)
                    counts[cluster_id] += 1
                    row_count += 1
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    write_shareholder_catalog(catalog_path, platform, platform_mapping, counts)
    return {
        "platform": platform,
        "input": str(input_path),
        "output": str(output_path),
        "catalog": str(catalog_path),
        "rows": row_count,
        "clusters_in_input": len(counts),
        "noise_rows": counts.get(-1, 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--android-ngrams", type=Path, required=True)
    parser.add_argument("--ios-ngrams", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-evidence-rows", type=int, default=2)
    parser.add_argument("--min-mass-coverage", type=float, default=0.15)
    parser.add_argument("--min-score-share", type=float, default=0.60)
    parser.add_argument("--android-input", type=Path, help="Android score CSV to name")
    parser.add_argument("--android-output", type=Path, help="named Android score CSV")
    parser.add_argument("--android-catalog-output", type=Path, help="Android shareholder catalog JSON")
    parser.add_argument("--ios-input", type=Path, help="iOS score CSV to name")
    parser.add_argument("--ios-output", type=Path, help="named iOS score CSV")
    parser.add_argument("--ios-catalog-output", type=Path, help="iOS shareholder catalog JSON")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_evidence_rows < 1:
        raise SystemExit("--min-evidence-rows must be >= 1")
    if not 0 < args.min_mass_coverage <= 1 or not 0 < args.min_score_share <= 1:
        raise SystemExit("coverage/share thresholds must be in (0, 1]")
    score_args = {
        "android": (args.android_input, args.android_output, args.android_catalog_output),
        "ios": (args.ios_input, args.ios_output, args.ios_catalog_output),
    }
    for platform, (input_path, output_path, catalog_path) in score_args.items():
        if bool(input_path) != bool(output_path):
            raise SystemExit(
                f"--{platform}-input and --{platform}-output must be supplied together"
            )
        if catalog_path and not input_path:
            raise SystemExit(
                f"--{platform}-catalog-output requires --{platform}-input/output"
            )
    leaves = parse_taxonomy(args.taxonomy.resolve())
    rows = read_ngrams(args.android_ngrams.resolve(), "android")
    rows += read_ngrams(args.ios_ngrams.resolve(), "ios")
    validation, mappings = analyse(
        leaves, rows, args.min_evidence_rows,
        args.min_mass_coverage, args.min_score_share,
    )
    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "taxonomy_validation.csv", validation)
    write_csv(output_dir / "cluster_mapping.csv", mappings)
    reviewed = [row for row in mappings if row["naming_source"] == "reviewed_literal_evidence"]
    unresolved = [
        row for row in mappings
        if row["naming_source"] in {"insufficient_observed_evidence", "noise_cluster"}
    ]
    if reviewed:
        write_csv(output_dir / "reviewed_cluster_mapping.csv", reviewed)
    if unresolved:
        write_csv(output_dir / "unresolved_clusters.csv", unresolved)
    applied = []
    for platform, (input_arg, output_arg, catalog_arg) in score_args.items():
        if input_arg is None:
            continue
        input_path = input_arg.resolve()
        output_path = output_arg.resolve()
        catalog_path = (
            catalog_arg.resolve()
            if catalog_arg else output_path.with_suffix(".shareholder_catalog.json")
        )
        applied.append(apply_mapping_to_scores(
            input_path, output_path, catalog_path, platform, mappings, args.overwrite
        ))
    summary = {
        "taxonomy": {"modules": len({leaf.module for leaf in leaves}),
                     "submodules": len({(leaf.module, leaf.submodule) for leaf in leaves}),
                     "details": len(leaves)},
        "observed": {
            platform: {
                "modules": len({row["module"] for row in validation if row["platform"] == platform and row["module_observed"] == "true"}),
                "submodules": len({(row["module"], row["submodule"]) for row in validation if row["platform"] == platform and row["submodule_observed"] == "true"}),
                "details": sum(row["platform"] == platform and row["detail_observed"] == "true" for row in validation),
            } for platform in PLATFORMS
        },
        "clusters": dict(Counter(str(row["platform"]) for row in mappings)),
        "unclassified_clusters": sum(row["module"] == "Chưa phân loại" for row in mappings),
        "reviewed_literal_clusters": len(reviewed),
        "thresholds": {
            "min_evidence_rows": args.min_evidence_rows,
            "min_mass_coverage": args.min_mass_coverage,
            "reviewed_min_row_coverage": 0.50,
            "min_score_share": args.min_score_share,
        },
        "output_dir": str(output_dir),
        "applied": applied,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
