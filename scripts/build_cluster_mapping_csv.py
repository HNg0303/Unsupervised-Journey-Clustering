#!/usr/bin/env python3
"""Build cluster_mapping.csv directly from n-grams, catalogs, and sitemap.

N-gram rank and mass jointly determine the primary function. The medoid may
corroborate but cannot dominate cluster-wide evidence. Weak/conflicting
clusters receive a general sitemap-style name and an explicit review flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_cluster_naming import COMPILED  # noqa: E402

DEFAULT_BASE = ROOT / "output/scores/pca48_ngrams12_500"
OUTPUT_COLUMNS = (
    "platform", "cluster_id", "cluster_name", "business_family",
    "business_submodule", "business_detail", "canonical_level_2_code",
    "naming_confidence", "journey_count", "journey_share", "naming_source",
    "needs_review", "naming_reason", "evidence_share", "ngram_primary_function",
    "ngram_evidence_share", "secondary_function_codes",
    "ngram_medoid_agreement", "top_ngrams", "medoid_path",
)

CODE_TO_SITEMAP_PAIR = {
    "payment.checkout": ("Thanh toán", "Thanh toán hoá đơn/khoản thu"),
    "payment.history": ("Thanh toán", "Lịch sử thanh toán"),
    "payment.transaction": ("Thanh toán", "Thanh toán hoá đơn/khoản thu"),
    "payment.invoice": ("Thanh toán", "Thanh toán hoá đơn/khoản thu"),
    "payment.utility": ("Thanh toán", "Tiện ích"),
    "payment.service_request": ("Thanh toán", "Thanh toán hoá đơn/khoản thu"),
    "support.request": ("Trung tâm hỗ trợ", "Tạo yêu cầu hỗ trợ"),
    "support.center": ("Trung tâm hỗ trợ", "Tổng quan hỗ trợ"),
    "support.technical": ("Trung tâm hỗ trợ", "Tạo yêu cầu hỗ trợ"),
    "support.procedure": ("Trung tâm hỗ trợ", "Tạo yêu cầu hỗ trợ"),
    "support.billing": ("Trung tâm hỗ trợ", "Tạo yêu cầu hỗ trợ"),
    "support.application": ("Trung tâm hỗ trợ", "Tạo yêu cầu hỗ trợ"),
    "support.proactive": ("Trung tâm hỗ trợ", "Tạo yêu cầu hỗ trợ"),
    "contract.acceptance_record": ("Tài khoản", "Hợp đồng và dịch vụ"),
    "contract.document": ("Tài khoản", "Hợp đồng và dịch vụ"),
    "contract.ownership": ("Tài khoản", "Hợp đồng và dịch vụ"),
    "contract.information": ("Tài khoản", "Hợp đồng và dịch vụ"),
    "authentication.vneid": ("Tài khoản", "Cài đặt => Tài khoản và bảo mật"),
    "authentication.identity": ("Tài khoản", "Cài đặt => Tài khoản và bảo mật"),
    "authentication.document": ("Tài khoản", "Hợp đồng và dịch vụ"),
    "authentication.session": ("Tài khoản", "Cài đặt => Tài khoản và bảo mật"),
    "service_management.relocation": ("Tài khoản", "Hợp đồng và dịch vụ"),
    "service_management.lifecycle": ("Quản lý dịch vụ", "Internet (Hợp đồng)"),
    "service_management.status": ("Quản lý dịch vụ", "Internet (Hợp đồng)"),
    "service_management.management": ("Quản lý dịch vụ", "Tổng quan dịch vụ"),
    "device_management.internet": ("Quản lý dịch vụ", "Internet (Hợp đồng)"),
    "device_management.home_device": ("Quản lý dịch vụ", "Internet (Hợp đồng)"),
    "device_management.service_device": ("Quản lý dịch vụ", "Camera"),
    "commerce.loyalty": ("Loyalty", "Ưu đãi của tôi"),
    "commerce.catalog": ("Đăng ký dịch vụ", "Menu dịch vụ"),
    "commerce.subscription": ("Đăng ký dịch vụ", "Menu dịch vụ"),
    "commerce.campaign": ("Home", "Banner"),
    "engagement.notification": ("Home", "Noti + search + Quét QR"),
    "engagement.referral": ("Tài khoản", "Lan toả cùng Hi FPT"),
    "engagement.home": ("Home", "Trang chủ"),
    "feedback.post_support": ("Trung tâm hỗ trợ", "Tạo yêu cầu hỗ trợ"),
    "feedback.billing": ("Thanh toán", "Thanh toán hoá đơn/khoản thu"),
    "feedback.application": ("Tài khoản", "Đánh giá của tôi"),
    "utility.smart_rental": ("Home", "Nhà trọ thông minh"),
    "utility.experience": ("Home", "Chỉ có tại Hi FPT"),
    "navigation.cross_channel": ("Home", "Taskbar"),
    "navigation.post_login": ("Home", "Taskbar"),
    "navigation.dynamic_content": ("Home", "Banner"),
    "navigation.main_tab": ("Home", "Taskbar"),
    "unclassified.dynamic": ("Home", "Hành trình tổng quát"),
}

# Level-3 labels are intentionally sparse. They are emitted only when a high-
# confidence level-2 decision also has repeated, action-specific evidence.
DETAIL_RULES = {
    "engagement.notification": (
        (re.compile(r"details?noti|notificationdetail|noti(?:fication)?_detail|view_os_noti", re.I), "Xem chi tiết thông báo"),
        (re.compile(r"notification.*(?:setting|preference)|setting.*notification", re.I), "Bật/tắt nhận thông báo theo danh mục"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--sitemap", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: <base>/cluster_mapping.csv")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalise_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _clean_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sitemap_hierarchy(path: Path) -> dict[str, tuple[str, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 4:
        raise ValueError("sitemap must contain a family header row and child rows")
    hierarchy: dict[str, tuple[str, dict[str, str]]] = {}
    for column, raw_family in enumerate(rows[2]):
        if not raw_family.strip():
            continue
        family = _clean_label(raw_family)
        children: dict[str, str] = {}
        for row in rows[3:]:
            if column < len(row) and row[column].strip():
                child = _clean_label(row[column])
                children[_normalise_label(child)] = child
        hierarchy[_normalise_label(family)] = (family, children)
    return hierarchy


def _model_dir(base: Path, platform: str) -> Path:
    return base / platform / "model_version=latest" / f"platform={platform}" / "model_output"


def _detail_rows(base: Path, platform: str) -> dict[int, dict[str, str]]:
    path = _model_dir(base, platform) / f"{platform}_cluster_business_mapping.csv"
    if not path.exists():
        return {}
    return {int(row["cluster"]): row for row in _read_csv(path)}


def _match_penalty(code: str, text: str) -> float:
    """Prevent generic wrapper actions from deciding the business name alone."""
    if code == "payment.transaction":
        # These callbacks are emitted from Home/Loyalty after many unrelated
        # journeys. They are not payment evidence unless the same 2-gram also
        # contains an explicit payment/result screen.
        if re.search(r"transaction_result/(?:home|loyalty)", text, re.I):
            return 0.05
        if not re.search(
            r"payment|payement|invoice|bill|checkout|napas|vietqr|gach_no|"
            r"result_payment|retry_payment|successful_payment|fail_payment",
            text,
            re.I,
        ):
            return 0.20
    if code == "commerce.campaign" and not re.search(
        r"banner|campaign|exclusive|recommend|hot.?deal", text, re.I
    ):
        return 0.02
    if code == "navigation.dynamic_content" and not re.search(
        r"configured_feature|dynamic.*banner", text, re.I
    ):
        return 0.02
    if code == "engagement.home" and not re.search(
        r"try_feature|exclusive_feature|home.*interaction", text, re.I
    ):
        return 0.10
    return 1.0


def _score_evidence(ngrams: list[dict[str, str]], medoid: str) -> dict[str, object]:
    """Score business evidence with mass as the dominant signal.

    ``cluster_mass`` already captures how representative a 2-gram is for the
    cluster. The previous formula added a 0.25 floor to every matched n-gram,
    allowing a rare token to beat a dominant path. Here mass is never floored;
    rank only discounts otherwise comparable rows. Medoid evidence is a small
    corroborating bonus and cannot replace cluster-wide n-gram evidence.
    """
    ngram_scores: Counter[str] = Counter()
    ngram_hits: Counter[str] = Counter()
    total_weight = 0.0
    matched_weight = 0.0
    for row in ngrams:
        # Normalize known tracking typos before matching business vocabulary.
        text = row["ngram"].replace("loylaty", "loyalty")
        rank = int(row["rank"])
        mass = max(float(row["cluster_mass"]), 0.0)
        weight = mass / math.sqrt(max(rank, 1))
        total_weight += weight
        row_matched = False
        for code, _, _, pattern in COMPILED:
            if pattern.search(text):
                ngram_scores[code] += weight * _match_penalty(code, text)
                ngram_hits[code] += 1
                row_matched = True
        if row_matched:
            matched_weight += weight

    medoid_codes = {
        code for code, _, _, pattern in COMPILED if pattern.search(medoid)
    }
    combined = Counter(ngram_scores)
    medoid_bonus = total_weight * 0.08
    for code in medoid_codes:
        combined[code] += medoid_bonus
    if not combined:
        return {
            "code": "unclassified.dynamic", "confidence": "low", "share": 0.0,
            "coverage": 0.0, "ngram_primary": "", "secondary": [], "agreement": False,
            "reason": "Không có token nghiệp vụ đủ mạnh trong n-gram hoặc medoid",
        }

    ranked = combined.most_common()
    code, top_score = ranked[0]
    ngram_primary = ngram_scores.most_common(1)[0][0] if ngram_scores else ""
    total_ngram_score = sum(ngram_scores.values())
    share = ngram_scores[code] / total_ngram_score if total_ngram_score else 0.0
    coverage = matched_weight / total_weight if total_weight else 0.0
    hits = ngram_hits[code]
    agreement = code in medoid_codes
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score / max(second_score, 0.25)
    if share >= 0.65 and coverage >= 0.55 and hits >= 2 and agreement:
        confidence = "high"
        reason = "Mass n-gram chiếm ưu thế, coverage cao và thống nhất với medoid"
    elif share >= 0.55 and coverage >= 0.30 and (hits >= 2 or agreement) and margin >= 1.20:
        confidence = "medium"
        reason = "Mass n-gram đủ trội nhưng coverage hoặc medoid chỉ ở mức trung bình"
    else:
        confidence = "low"
        reason = "Mass evidence yếu/xung đột; dùng tên xấp xỉ theo sitemap và cần review"
    return {
        "code": code, "confidence": confidence, "share": share, "coverage": coverage,
        "ngram_primary": ngram_primary,
        "secondary": [candidate for candidate, _ in ranked[1:4]],
        "agreement": agreement, "reason": reason,
    }


def _refine_pair(code: str, evidence_text: str) -> tuple[str, str]:
    low = evidence_text.casefold()
    if code == "support.center":
        if re.search(r"chat", low):
            return "Trung tâm hỗ trợ", "Chat CSKH"
        if re.search(r"request_history|request_detail|support.*history", low):
            return "Trung tâm hỗ trợ", "Lịch sử hỗ trợ"
        if re.search(r"faq|frequently", low):
            return "Trung tâm hỗ trợ", "Câu hỏi thường gặp"
    if code == "commerce.loyalty":
        if re.search(r"member|fgold|loylaty_member|loyalty_member", low):
            return "Loyalty", "Thông tin KHTT"
        if re.search(r"redeem|exchange|doi.?qua|tich.?diem", low):
            return "Loyalty", "Tích điểm/Đôi quà"
        if re.search(r"hot.?deal|exclusive", low):
            return "Loyalty", "Hot Deal/ Độc quyền => Đặc quyền ưu tiên"
    if code == "utility.experience" and re.search(r"tarot|sao.?may", low):
        return "Home", "Tarot"
    if code == "utility.experience" and re.search(r"smart.?rental|nha.?tro", low):
        return "Home", "Nhà trọ thông minh"
    return CODE_TO_SITEMAP_PAIR.get(code, CODE_TO_SITEMAP_PAIR["unclassified.dynamic"])


def _detail_from_evidence(
    code: str,
    confidence: str,
    ngrams: list[dict[str, str]],
    medoid: str,
) -> str:
    """Return a level-3 label only with repeated n-gram/medoid support."""
    if confidence != "high":
        return ""
    for pattern, label in DETAIL_RULES.get(code, ()):
        hits = sum(bool(pattern.search(row["ngram"])) for row in ngrams)
        medoid_support = bool(pattern.search(medoid))
        if hits >= 2 or (hits >= 1 and medoid_support):
            return label
    return ""


def _canonical_pair(
    pair: tuple[str, str],
    hierarchy: dict[str, tuple[str, dict[str, str]]],
) -> tuple[str, str, str]:
    family, submodule = pair
    family_entry = hierarchy.get(_normalise_label(family))
    if family_entry:
        canonical_family, children = family_entry
        canonical_submodule = children.get(_normalise_label(submodule))
        if canonical_submodule:
            return canonical_family, canonical_submodule, "sitemap_exact"
        return canonical_family, _clean_label(submodule), "sitemap_style_approximation"
    return _clean_label(family), _clean_label(submodule), "sitemap_style_approximation"


def build_mapping(base: Path, sitemap_path: Path) -> list[dict[str, object]]:
    hierarchy = _sitemap_hierarchy(sitemap_path)
    output: list[dict[str, object]] = []
    for platform in ("android", "ios"):
        model_dir = _model_dir(base, platform)
        catalog = json.loads(
            (model_dir / f"{platform}_cluster_catalog.json").read_text(encoding="utf-8")
        )
        details = _detail_rows(base, platform)
        ngrams: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in _read_csv(model_dir / f"{platform}_cluster_ngrams.csv"):
            ngrams[int(row["cluster"])].append(row)

        for catalog_row in sorted(catalog, key=lambda row: int(row["cluster"])):
            cluster = int(catalog_row["cluster"])
            cluster_ngrams = ngrams.get(cluster, [])
            medoid = str(catalog_row.get("medoid_path", "") or "")
            if len(cluster_ngrams) != 8:
                raise ValueError(f"expected 8 n-grams for {(platform, cluster)}")

            if cluster == -1:
                family, submodule, detail = (
                    "Chưa phân loại", "Journey hỗn hợp/nhiễu", ""
                )
                code, confidence, source = "unclassified.dynamic", "low", "noise_cluster"
                needs_review, primary, evidence_share = False, "", 0.0
                mass_coverage = 0.0
                secondary: list[str] = []
                agreement = False
                reason = "Cluster noise -1 không gán một nghiệp vụ duy nhất"
            else:
                scoring = _score_evidence(cluster_ngrams, medoid)
                code = str(scoring["code"])
                confidence = str(scoring["confidence"])
                evidence_share = float(scoring["share"])
                mass_coverage = float(scoring["coverage"])
                secondary = list(scoring["secondary"])
                agreement = bool(scoring["agreement"])
                reason = str(scoring["reason"])
                primary = str(scoring["ngram_primary"])
                evidence_text = " ".join(row["ngram"] for row in cluster_ngrams) + " " + medoid
                family, submodule, source = _canonical_pair(
                    _refine_pair(code, evidence_text), hierarchy
                )
                needs_review = confidence == "low" or code == "unclassified.dynamic"
                detailed = details.get(cluster, {})
                candidate_detail = detailed.get("business_detail", "").strip()
                detail_is_supported = (
                    confidence == "high" and bool(candidate_detail)
                    and detailed.get("confidence", "").strip().lower() == "high"
                    and detailed.get("function_code", "").strip().startswith(code + ".")
                )
                detail = candidate_detail if detail_is_supported else ""
                if not detail:
                    detail = _detail_from_evidence(
                        code, confidence, cluster_ngrams, medoid
                    )

            top_ngrams = " || ".join(
                f"mass={float(row['cluster_mass']):.4f}; rank={int(row['rank'])}; "
                f"lift={float(row['lift']):.4f}; {row['ngram']}"
                for row in sorted(
                    cluster_ngrams,
                    key=lambda row: (-float(row["cluster_mass"]), int(row["rank"])),
                )
            )
            name_parts = [family, submodule] + ([detail] if detail else [])
            output.append({
                "platform": platform,
                "cluster_id": cluster,
                "cluster_name": " | ".join(name_parts),
                "business_family": family,
                "business_submodule": submodule,
                "business_detail": detail,
                "canonical_level_2_code": code,
                "naming_confidence": confidence,
                "journey_count": int(catalog_row["size"]),
                "journey_share": float(catalog_row["share"]),
                "naming_source": source,
                "needs_review": str(needs_review).lower(),
                "naming_reason": reason,
                "evidence_share": round(mass_coverage, 4),
                "ngram_primary_function": primary,
                "ngram_evidence_share": round(evidence_share, 4),
                "secondary_function_codes": "; ".join(secondary),
                "ngram_medoid_agreement": str(agreement).lower(),
                "top_ngrams": top_ngrams,
                "medoid_path": medoid,
            })
    return output


def main() -> int:
    args = parse_args()
    base = args.base if args.base.is_absolute() else ROOT / args.base
    output = args.output or base / "cluster_mapping.csv"
    output = output if output.is_absolute() else ROOT / output
    sitemap = args.sitemap if args.sitemap.is_absolute() else ROOT / args.sitemap
    rows = build_mapping(base, sitemap)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "output": str(output),
        "rows": len(rows),
        "platform_rows": {
            platform: sum(row["platform"] == platform for row in rows)
            for platform in ("android", "ios")
        },
        "confidence": Counter(row["naming_confidence"] for row in rows),
        "needs_review": sum(row["needs_review"] == "true" for row in rows),
        "with_business_detail": sum(bool(row["business_detail"]) for row in rows),
    }, ensure_ascii=False, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
