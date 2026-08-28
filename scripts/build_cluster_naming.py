import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/scores/pca48_ngrams12_500"
OUTPUT = BASE / "Cluster_naming.csv"
AUDIT = BASE / "cluster_naming_audit.csv"


# Two-level canonical business taxonomy from hifpt_function_tree.md. Patterns are
# intentionally redundant: a label needs repeated evidence across top n-grams and
# the medoid, rather than a single rare token.
RULES = [
    ("payment.checkout", "Thanh toán", "Thực hiện thanh toán", r"checkout|confirm_payment|payment_infor|payment_method|method_selection|save_qr|paymentqr|thanh.?toan|payement|\bpay\b"),
    ("payment.history", "Thanh toán", "Lịch sử thanh toán", r"payment_history|history_payment|history.*invoice|invoice.*history|lich.?su.*thanh"),
    ("payment.transaction", "Thanh toán", "Kết quả giao dịch", r"transaction|payment_success|payment_fail|payment_result|result_payment|retry_payment|gach_no"),
    ("payment.invoice", "Thanh toán", "Tra cứu hợp đồng và hóa đơn", r"invoice|bill|billing|contract_lookup|debt|cuoc|hoa.?don|paymenthome"),
    ("payment.utility", "Thanh toán", "Tiện ích thanh toán", r"auto.?pay|prepaid|wallet|payment_remind|billing_channel|payment_setting|extension"),
    ("payment.service_request", "Thanh toán", "Thanh toán cho yêu cầu dịch vụ", r"relocation_fee|document_payment|econtract.*pay|support.*pay"),
    ("support.request", "Hỗ trợ", "Tạo yêu cầu hỗ trợ", r"support_create|create_request|support_send|select_type_support|select_content_support|confirm_technical_request|request_create|request.*attachment"),
    ("support.center", "Hỗ trợ", "Trung tâm hỗ trợ", r"supportpage|support_home|support_center|list_support_request|request_history|request_detail|faq|chat.*support"),
    ("support.technical", "Hỗ trợ", "Hỗ trợ kỹ thuật", r"internet_issue|wifi.*issue|technical_request|tv_issue|camera_issue|modem.*issue|network.*support"),
    ("support.procedure", "Hỗ trợ", "Hỗ trợ thủ tục", r"ownership_transfer|service_relocation|customer_information|service_lifecycle|procedure"),
    ("support.billing", "Hỗ trợ", "Hỗ trợ cước phí", r"invoice_inquiry|payment_feedback|prepaid_request|support.*billing|support.*invoice"),
    ("support.application", "Hỗ trợ", "Hỗ trợ ứng dụng", r"login_issue|system_incident|application_issue|support.*otp|support.*loyalty"),
    ("support.proactive", "Hỗ trợ", "Hỗ trợ chủ động", r"one_tap_request|lite_login|lite_incident|proactive.*issue"),
    ("contract.acceptance_record", "Hợp đồng", "Biên bản nghiệm thu", r"acceptance|bbnt|detailreport|minutes|bien.?ban.*nghiem"),
    ("contract.document", "Hợp đồng", "Hợp đồng và phụ lục", r"econtract|electronic_record|contract_appendix|quoteandminutes|sign.*contract|contract.*sign"),
    ("contract.ownership", "Hợp đồng", "Chủ sở hữu hợp đồng", r"owner_check|owner_otp|ownership|transfer_owner|chinh.?chu"),
    ("contract.information", "Hợp đồng", "Thông tin hợp đồng", r"choosecontract|choose_contract|change_contract|contract_info|contract_detail|customer_update|phone_update|identity_document"),
    ("authentication.vneid", "Xác thực", "Đăng nhập VNeID", r"vneid|sso.*login|consent"),
    ("authentication.identity", "Xác thực", "Xác minh danh tính", r"cccd|identity|contract_match|eligibility_check|ekyc"),
    ("authentication.document", "Xác thực", "Xác thực giao dịch/biên bản", r"owner_otp|vneid_sign|signing.*otp|document.*auth"),
    ("authentication.session", "Xác thực", "Đăng nhập và phiên", r"login|logout|guest|fpt.?id|session|signin|sign_in"),
    ("service_management.relocation", "Quản lý dịch vụ", "Chuyển địa điểm", r"relocation|move_address|new_address|chuyen.?dia.?diem"),
    ("service_management.lifecycle", "Quản lý dịch vụ", "Thay đổi vòng đời dịch vụ", r"upgrade|restore|suspend|terminate|tam.?ngung|nang.?cap"),
    ("service_management.status", "Quản lý dịch vụ", "Tình trạng dịch vụ", r"service_status|internet_status|issue_detection|recommended_action|mirrorcle|csoc"),
    ("service_management.management", "Quản lý dịch vụ", "Quản lý dịch vụ tổng quát", r"service_management|manage_service|add_service|service_overview"),
    ("device_management.internet", "Quản lý thiết bị", "Quản lý Internet/Wi-Fi", r"manage_modem|modem|wifi_credentials|wifi_infor|btn_update_wifi|action_wifi|wifi_bottom_sheet|changewifiinfor|fslistconnecteddevice|fshomevc|change_wifi|internet_management|fprotect|management_device|network_device"),
    ("device_management.home_device", "Quản lý thiết bị", "Thiết bị trong nhà", r"home_device|device_relocation|wifi_quality|device.*quality"),
    ("device_management.service_device", "Quản lý thiết bị", "Thiết bị dịch vụ", r"camera_support|tv_support|service_device"),
    ("commerce.loyalty", "Thương mại", "Ưu đãi và khách hàng thân thiết", r"promotion|offer|voucher|hot.?deal|fgold|loyalty|redeem|uu.?dai"),
    ("commerce.catalog", "Thương mại", "Khám phá sản phẩm và dịch vụ", r"product_detail|recommendation|catalog|register_service|service_registration|product"),
    ("commerce.subscription", "Thương mại", "Danh mục đăng ký dịch vụ", r"fpt.?play|camera.?ai|ultra.?fast|f.?safe|fsafe|combo|subscription|register.*internet|access.?point"),
    ("commerce.campaign", "Thương mại", "Chiến dịch và nội dung thương mại", r"exclusive_offer|recommend_for_you|campaign|adsview|banner"),
    ("engagement.notification", "Tương tác", "Thông báo", r"notification|details?noti|noti(?:fication)?_detail|view_os_noti|home/notification|push|mark_all_read|bulk_delete|deeplink.*notification"),
    ("engagement.referral", "Tương tác", "Lan tỏa cùng HiFPT", r"invite_friend|referral|become_partner|acquisition_source"),
    ("engagement.home", "Tương tác", "Tương tác tại Home", r"home_pull_to_refresh|home.*interaction|login_prompt|try_feature|exclusive_feature"),
    ("feedback.post_support", "Phản hồi", "Đánh giá sau hỗ trợ", r"\bnps\b|\bcsat\b|post_support|rating.*support"),
    ("feedback.billing", "Phản hồi", "Phản hồi về thanh toán/cước", r"feedback.*payment|feedback.*invoice|billing.*feedback"),
    ("feedback.application", "Phản hồi", "Phản hồi về ứng dụng", r"app_feedback|application.*issue|report_app"),
    ("utility.smart_rental", "Tiện ích", "Nhà trọ thông minh", r"smart_rental|rental|tenant|room_management|receipt_management|nha.?tro"),
    ("utility.experience", "Tiện ích", "Tiện ích trải nghiệm", r"tarot|sao.?may|game-checkin|game|sandbox|about_hifpt"),
    ("navigation.cross_channel", "Điều hướng", "Điều hướng liên ứng dụng/kênh", r"webview|deeplink|open_url|external|qr_handoff"),
    ("navigation.post_login", "Điều hướng", "Điều hướng sau đăng nhập", r"resume_destination|login_gate|resume_after_login"),
    ("navigation.dynamic_content", "Điều hướng", "Điều hướng động", r"do_action|configured_feature|dynamic.*banner"),
    ("navigation.main_tab", "Điều hướng", "Thanh điều hướng chính", r"nav_home|nav_pay|nav_payment|nav_support|nav_promotion|nav_account|main_tab|bottom_nav"),
]

COMPILED = [(code, l1, l2, re.compile(pattern, re.I)) for code, l1, l2, pattern in RULES]


def load_cluster_labels(platform):
    model_dir = BASE / platform / "model_version=latest" / f"platform={platform}" / "model_output"
    catalog = json.loads((model_dir / f"{platform}_cluster_catalog.json").read_text())
    ngrams = defaultdict(list)
    with (model_dir / f"{platform}_cluster_ngrams.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["rank"]) <= 12:
                ngrams[int(row["cluster"])].append(row)

    labels = {}
    for item in catalog:
        cluster = int(item["cluster"])
        if cluster == -1:
            labels[cluster] = {
                "level_1": "Chưa phân loại", "level_2": "Tính năng động chưa có mapping",
                "name": "Chưa phân loại > Journey hỗn hợp/nhiễu", "code": "unclassified.dynamic",
                "confidence": "low", "evidence": "Noise class (-1); no forced business label",
            }
            continue

        rows = ngrams.get(cluster, [])
        max_mass = max([float(x["cluster_mass"]) for x in rows] or [1.0])
        scores = defaultdict(float)
        hits = defaultdict(list)
        for row in rows:
            text = row["ngram"].lower()
            # rank decay plus normalized within-cluster mass prevents a rare, high-lift
            # n-gram from deciding the label on its own.
            weight = (1.0 / math.sqrt(int(row["rank"]))) * (0.35 + 0.65 * float(row["cluster_mass"]) / max_mass)
            for code, l1, l2, pattern in COMPILED:
                if pattern.search(text):
                    scores[(code, l1, l2)] += weight
                    hits[(code, l1, l2)].append(row["ngram"])

        medoid = (item.get("medoid_path") or "").lower()
        for code, l1, l2, pattern in COMPILED:
            found = pattern.findall(medoid)
            if found:
                # Medoid corroborates the cluster evidence but cannot outweigh several
                # common n-grams by itself.
                scores[(code, l1, l2)] += min(0.9, 0.22 * len(found))
                hits[(code, l1, l2)].append("medoid: " + (item.get("medoid_path") or "")[:180])

        if not scores:
            labels[cluster] = {
                "level_1": "Chưa phân loại", "level_2": "Tính năng động chưa có mapping",
                "name": "Chưa phân loại > Chưa đủ bằng chứng", "code": "unclassified.dynamic",
                "confidence": "low", "evidence": "No repeated canonical-function evidence",
            }
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        (code, l1, l2), top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        n_support = len(hits[(code, l1, l2)])
        # Ambiguous close races without repeated support are not forced into a domain.
        if top < 0.8 or (top < second * 1.12 and n_support < 3):
            labels[cluster] = {
                "level_1": "Chưa phân loại", "level_2": "Tính năng động chưa có mapping",
                "name": "Chưa phân loại > Journey hỗn hợp", "code": "unclassified.dynamic",
                "confidence": "low", "evidence": "Competing/weak signals: " + "; ".join(x[0][0] for x in ranked[:3]),
            }
        else:
            ratio = top / max(second, 0.25)
            confidence = "high" if n_support >= 3 and ratio >= 1.5 else "medium"
            labels[cluster] = {
                "level_1": l1, "level_2": l2, "name": f"{l1} > {l2}", "code": code,
                "confidence": confidence, "evidence": " | ".join(hits[(code, l1, l2)][:3]),
            }
    return labels, catalog


def main():
    all_labels = {}
    catalogs = {}
    for platform in ("ios", "android"):
        all_labels[platform], catalogs[platform] = load_cluster_labels(platform)

    # Compact lookup only: one row per platform + cluster. The aliases match the
    # repository's cluster-name mapping convention while retaining the explicit
    # two-level HiFPT taxonomy requested for downstream joins.
    header = [
        "platform", "cluster_id", "cluster", "cluster_name", "cluster_name_vi",
        "business_family", "business_family_vi", "cluster_name_level_1",
        "cluster_name_level_2", "canonical_level_2_code", "naming_confidence",
        "journey_count", "journey_share", "medoid_journey_id", "medoid_length",
        "top_entry_token", "top_exit_token", "ngrams",
    ]
    counts = defaultdict(int)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=header)
        writer.writeheader()
        for platform in ("ios", "android"):
            model_dir = BASE / platform / "model_version=latest" / f"platform={platform}" / "model_output"
            ng_by_cluster = defaultdict(list)
            with (model_dir / f"{platform}_cluster_ngrams.csv").open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row["rank"]) <= 8:
                        ng_by_cluster[int(row["cluster"])].append(row["ngram"])
            for item in sorted(catalogs[platform], key=lambda x: int(x["cluster"])):
                cluster = int(item["cluster"])
                label = all_labels[platform][cluster]
                writer.writerow({
                    "platform": platform,
                    "cluster_id": cluster,
                    "cluster": cluster,
                    "cluster_name": label["name"],
                    "cluster_name_vi": label["name"],
                    "business_family": label["level_1"],
                    "business_family_vi": label["level_1"],
                    "cluster_name_level_1": label["level_1"],
                    "cluster_name_level_2": label["level_2"],
                    "canonical_level_2_code": label["code"],
                    "naming_confidence": label["confidence"],
                    "journey_count": item["size"],
                    "journey_share": item["share"],
                    "medoid_journey_id": item.get("medoid_journey_id", ""),
                    "medoid_length": item.get("medoid_length", ""),
                    "top_entry_token": item.get("top_entry_token", ""),
                    "top_exit_token": item.get("top_exit_token", ""),
                    "ngrams": " || ".join(ng_by_cluster[cluster]),
                })
                counts[platform] += 1

    # Compact audit companion used for QA; not the requested deliverable.
    audit_header = ["platform", "cluster", "cluster_size", "cluster_share", "cluster_name_level_1", "cluster_name_level_2", "canonical_level_2_code", "naming_confidence", "naming_evidence", "medoid_path"]
    with AUDIT.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=audit_header)
        writer.writeheader()
        for platform in ("ios", "android"):
            for item in catalogs[platform]:
                cluster = int(item["cluster"]); label = all_labels[platform][cluster]
                writer.writerow({"platform": platform, "cluster": cluster, "cluster_size": item["size"], "cluster_share": item["share"], "cluster_name_level_1": label["level_1"], "cluster_name_level_2": label["level_2"], "canonical_level_2_code": label["code"], "naming_confidence": label["confidence"], "naming_evidence": label["evidence"], "medoid_path": item.get("medoid_path", "")})

    print(json.dumps({"output": str(OUTPUT), "audit": str(AUDIT), "mapping_rows": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
