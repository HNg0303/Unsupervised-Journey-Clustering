from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "output/scores/pca48_ngrams12_500/cluster_naming_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("cluster_naming_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


def ngram(rank: int, mass: float, text: str) -> dict[str, str]:
    return {
        "rank": str(rank), "cluster_mass": str(mass),
        "lift": "10", "ngram": text,
    }


class ReasonedClusterNamingPipelineTest(unittest.TestCase):
    def test_notification_destination_and_medoid_override_rare_transaction_callback(self):
        rows = [
            ngram(3, 75.1421, "view@HomeVC view@DetailsNotiVC"),
            ngram(4, 3.0434, "action@HomeVC#Home/Home_pull_to_refresh view@DetailsNotiVC"),
            ngram(5, 2.1419, "action@DetailsNotiVC#adsview view@HomeVC"),
            ngram(2, 1.8403, "action@HomeVC#transaction_result/home view@DetailsNotiVC"),
            ngram(1, 1.7660, "action@HomeVC#btn_update_wifi_infor view@DetailsNotiVC"),
        ]
        result = pipeline.score_cluster(
            rows,
            "view@HomeVC -> view@DetailsNotiVC -> "
            "action@DetailsNotiVC#view_os_noti -> view@HomeVC",
        )
        self.assertEqual(result["family"], "Home")
        self.assertEqual(result["submodule"], "Noti + search + Quét QR")
        self.assertEqual(result["detail"], "Xem chi tiết thông báo")

    def test_login_screen_is_treated_as_gate_to_payment_intent(self):
        rows = [
            ngram(1, 180, "action@PaymentHomeVC#Home/Nav_payement view@PopupInviteLoginVC"),
            ngram(2, 90, "view@PaymentHomeVC view@LoginVC"),
            ngram(3, 45, "action@PopupInviteLoginVC#home_guest/bill_icon/bill_home/payment_history view@PaymentHomeVC"),
        ]
        result = pipeline.score_cluster(
            rows,
            "view@HomeGuestVC -> view@PaymentHomeVC -> view@PopupInviteLoginVC -> view@PaymentHomeVC",
        )
        self.assertEqual(result["family"], "Thanh toán")
        self.assertEqual(result["submodule"], "Thanh toán hoá đơn/khoản thu")

    def test_doctor_smart_support_flow_is_not_named_service_management(self):
        rows = [
            ngram(5, 3.3053, "action@DoctorSmartVC#Home/Support/check_error/scan_result/set_appointment/agree_appointment view@UIPageViewController"),
            ngram(1, 1.4452, "action@DoctorSmartVC#scan_result/reject_reboot action@DoctorSmartVC#Home/Support/check_error/scan_result/need_staff_result"),
            ngram(2, 0.7989, "action@_UIRemoteInputViewController#Home/Support/Create_Request_Screen/Confirm_Technical_Request view@DoctorSmartVC"),
            ngram(7, 0.5140, "action@PopupRequestChossenVC#select_type_support action@PopupRequestChossenVC#select_content_support"),
            ngram(4, 0.3335, "action@PopupRequestChossenVC#home/support_create/support_send action@PopupRequestChossenVC#add_note"),
        ]
        result = pipeline.score_cluster(
            rows,
            "view@SupportPageViewController -> view@SupportCreatorVC -> "
            "action@SupportCreatorVC#Home/Nav_support -> view@PopupRequestChossenVC -> "
            "action@PopupRequestChossenVC#select_type_support -> "
            "action@PopupRequestChossenVC#select_content_support -> "
            "action@PopupRequestChossenVC#home/support_create/support_send",
        )
        self.assertEqual(result["family"], "Trung tâm hỗ trợ")
        self.assertEqual(result["submodule"], "Tạo yêu cầu hỗ trợ")

    def test_apply_csv_removes_stale_function_codes_and_maps_by_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            output = root / "named.csv"
            catalog = root / "shareholder.json"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["journey_id", "platform", "cluster", "function_code", "cluster_name"],
                )
                writer.writeheader()
                writer.writerow({
                    "journey_id": "j1", "platform": "ios", "cluster": "7",
                    "function_code": "stale.code", "cluster_name": "stale name",
                })
            mapping = [{
                "platform": "ios", "cluster_id": 7,
                "cluster_name": "Home | Trang chủ", "business_family": "Home",
                "business_submodule": "Trang chủ", "business_detail": "",
                "naming_confidence": "medium", "mass_coverage": 0.7,
                "family_score_share": 1.0, "submodule_score_share": 1.0,
                "medoid_agreement": True, "reasoning": "test",
                "supporting_evidence": "view@HomeVC", "alternative_submodules": "",
                "top_ngrams": "view@HomeVC", "medoid_path": "view@HomeVC",
                "journey_count": 1, "journey_share": 1.0,
            }]
            pipeline.apply_mapping(source, output, catalog, "ios", mapping, False)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
                self.assertEqual(row["cluster_name"], "Home | Trang chủ")
                self.assertNotIn("function_code", row)


if __name__ == "__main__":
    unittest.main()
