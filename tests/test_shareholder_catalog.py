from __future__ import annotations

import unittest

from scripts.build_shareholder_cluster_catalog import classify, translate_cluster_name


class ShareholderCatalogNamingTest(unittest.TestCase):
    def record(self, path: str) -> dict:
        return {
            "cluster": 58,
            "label_kind": "cluster",
            "medoid_path": path,
            "top_entry_token": "view@HOME",
            "top_exit_token": "action@btn_back",
        }

    def ngram(self, rank: int, text: str) -> dict[str, str]:
        return {"rank": str(rank), "ngram": text, "lift": "10", "cluster_mass": "1"}

    def test_specific_name_requires_top_three_ngram_evidence(self):
        family, name, _, _ = classify(
            self.record("view@VietQRPaymentScreen"),
            [
                self.ngram(1, "action@payment start"),
                self.ngram(2, "view@payment info"),
                self.ngram(3, "action@checkout"),
                self.ngram(5, "view@VietQRPaymentScreen"),
            ],
        )
        self.assertEqual(family, "payments")
        self.assertEqual(name, "Bill payment and checkout")

    def test_specific_name_is_allowed_when_top_three_support_it(self):
        family, name, _, _ = classify(
            self.record("view@VietQRPaymentScreen"),
            [self.ngram(1, "view@VietQRPaymentScreen action@confirm")],
        )
        self.assertEqual(family, "payments")
        self.assertEqual(name, "VietQR payment")

    def test_dominant_login_signal_beats_incidental_prepaid_ngram(self):
        family, name, _, _ = classify(
            self.record(
                "view@SplashActivity -> view@home_guest -> "
                "view@android/login -> action@home_guest/guest_login"
            ),
            [
                {**self.ngram(1, "view@home_guest view@android/login"), "cluster_mass": "2.1024"},
                {
                    **self.ngram(2, "action@home_guest/guest_login action@Home/payment/choose_prepaid_package END"),
                    "cluster_mass": "1.8766",
                },
                {**self.ngram(3, "view@home_guest view@android/login END"), "cluster_mass": "4.7036"},
            ],
        )
        self.assertEqual(family, "authentication")
        self.assertEqual(name, "Guest login")

    def test_primary_account_journey_is_not_relabelled_by_payment_ngram(self):
        family, name, _, _ = classify(
            self.record(
                "view@HOME -> view@android/Account -> view@account_infor -> "
                "action@account_infor#android/Home/Account/Tips_Edit_Profile"
            ),
            [
                self.ngram(
                    1,
                    "action@HOME#Home/Account/Payment_Reminder_Schedule "
                    "view@android/Home/Payment/Schedule_payment view@android/Account",
                ),
                self.ngram(2, "action@account_infor#android/Home/Account/Tips_Edit_Profile view@HOME"),
            ],
        )
        self.assertEqual(family, "account")
        self.assertEqual(name, "Profile and account journey")

    def test_single_unique_action_does_not_claim_completed_journey(self):
        family, name, signals, _ = classify(
            self.record(
                "view@SplashActivity -> view@MainAppActivity -> view@HOME -> "
                "view@android/Home -> action@android/Home#go_to_screen/SCAN_QR"
            ),
            [
                self.ngram(1, "view@HOME view@android/Home"),
                self.ngram(2, "view@android/Home action@android/Home#go_to_screen/SCAN_QR"),
            ],
        )
        self.assertEqual(family, "navigation")
        self.assertEqual(name, "App launch and home browsing")
        self.assertTrue(any("single unique action" in signal for signal in signals))

    def test_single_action_is_specific_when_destination_view_confirms_it(self):
        family, name, _, _ = classify(
            self.record(
                "view@HOME -> action@HOME#go_to_screen/SCAN_QR -> view@ScanQRScreen"
            ),
            [self.ngram(1, "action@HOME#go_to_screen/SCAN_QR view@ScanQRScreen")],
        )
        self.assertEqual(family, "utility")
        self.assertEqual(name, "QR scanning")

    def test_vietnamese_fallback_names_follow_reference_style(self):
        self.assertEqual(
            translate_cluster_name("Customer support journey"),
            "Hành trình hỗ trợ khách hàng",
        )
        self.assertEqual(
            translate_cluster_name("Wi-Fi control — 5 GHz on"),
            "Điều khiển Wi-Fi — 5 GHz bật",
        )


if __name__ == "__main__":
    unittest.main()
