"""Semantic enrichment stage.

Every case below is a real (screen, target) pair taken from the July corpus.
The point of the assertions is not that a particular string maps to a
particular label - it is that the *rules* hold: the two platforms agree, the
generic gestures inherit, and nothing is labelled without evidence.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from journey_clustering import semantics as S
from journey_clustering.taxonomy import GENERAL, UNKNOWN


def label(event_type: str, screen: str, target: str = "") -> S.SemanticLabel:
    return S.classify_event(event_type, screen, target)


def section(event_type: str, screen: str, target: str = "") -> tuple[str, str]:
    result = label(event_type, screen, target)
    return result.business_family, result.business_module


class PathNormalisationTest(unittest.TestCase):
    def test_slash_snake_and_camel_reach_the_same_words(self) -> None:
        self.assertEqual(S.normalize_path("manage_modem"), ("manage", "modem"))
        self.assertEqual(S.normalize_path("ManageModemVC"), ("manage", "modem"))
        self.assertEqual(S.normalize_path("manage/modem"), ("manage", "modem"))

    def test_dynamic_identifiers_and_infrastructure_are_dropped(self) -> None:
        self.assertEqual(
            S.normalize_path("android/Home/{id}/do_action/CALL_API_GET_ACTION"),
            ("home", "doaction", "call", "get"),
        )
        self.assertEqual(S.normalize_path("select_type_-1_-1"), ("select", "type"))
        self.assertEqual(S.normalize_path("hi.fpt.vn/web/shop/search"), ("shop", "search"))

    def test_query_string_and_build_suffix_do_not_reach_the_taxonomy(self) -> None:
        with_query = S.normalize_path("hi.fpt.vn/web/shop/product-management-v920?cat_id=2")
        without_query = S.normalize_path("hi.fpt.vn/web/shop/product-management")
        self.assertEqual(with_query, without_query)

    def test_digit_led_units_survive_camel_splitting(self) -> None:
        self.assertEqual(S.normalize_path("Home/action_wifi_5GHz"), ("home", "wifi", "5ghz"))

    def test_missing_sentinels_normalise_to_nothing(self) -> None:
        for missing in ("<missing>", "<none>", "", "   ", None, "nan", "None"):
            self.assertEqual(S.normalize_path(missing), ())

    def test_a_missing_screen_is_unknown_not_a_container(self) -> None:
        # `astype(str)` on a NaN column yields the literal "nan"; that must not
        # look like an empty-but-real container path.
        for missing in ("<missing>", "nan", "None", ""):
            self.assertEqual(label("view", missing).business_family, UNKNOWN)


class InternetDeviceManagementTest(unittest.TestCase):
    """The biggest slice of the corpus, and the one that motivated the stage."""

    def test_android_and_ios_device_management_agree(self) -> None:
        android = "internet_fprotect_screen/management_device/management_device_screen"
        ios = "FsListConnectedDeviceVC"
        self.assertEqual(section("view", android), ("internet", "device"))
        self.assertEqual(section("view", ios), section("view", android))

    def test_device_detail_stays_in_the_same_module(self) -> None:
        detail = (
            "internet_fprotect_screen/management_device/management_device_screen"
            "/devices_details/device_information"
        )
        self.assertEqual(section("view", detail), ("internet", "device"))
        self.assertEqual(section("view", "DeviceDetailVC"), ("internet", "device"))

    def test_blocking_a_device_is_a_commit_operation(self) -> None:
        result = label(
            "action",
            "FsListConnectedDeviceVC",
            "internet_fprotect_screen/management_device/management_device_screen"
            "/devices_details/device_information/locking_internet",
        )
        self.assertEqual(result.operation, "block")
        self.assertEqual(result.operation_stage, "commit")
        self.assertEqual(result.business_object, "device")

    def test_unblocking_is_the_opposite_operation_on_the_same_object(self) -> None:
        blocked = label(
            "action",
            "FsListConnectedDeviceVC",
            "internet_fprotect_screen/management_device/management_device_screen"
            "/devices_details/device_information/locking_internet",
        )
        unblocked = label(
            "action",
            "FsListConnectedDeviceVC",
            "internet_fprotect_screen/management_device/management_device_screen"
            "/devices_details/device_information/un_locking_internet",
        )
        self.assertEqual(unblocked.operation, "unblock")
        self.assertEqual(blocked.business_object, unblocked.business_object)
        self.assertEqual(
            (blocked.business_family, blocked.business_module),
            (unblocked.business_family, unblocked.business_module),
        )

    def test_parental_control_is_its_own_module(self) -> None:
        self.assertEqual(
            section("view", "internet_fprotect_screen/view_profiles_devices/profiles_devices_detail/internet_break_screen"),
            ("internet", "parental_control"),
        )


class WifiAndModemTest(unittest.TestCase):
    def test_modem_control_agrees_across_platforms(self) -> None:
        android = "android/home/home_service_management/internet_service_tab/modem_control"
        self.assertEqual(section("view", android), ("internet", "modem"))
        self.assertEqual(section("view", "ManageModemVC"), ("internet", "modem"))

    def test_wifi_control_agrees_across_platforms(self) -> None:
        android = "android/home/home_service_management/internet_service_tab/wifi_control"
        self.assertEqual(section("view", android), ("internet", "wifi"))
        self.assertEqual(section("view", "ManageWiFiVC"), ("internet", "wifi"))

    def test_modem_power_toggle_and_restart_are_distinct_operations(self) -> None:
        toggle = label("action", "ManageModemVC", "do_action/MODEM_TURN_ON_OFF")
        restart = label("action", "ManageModemVC", "do_action/MODEM_RESET")
        self.assertEqual(toggle.operation, "toggle")
        self.assertEqual(restart.operation, "restart")
        self.assertEqual(restart.operation_stage, "commit")
        self.assertEqual(toggle.business_object, "modem")

    def test_wifi_band_toggles_keep_the_wifi_module_from_the_home_screen(self) -> None:
        for target in (
            "Home/action_wifi_5GHz",
            "android/wifi_bottom_sheet_home/turn_on_wifi_2GHz",
        ):
            self.assertEqual(section("action", "HomeVC", target), ("internet", "wifi"))

    def test_turning_wifi_on_and_off_are_opposite_operations(self) -> None:
        on = label("action", "HomeVC", "android/wifi_bottom_sheet_home/turn_on_wifi_5GHz")
        off = label("action", "HomeVC", "android/wifi_bottom_sheet_home/turn_off_wifi_5GHz")
        self.assertEqual((on.operation, off.operation), ("enable", "disable"))
        self.assertEqual(on.operation_stage, "configure")


class PaymentTest(unittest.TestCase):
    def test_payment_home_agrees_across_platforms(self) -> None:
        self.assertEqual(section("view", "android/home/payment"), ("payment", GENERAL))
        self.assertEqual(section("view", "PaymentHomeVC"), ("payment", GENERAL))

    def test_checkout_is_a_specific_module_and_a_commit(self) -> None:
        result = label(
            "action",
            "android/home/payment/payment_infor",
            "android/home/payment/payment_infor/checkout",
        )
        self.assertEqual((result.business_family, result.business_module), ("payment", "checkout"))
        self.assertEqual(result.operation, "pay")
        self.assertEqual(result.operation_stage, "commit")

    def test_payment_sub_areas_separate(self) -> None:
        self.assertEqual(section("view", "android/Home/Payment/payment_history"), ("payment", "history"))
        self.assertEqual(section("view", "android/home/payment/prepaid"), ("payment", "prepaid"))
        self.assertEqual(section("view", "android/home/payment/my_cards"), ("payment", "method"))
        self.assertEqual(section("view", "HomeAutoPayVC"), ("payment", "autopay"))

    def test_ios_prepaid_screen_matches_the_android_path(self) -> None:
        self.assertEqual(section("view", "PrePaidVC"), ("payment", "prepaid"))


class GenericActionTest(unittest.TestCase):
    def test_back_inherits_the_screen_it_left(self) -> None:
        result = label("action", "ManageModemVC", "btn_back")
        self.assertEqual((result.business_family, result.business_module), ("internet", "modem"))
        self.assertEqual(result.operation, "back")
        self.assertEqual(result.operation_stage, "abort")

    def test_the_same_gesture_inherits_a_different_context(self) -> None:
        self.assertEqual(section("action", "PaymentHomeVC", "btn_back"), ("payment", GENERAL))
        self.assertEqual(section("action", "ManageWiFiVC", "btn_back"), ("internet", "wifi"))

    def test_popup_and_confirm_inherit_rather_than_forming_their_own_family(self) -> None:
        for target in ("popup", "confirm", "/", "/Body/BackButton"):
            self.assertEqual(
                section("action", "ManageModemVC", target),
                ("internet", "modem"),
                msg=f"target {target!r} should inherit the modem screen",
            )

    def test_a_generic_gesture_does_not_borrow_an_object_from_its_own_name(self) -> None:
        # `select` names an operation, not a business object: the object must
        # come from the screen, otherwise every popup would invent a noun.
        result = label("action", "ManageModemVC", "select")
        self.assertEqual(result.business_object, "modem")
        self.assertEqual(result.operation, "select")

    def test_a_specific_target_overrides_the_screen_it_was_pressed_on(self) -> None:
        # Pressed on the home screen, but the target names the payment area.
        self.assertEqual(section("action", "HomeVC", "Home/Nav_payement"), ("payment", GENERAL))


class CrossNamespaceAliasTest(unittest.TestCase):
    def test_product_abbreviations_resolve_to_one_concept(self) -> None:
        for spelling in ("internet_fprotect_screen", "FsHomeVC", "android/home/home_fsafe/profiles"):
            self.assertEqual(
                label("view", spelling).business_family,
                "internet",
                msg=f"{spelling!r} should read as the FSafe product",
            )

    def test_misspelled_production_event_names_are_repaired(self) -> None:
        self.assertEqual(section("action", "HomeVC", "Home/Nav_payement"), ("payment", GENERAL))
        self.assertEqual(label("view", "loylaty_promotion_view").business_family, "loyalty")

    def test_ios_word_splitting_matches_the_android_spelling(self) -> None:
        self.assertEqual(S.normalize_path("ManageWiFiVC"), S.normalize_path("manage_wifi"))
        self.assertEqual(S.normalize_path("EContractActivity"), S.normalize_path("econtract"))
        self.assertEqual(S.normalize_path("ManageAPVC"), S.normalize_path("manage_access_point"))

    def test_acronym_expansion_reaches_the_same_module(self) -> None:
        self.assertEqual(section("view", "ManageAPVC"), ("internet", "modem"))
        self.assertEqual(section("view", "hi.fpt.vn/dkol/product-detail"), ("shop", "catalog"))


class UnknownFallbackTest(unittest.TestCase):
    def test_an_unmapped_target_without_context_stays_unknown(self) -> None:
        result = label("action", "<missing>", "zzz_unmapped_widget")
        self.assertEqual(result.business_family, UNKNOWN)
        self.assertEqual(result.business_module, UNKNOWN)
        self.assertEqual(result.business_object, UNKNOWN)
        self.assertEqual(result.operation, UNKNOWN)
        self.assertEqual(result.semantic_confidence, 0.0)

    def test_an_unmapped_screen_stays_unknown_rather_than_guessing(self) -> None:
        self.assertEqual(label("view", "zzz_unmapped_widget").business_family, UNKNOWN)

    def test_a_generic_gesture_without_context_stays_unknown(self) -> None:
        self.assertEqual(label("action", "<missing>", "btn_back").business_family, UNKNOWN)

    def test_navigation_containers_are_chrome_not_unknown(self) -> None:
        for container in ("MainTabBarController", "SplashActivity", "BaseNavigation", "PopupVC"):
            self.assertEqual(
                label("view", container).business_family,
                "chrome",
                msg=f"{container!r} is UI plumbing, not a mystery",
            )

    def test_confidence_ranks_evidence(self) -> None:
        full = label("action", "ManageModemVC", "do_action/MODEM_TURN_ON_OFF")
        inherited = label("action", "ManageModemVC", "btn_back")
        hub_only = label("view", "HomeVC")
        self.assertGreater(full.semantic_confidence, inherited.semantic_confidence)
        self.assertGreater(inherited.semantic_confidence, hub_only.semantic_confidence)
        self.assertGreater(hub_only.semantic_confidence, 0.0)

    def test_evidence_names_the_phrase_that_decided_each_slot(self) -> None:
        evidence = label("view", "ManageModemVC").semantic_evidence
        self.assertIn("family=internet<-screen:modem", evidence)
        self.assertIn("module=modem<-screen:modem", evidence)


class DeterminismTest(unittest.TestCase):
    def test_classification_is_a_pure_function(self) -> None:
        first = label("action", "ManageModemVC", "do_action/MODEM_RESET")
        second = label("action", "ManageModemVC", "do_action/MODEM_RESET")
        self.assertEqual(first, second)

    def test_labels_are_frozen(self) -> None:
        with self.assertRaises(Exception):
            label("view", "HomeVC").business_family = "payment"  # type: ignore[misc]


class AnnotateFrameTest(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"event_type": "view", "screen_context": "ManageModemVC", "segment_name": "ManageModemVC"},
                {"event_type": "action", "screen_context": "ManageModemVC", "segment_name": "btn_back"},
                {"event_type": "view", "screen_context": "PaymentHomeVC", "segment_name": "PaymentHomeVC"},
                {"event_type": "action", "screen_context": "<missing>", "segment_name": "zzz_widget"},
            ]
        )

    def test_source_columns_are_untouched_and_semantics_are_added(self) -> None:
        source = self.frame()
        out = S.annotate_semantics(source)
        pd.testing.assert_frame_equal(out[source.columns.tolist()], source)
        for column in S.SEMANTIC_COLUMNS:
            self.assertIn(column, out.columns)
        self.assertEqual(out["business_family"].tolist(), ["internet", "internet", "payment", UNKNOWN])

    def test_report_and_review_queue(self) -> None:
        out = S.annotate_semantics(self.frame())
        report = dict(zip(*S.semantic_report(out).to_dict("list").values()))
        self.assertEqual(report["events"], 4)
        self.assertAlmostEqual(report["family_unknown_rate"], 0.25)
        queue = S.unknown_examples(out)
        self.assertEqual(queue["segment_name"].tolist(), ["zzz_widget"])

    def test_a_frame_without_the_required_columns_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            S.annotate_semantics(pd.DataFrame({"event_type": ["view"]}))


if __name__ == "__main__":
    unittest.main()
