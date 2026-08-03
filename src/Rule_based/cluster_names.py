"""Human names for every cluster, per platform and per route.

Cluster ids are arbitrary integers assigned by HDBSCAN - they carry no meaning
and they change if you refit. These names are the layer that makes the output
sayable out loud: "iOS e-contract list thrash" instead of "iOS route A cluster
40". Everything downstream (catalogs, scored output, dashboards) can join
against this.

Each entry is (family, name, note):

    family  coarse behavioural group - the axis you actually report on
    name    the short label to use in conversation
    note    what the evidence was, so a name can be challenged

Names were assigned from three pieces of evidence per cluster: the medoid path,
the highest-lift tokens within the cluster, and the behavioural stats (length,
back rate, revisit ratio, loop share).

IMPORTANT: these are tied to the run in `outputs/clusters/`. Refit the model and
the ids move. Regenerate with `scripts/apply_cluster_names.py --audit` to see
which ids no longer match their described behaviour.
"""

from __future__ import annotations

import pandas as pd

NOISE = ("noise", "Unassigned", "HDBSCAN found no dense neighbourhood - review these first")

# --------------------------------------------------------------------------
# ANDROID - route A (tf-idf 1..4-grams), 29 clusters
# --------------------------------------------------------------------------
ANDROID_A: dict[int, tuple[str, str, str]] = {
    -1: NOISE,
    25: ("payment", "Guest bill payment", "full guest flow: billing tab > search contract > select > pay"),
    8: ("contract", "Contract switching", "change_contract > contract list > choose > back. Fast and mechanical"),
    19: ("service", "Internet package upgrade (web)", "dkol/update-package webview. revisit 0.41, loops 26% - friction"),
    26: ("payment", "Postpaid bill payment", "payment tab > select bill > click pay > payment info"),
    18: ("payment", "Prepaid / utilities browse", "enters prepaid or utilities, backs out. 7s median"),
    20: ("econtract", "E-contract signing with VNeID", "unconfirmed e-contract > VNeID > confirm CCCD. 24 steps"),
    1: ("auth", "OAuth login (external)", "login > AuthorizationManagement > RedirectUriReceiver. Zero back, zero revisit"),
    24: ("support", "Support request opened then closed", "Nav_support > support_create > confirm_close. back 0.18"),
    11: ("account", "Account > e-contract browse", "account home > acceptance record / e-contract list"),
    15: ("contract", "Contract sharing and permissions", "e-counter share management: add phone, linked phone, permissions"),
    3: ("auth", "OTP login (guest path)", "guest login > OTP screen"),
    21: ("auth", "Guest login prompt (after invite popup)", "dismisses invite popup, then taps login header"),
    12: ("ecounter", "Change service location", "e-counter change location > verify"),
    23: ("support", "Support request submit (fast)", "0.9s median span - submits without reading"),
    9: ("service", "Modem control", "home service management > internet tab > modem control"),
    27: ("auth", "Logout", "Account > Log_out > confirm popup > login screen"),
    16: ("guest", "Guest payment hits login wall", "guest billing > LoginForGuestDialog. Conversion blocker"),
    6: ("utility", "QR scan", "home header > scan_qr screen"),
    28: ("auth", "Logout then guest browse", "logout, lands on guest home instead of leaving"),
    17: ("econtract", "E-contract peek and back out", "opens unconfirmed e-contract, hits hd_pl/back"),
    4: ("auth", "OTP login (member path)", "continue_login > login_with_otp > OTP screen"),
    5: ("promotion", "Promotions browse", "Nav_promotion > loyalty promotion webview"),
    22: ("auth", "Guest login tap (minimal)", "guest home > login header > login. 4 steps, no popup"),
    0: ("notification", "Notification detail", "notification detail activity. loops in 41% of journeys"),
    10: ("support", "Support request closed immediately", "same shape as A24, shorter. back 0.19"),
    7: ("account", "Profile open", "Nav_profile > account home. 5 steps"),
    2: ("auth", "FID login", "continue_login > login_with_fid > Authorization"),
    14: ("account", "Profile > contract management", "Nav_profile > account > contract management"),
    13: ("ecounter", "Change address (e-counter)", "34 steps, revisit 0.49, loops 47% - worst Android friction"),
}

# --------------------------------------------------------------------------
# ANDROID - route B (PrefixSpan patterns), 34 clusters
# --------------------------------------------------------------------------
ANDROID_B: dict[int, tuple[str, str, str]] = {
    -1: NOISE,
    3: ("payment", "Guest bill payment", "guest billing checkbox + buttons + text field"),
    6: ("payment", "Bill payment", "payment screen > payment info"),
    14: ("auth", "Guest login prompt", "guest home > login header > login"),
    20: ("payment", "Postpaid bill payment", "select bill > click pay"),
    17: ("support", "Support request", "support_create + return to home"),
    32: ("econtract", "E-contract signing (VNeID)", "go_to_screen > unconfirmed e-contract > confirm CCCD"),
    24: ("contract", "Contract sharing", "contract management > e-counter share management"),
    29: ("contract", "Contract switch then service", "change_contract + service management"),
    4: ("auth", "OAuth login (external)", "Authorization > RedirectUriReceiver"),
    11: ("auth", "Guest login prompt (variant)", "same shape as B14"),
    34: ("support", "Support request closed", "Nav_support > confirm_close"),
    1: ("service", "Internet package upgrade (web)", "30 steps, revisit 0.59, loops 43% - friction"),
    12: ("auth", "OTP login", "guest login > OTP screen"),
    23: ("account", "Profile open", "Nav_profile > account"),
    5: ("auth", "FID login", "login_with_fid > Authorization"),
    8: ("ecounter", "Change service location", "contract management > e-counter change location"),
    7: ("payment", "Payment tab bounce", "Nav_payement > payment > back to home"),
    22: ("utility", "QR scan", "home header > scan_qr"),
    31: ("navigation", "Home browse", "HOME + android/Home only, no destination"),
    18: ("econtract", "E-contract confirm (deep)", "deep FE_CONTRACT_BLOCK / CONTRACT_CONFIRM path"),
    26: ("contract", "Contract management and list", "23 steps, revisit 0.45"),
    21: ("service", "Modem control", "internet tab > modem control"),
    33: ("support", "Support request closed (variant)", "same shape as B34"),
    2: ("account", "Profile then payment", "Nav_profile > account > payment > login"),
    10: ("guest", "Guest payment login wall", "guest billing > LoginForGuestDialog"),
    15: ("account", "Account > e-contract", "account > contract management > e-contract"),
    30: ("contract", "Contract switch", "change_contract > contract list"),
    19: ("payment", "Payment after invite dismiss", "invite_update_close then Nav_payement"),
    13: ("account", "Profile / home toggle", "Nav_profile and Nav_home alternating"),
    25: ("contract", "Contract sharing (long)", "33.5 steps, revisit 0.52"),
    28: ("contract", "Contract switch abandoned", "change_contract > choose > btn_back"),
    9: ("guest", "Guest bill lookup", "invite dismiss > guest billing text field"),
    16: ("contract", "Contract switch (short)", "change_contract > contract list, 7 steps"),
    27: ("navigation", "Home browse (short)", "4 steps on HOME only"),
}

# --------------------------------------------------------------------------
# iOS - route A (tf-idf 1..4-grams), 50 clusters
# --------------------------------------------------------------------------
IOS_A: dict[int, tuple[str, str, str]] = {
    -1: NOISE,
    26: ("guest", "Guest banner tap > login wall", "guest banner > PopupInviteLogin. Largest iOS archetype"),
    47: ("service", "Internet package upgrade (web)", "dkol/update-package. revisit 0.44, loops 20% - friction"),
    8: ("support", "Support article browse", "support page view controller > description > request list"),
    10: ("auth", "OTP login", "continue_login > login_with_otp > OTPCreatePin. Cleanest cluster in the set"),
    32: ("service", "Service manage back-out", "Home > btn_back > ServiceManage. back 0.20"),
    28: ("popup", "Reminder popup skipped", "remind_noti shown > skip_remind"),
    31: ("payment", "Payment tab bounce", "Nav_payement > PaymentHome > back to home"),
    39: ("econtract", "E-contract identity input", "tab_info_order > InputIdentification. loops in 52%"),
    30: ("notification", "Notification browse > detail", "view_all > view_noti > DetailsNoti. loops 34%"),
    37: ("contract", "Contract sharing and permissions", "share management, add/remove linked phone. revisit 0.50"),
    33: ("econtract", "E-contract peek and back out", "EContractHome > hd_pl/back"),
    48: ("shop", "Shop webview browse", "STWebRemote > web/shop/home. loops 51%"),
    45: ("payment", "Bill payment (full flow)", "select_bill > pay > PayAction > PaymentResult"),
    11: ("account", "Profile open and return", "Nav_profile > AccountHome > Home"),
    15: ("auth", "Logout then clear login form", "Log_out > popup > LoginVC login_button_clear"),
    29: ("notification", "Notification list open", "Header go_to_screen > noti/view_all"),
    9: ("auth", "OTP login > contract pick", "OTP login then ChooseContract"),
    22: ("contract", "Contract pick > e-contract", "choose_contract > EContractHome"),
    43: ("shop", "Shop order history", "orderHistoryButton > order-history > OrderDetail. back 0.24"),
    1: ("promotion", "Promotions browse", "Nav_promotion > loyalty promotion webview"),
    27: ("payment", "Payment settings and schedule", "Payment/Setting > PaymentSchedule / ExtendService"),
    12: ("account", "Profile open and return (tab)", "Nav_profile > AccountHome > Nav_home"),
    25: ("popup", "Invite popup dismissed", "PopupBigMessage > invite_update_close"),
    42: ("shop", "Shop webview back-out", "web/shop/home > BackButton. back 0.24"),
    2: ("account", "Personal info screen", "Nav_profile > PersonalVC > Nav_home"),
    23: ("contract", "Contract switch", "Home/change_contract > ChooseContract"),
    34: ("payment", "Prepaid payment", "choose_prepaid > PrePaidVC > pay"),
    6: ("contract", "Contract pick > service", "choose_contract > ServiceManage"),
    18: ("contract", "Contract switch (fast)", "change_contract > choose_contract. 4 steps"),
    4: ("contract", "Contract switch (with return)", "change_contract > choose, returns to Home"),
    36: ("econtract", "E-contract PDF sign", "tab_hd_plhd > PDFHost > SuccessSignEcontract"),
    49: ("shop", "Package/product webview thrash", "revisit 0.57, loops 68% - worst loop share on iOS"),
    41: ("payment", "Pay on behalf", "BehalfPayment > BehalfPaymentResult. loops 56%"),
    24: ("contract", "Contract switch abandoned", "change_contract > btn_back / list reload. back 0.17"),
    21: ("contract", "Contract list back-out", "ChooseContract > btn_back. back 0.23"),
    44: ("shop", "Product detail webview", "dkol/product-detail > CreateOrder. revisit 0.50"),
    14: ("auth", "Logout", "Log_out > popup > LoginVC"),
    0: ("shop", "Shop search", "open_url_in_app_with_access_token > web/shop/search"),
    7: ("service", "Contract pick > internet manage", "choose_contract > other_manage_internet > ServiceManage"),
    17: ("contract", "Contract switch (confirmed)", "change_contract > choose_contract, both present"),
    19: ("auth", "Logout (fast)", "4 steps, action ratio 0.50"),
    5: ("contract", "Contract switch > service", "change_contract > choose > ServiceManage"),
    38: ("ecounter", "Change contact info (e-counter)", "change phone/email > confirm. 27 steps, loops 40%"),
    13: ("auth", "Logout via account popup", "AccountHome popup > LoginVC login_button_clear"),
    16: ("auth", "Logout (clean)", "4 steps, revisit 0.0"),
    46: ("shop", "Package order and payment (web)", "order-history > UpdatePackage policy > payment_infor. 25 steps"),
    40: ("econtract", "E-contract list thrash", "38 steps, revisit 0.64, loops 53% - worst friction on iOS"),
    3: ("auth", "Logout from personal info", "PersonalVC popup > Log_out > LoginVC"),
    20: ("auth", "Logout with form clear", "Log_out > login_button_clear"),
    35: ("service", "Package page immediate back-out", "update-package > webHeader BackButton. back 0.23"),
}

# --------------------------------------------------------------------------
# iOS - route B (PrefixSpan patterns), 56 clusters
# --------------------------------------------------------------------------
IOS_B: dict[int, tuple[str, str, str]] = {
    -1: NOISE,
    32: ("auth", "OTP login", "continue_login > login_with_otp > OTPCreatePin"),
    7: ("navigation", "Home browse", "HomeVC only, no destination reached"),
    47: ("payment", "Payment tab > bill info", "Nav_payement > PaymentHome > payment_infor"),
    16: ("shop", "Shop webview browse", "STWebRemote / STWebpage. loops 47%"),
    50: ("payment", "Bill payment (full)", "payment_infor > PayAction > PaymentResult. 18 steps"),
    45: ("econtract", "E-contract PDF view", "PDFHost + loading view + hd_pl tabs"),
    14: ("support", "Support article browse", "support page view controllers"),
    28: ("popup", "Reminder popup skipped", "remind_noti > skip_remind"),
    26: ("econtract", "E-contract peek and back", "hd_pl/back + tab switching"),
    36: ("account", "Profile open", "Nav_profile > AccountHome"),
    29: ("service", "Contract pick > internet manage", "choose_contract > other_manage_internet"),
    54: ("contract", "Contract switch", "change_contract > choose_contract"),
    10: ("service", "Package upgrade > order", "update-package > CreateOrder. 37 steps, loops 45%"),
    43: ("contract", "Contract switch (fast)", "4 steps"),
    23: ("service", "Internet manage open", "other_manage_internet > ServiceManage"),
    39: ("contract", "Contract switch (no pick)", "change_contract but never chooses"),
    6: ("service", "Package upgrade browse", "update-package + BackButton. 32 steps"),
    9: ("guest", "Guest profile > contract", "guest Nav_profile > AccountHome > ManagerContract"),
    33: ("contract", "Contract switch (variant)", "choose_contract first, change_contract after"),
    37: ("account", "Profile / home toggle", "AccountHome > Nav_home alternating"),
    52: ("contract", "Contract sharing", "ManagerContract > ShareManagement. 13 steps"),
    48: ("contract", "Contract switch abandoned", "change_contract > btn_back"),
    24: ("service", "Service manage back-out", "Home btn_back > ServiceManage. back 0.21"),
    42: ("account", "Profile revisit loop", "Nav_profile > AccountHome, loops 26%"),
    46: ("contract", "Contract list browse", "ChooseContract without acting"),
    1: ("auth", "Logout then re-login", "login_button_clear > OTPCreatePin"),
    15: ("popup", "Popup sequence", "PopupVC chain. loops 37%"),
    53: ("account", "Profile open (fast)", "4 steps"),
    22: ("service", "Internet manage back-out", "btn_back > other_manage_internet. back 0.19"),
    3: ("popup", "Reminder popup skipped (variant)", "same shape as B28"),
    34: ("payment", "Prepaid payment", "choose_contract > PrePaidVC > Nav_payement. 20 steps"),
    20: ("service", "Package page back-out", "update-package > BackButton. back 0.19"),
    13: ("account", "Account home browse", "AccountHome > HomeVC only"),
    2: ("payment", "Payment home browse", "PaymentHome > HomeVC only"),
    27: ("support", "Support browse > e-contract", "support pages then EContractHome"),
    31: ("contract", "Contract switch abandoned (variant)", "ChooseContract btn_back"),
    4: ("service", "Package order confirm", "UpdatePackage policy > continue > CreateOrder"),
    21: ("auth", "FID login", "login_with_fid > continue_login > OTP"),
    17: ("service", "Modem and WiFi management", "ManageModem + ManageWiFi + connected devices. 23.5 steps"),
    19: ("payment", "Payment tab open", "Nav_payement > PaymentHome"),
    5: ("service", "TV service package browse", "click_tv_service + update-package. 34 steps, revisit 0.55"),
    41: ("navigation", "Home deeplink > payment / e-contract", "Home go_to_screen fan-out"),
    40: ("contract", "Contract switch back-out", "btn_back > change_contract > ServiceManage"),
    35: ("contract", "Contract sharing and permissions", "ManagerContract + PhonePermissions. 37 steps, revisit 0.55"),
    25: ("guest", "Guest payment tab bounce", "guest home > PaymentHome > Nav_home"),
    30: ("account", "Profile open (minimal)", "4 steps"),
    12: ("contract", "Contract management", "ManagerContract go_to_screen. 16 steps"),
    8: ("account", "Personal info screen", "PersonalVC > Nav_home"),
    49: ("econtract", "E-contract payment", "EContract > other_payment_method > PayAction. loops 74%"),
    44: ("shop", "Shop order history (web)", "order-history > OrderDetail > product-detail. loops 72%"),
    56: ("contract", "Contract switch (variant 2)", "change_contract > choose_contract"),
    18: ("service", "Package > contract switch", "ServiceManage change_contract > BackButton"),
    11: ("service", "Package order (short)", "UpdatePackage continue > CreateOrder. 19 steps"),
    38: ("contract", "Contract switch abandoned (short)", "change_contract > btn_back"),
    51: ("shop", "Product / order webview", "product-detail + order-detail. loops 60%"),
    55: ("contract", "Contract switch (variant 3)", "change_contract > choose_contract"),
}

NAMES: dict[tuple[str, str], dict[int, tuple[str, str, str]]] = {
    ("android", "a_tfidf"): ANDROID_A,
    ("android", "b_prefixspan"): ANDROID_B,
    ("ios", "a_tfidf"): IOS_A,
    ("ios", "b_prefixspan"): IOS_B,
}

# Families ordered by how they read in a report, not alphabetically.
FAMILY_ORDER: tuple[str, ...] = (
    "payment", "contract", "econtract", "service", "auth", "account",
    "support", "shop", "notification", "promotion", "popup", "guest",
    "ecounter", "utility", "navigation", "noise",
)


def lookup(platform: str, route: str, cluster: int) -> tuple[str, str, str]:
    """(family, name, note) for one cluster; falls back rather than raising."""
    table = NAMES.get((platform, route), {})
    return table.get(int(cluster), ("unnamed", f"cluster {cluster}", ""))


def apply_names(frame: pd.DataFrame, platform: str, route: str, column: str = "cluster") -> pd.DataFrame:
    """Add `family`, `cluster_name` and `naming_note` next to the cluster id."""
    out = frame.copy()
    triples = [lookup(platform, route, c) for c in out[column]]
    out.insert(out.columns.get_loc(column) + 1, "cluster_name", [t[1] for t in triples])
    out.insert(out.columns.get_loc(column) + 1, "family", [t[0] for t in triples])
    out["naming_note"] = [t[2] for t in triples]
    return out


def family_summary(frame: pd.DataFrame, platform: str, route: str) -> pd.DataFrame:
    """Roll clusters up to families - the level to actually report on."""
    named = apply_names(frame, platform, route)
    size_col = "size" if "size" in named.columns else None
    grouped = named.groupby("family", sort=False).agg(
        clusters=("cluster", "nunique"),
        journeys=(size_col, "sum") if size_col else ("cluster", "size"),
    )
    grouped = grouped.reindex([f for f in FAMILY_ORDER if f in grouped.index])
    grouped["share"] = (grouped["journeys"] / grouped["journeys"].sum()).round(4)
    return grouped.reset_index()
