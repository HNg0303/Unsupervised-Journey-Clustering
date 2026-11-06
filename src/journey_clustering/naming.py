"""Name clusters with the Hi FPT 3-level taxonomy and audit observed evidence.

Unlike the legacy naming script, this pipeline reads the taxonomy source of
truth directly.  A taxonomy node is eligible for naming only after a literal
screen/event in one of the supplied cluster n-gram files matches its entry in
SUBMODULE_EVIDENCE (and, for a feature, FEATURE_EVIDENCE).  Words of the
taxonomy labels are never evidence by themselves, so a node without tracker
events (e.g. "Mở thẻ tín dụng VPBank") is never named by rule.  A cluster is
named at the most specific level that passes: feature, submodule, then module.
Home and Smart Assistant dismissals only name a cluster that shows nothing
else.  Parent evidence never marks all of its children as observed.

Names a person set in the labeling app are kept with ``--preserve-names``.

The taxonomy input may be either the original Markdown table or an exported
CSV with the columns ``taxonomy_id``, ``business_family``,
``business_submodule`` and ``business_detail``.

Example:
  journey-name \
    --taxonomy /path/to/hifpt-journey-taxonomy.md \
    --android-ngrams /path/to/android_cluster_ngrams.csv \
    --ios-ngrams /path/to/ios_cluster_ngrams.csv \
    --output-dir output/scores/taxonomy_naming \
    --android-input /path/to/android_scores.csv \
    --android-output /path/to/android_scores_taxonomy_named.csv \
    --ios-input /path/to/ios_scores.csv \
    --ios-output /path/to/ios_scores_taxonomy_named.csv \
    --preserve-names /path/to/android_named_clusters.csv \
    --preserve-names /path/to/ios_named_clusters.csv \
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


# Submodule evidence, written by reading every submodule of the Hi FPT sitemap
# against the tracker vocabulary observed in the run-678 n-gram files. A token
# can only support a submodule through its own literal screen/event names; the
# label words of the sitemap are never enough on their own. A submodule absent
# from this table (for example "Mở thẻ tín dụng VPBank", which has no tracker
# event) is never named by the generic scorer.
SUBMODULE_EVIDENCE: dict[str, re.Pattern[str]] = {
    "M01.01": rx(r"authorizationmanagement|redirecturireceiver|login(?!_with_vneid|managebox)|otp_?screen|otpverifydevice|inappauthenotp|authentaccount|resend_otp|otp_resend|fid_popup"),
    "M01.02": rx(r"login_with_vneid|sso_?vneid"),
    "M01.03": rx(r"createpin|change_?pin|forgot_?pin|log_?out|save_?session"),
    "M01.04": rx(r"account_infor|updateprofile|edit_?profile|edit_avatar|full_name_block|date_birth_block|(?:fe)?male_option|go_to_screen/profile$"),
    "M01.05": rx(r"account_?settings?|accountsetting|language|(?<!payment_)policy(?!payment)|apppolicy|terms_of_use|thoa_thuan|chinh_sach_bao_mat|gioi-thieu-ve-hi-fpt|introduce_hifpt|hifpt-fromwhere|know_hifpt"),
    "M02.01": rx(r"(?:^|/)(?:homevc|home|homeactivity|mainactivity|hifpt_homeviewcontroller)$|home_guest$|homeguestvc|(?:^|/)banner$|home_pull_to_refresh|titleblockhome|explore_app|favorite_feature|fav_functions_btn|homecustomizefunction"),
    "M02.02": rx(r"global_search|header/(?:go_to_screen/)?search"),
    "M02.03": rx(r"fpt_?zone|esports?"),
    "M03.01": rx(r"contract_list|choosecontract|choose_contract|change_contract|select_contract|selectlistcontract|listcontract(?!__)|linkcontract|contract_link|btn_link_contract|confirm_link_contract|(?<!behalf/)search_contract|input_search_contract|managercontractvc|authmanagecontract|send_feedback_contract|contract_list/send_feedback"),
    "M03.02": rx(r"contract_management|contract_detail_info|click_contract_info|btn_info_contract|contract_manage_access|decentralizemanagement|e_?invoice|ebill|(?<!camera_)contract_detail|detail_contract_reload|confirm_contract_info"),
    "M03.03": rx(r"ecounter(?!_service_extend|restoreservice|changevneid|updatevneid)|information_change|change_registered_information|changephonenumberemail|change_cus_|update_information_(?:phone|email|id)|changechargeaddress|change_charge_address|change_invoice_address|(?<!autopay_)change_payment_method|changepm|selecttypeofinfotochange|changeinformation|confirmchangedata"),
    "M03.04": rx(r"changevneid|updatevneid|update_citizen_id|transfer_ownership|request_transfer"),
    "M03.05": rx(r"change_the_location|move_modem_around|changeaddress|inputnewaddress|choose_new_address|change_address|edit_address"),
    "M03.06": rx(r"suspend_service|temporary_?stop"),
    "M03.07": rx(r"restore_service|services_restore|restoreservice|upgrade_service|service_upgrade|upgrade_btn|button_upgrade|update-package"),
    "M04.01": rx(r"e[-_]?contract|econtract|hd_pl|plhd|appendix|contract_confirm|signed_successfully|quoteandminutes|e_contract_quotation|pricesecontract|quote_tab|confirm_quote|quote_confirm|download_bbdt"),
    "M04.02": rx(r"econtract_with_vneid|confirm_with_vneid|confirm_cccd|continue_with_vneid"),
    "M04.03": rx(r"acceptance|inspection_report|bbnt"),
    "M04.04": rx(r"thanh_?ly|liquidation"),
    "M05.01": rx(r"modem(?!_around)|rename_box|changenamebox|update_name_box|change_password_box|update_password_box|loginmanagebox"),
    "M05.02": rx(r"wifi(?!_?iot)|ssid_password_wifi_(?:home|guest)|action_wifi|changenamepasswifi|sharepasswordwifi|wifisharevc|copy_password|share_password|wifi_password_modem|schedule_wifi/click_button_block_schedule|time_turn_o(?:n|ff)"),
    "M05.03": rx(r"wifi_iot"),
    "M05.04": rx(r"network_?model|networkchart|visualizenetwork"),
    "M05.05": rx(r"management_device(?!_toggle)|devices?_(?:tab|details|searching|information|detail)|device_control|devicedetailvc|fslistconnecteddevice|connected_devices|locking_internet|block_device|delete_devices|rename_device|change_device_name|update_new_device_name|changeinfodevice|search_device|filter_device|view_device|settingdevicemanager"),
    "M05.06": rx(r"fprotecthistory|device_detection|identify_devices|history_access|trusted_devices"),
    "M05.07": rx(r"access-?point|ap_detail|ap_control|reboot_ap|view_ap_infor|wifi_router|router_detail|view_router|router_contact|wifi_bottom_sheet_ap"),
    "M06.01": rx(r"fshomevc|fsafe(?!_?go)|internet_f?protect(?:_screen)?|setting_fprotect"),
    "M06.02": rx(r"profiles_devices|home_fsafe/profiles|profile_detail|new_profile|change_profile|select_edit_profile|delete_profile|pcuserdetails|pccreatenewprofile|userlistcontrol|assignuser|pcuseraddnewdevice|profile_icon|device_control_profile"),
    "M06.03": rx(r"internet_break|block_schedule|access_?block|schedule_access_blocking|blockschedule|bedtime|night_timeframe|time_setting|update_time|delete_time|view_time_detail|management_device_toggle|child_?protect|childwatch|parental|confirm_off_device_control|block_internet|contentfilter"),
    "M06.04": rx(r"harmful|safe_?search|browsing_protection|tracking_protection|prevent_tracking|protect_browse_web|block_ads|protection_settings|check_block_content|blocked_content|content_block"),
    "M06.05": rx(r"website|web_threats|web_ev[en]+ts|violat|transgress|fsafelist(?:block|allow)|tab_tracking|search_block|detail_web"),
    "M07.01": rx(r"network_health_create_request|network_health|health_?check|internet_health|diagnostics"),
    "M07.02": rx(r"dr\.?_?smart|doctorsmart|errorchecking|check_error|suggest_fix|need_staff_result|cancel_scan|(?:accept|reject)_reboot"),
    "M07.03": rx(r"speed_?test"),
    "M07.04": rx(r"coverage"),
    "M07.05": rx(r"mirrorcle|proactive.?error"),
    "M08.01": rx(r"camera(?!_?cloud)|qldv/camera"),
    "M08.02": rx(r"renew_cloud|cloud_package|extend_cloud|cloud_camera|camera_cloud"),
    "M08.03": rx(r"cloud_combo|combo_cloud"),
    "M08.04": rx(r"tv_box/button_upgrade|tv_service|tv_box|qldv/tv|manage_fpt_play|fplay"),
    "M08.05": rx(r"ultra.?fast|ultrafast|autopay_uf|list-game|searchgame|check_game"),
    "M08.06": rx(r"fsafe_?go"),
    "M08.07": rx(r"prepaid-?wifi|fpt_?wifi|pay_?wifi"),
    "M09.01": rx(r"support_?create|supportcreator|check_report(?:_create)?|support_send|support_confirm(?!_select_schedule)|selectmediasupport|imagesupportviewer|select_type_support|select_content_support|popuprequestchossen|create_request|needsupport|support_claim"),
    "M09.02": rx(r"support_create_quick|intent_unknow|voice_command|voicevc|quick_action"),
    "M09.03": rx(r"supp?ort_?history|support_request_history|listsupportrequest|reportstatus|requeststatus|report_?detail|detailreport|support_?_?detail|support_status_detail|support_information|support_inprogress|support_complete|urge_process|call_request|call_technician|support_cancel|filterreport|support_filter"),
    "M09.04": rx(r"appointment|time_?slot|support_confirm_select_schedule"),
    "M09.05": rx(r"srconfirm"),
    "M09.06": rx(r"go_to_screen/chat$|chatbot|chatconversation|support_chat|chat_list_message|start_chat|faq|contactlocator|paylocator|searchofficebyprovince|support_location_list"),
    "M09.07": rx(r"proactive_support|floating_support"),
    "M09.08": rx(r"hi_?fpt_?lite"),
    "M10.01": rx(r"paymenthomevc|payment_home|^(?:android/)?home/payment$|payment_infor(?!mation_fgold)|select_bill|bill_home|bill_icon|action_tap_bill|click_update_bill|internetbillview|payment/pay$|confirm_payment|checkout|qr_payment|payment_qr|paymentcontractqr|pay_back_immediately"),
    "M10.02": rx(r"behalf|nonloginpayment|guestbilling|guestpaymenthistory|payment_guest|bill_on_search|input_contract_no_history"),
    "M10.03": rx(r"prepaid(?!-?wifi)|prepay|block_prepay"),
    "M10.04": rx(r"autopay|auto_pay|homeautopay|paymentautocan[c]?el|fixed_day|update_payment_day|choose_scheduled_payment_date"),
    "M10.05": rx(r"my_cards|mywallet|my_wallet|link_card|link_wallet|add_payment_method"),
    "M10.06": rx(r"payment/payment_qr|vietqr|paymentqrvc|confirm_payment/\w+_qr|select_method|select_new_method|choose_new_method|confirm_new_method|payment_method|other_methods|napas|select_bank_app|apple_?pay|google_?pay|momo|zalopay|shopeepay"),
    "M10.07": rx(r"select_vouchers|paymentvoucher|promotion_code|offer_payment|payment_information_fgold|payment_information/fgold"),
    "M10.08": rx(r"payment_?history|history_payment|historypayment|paymenthistory|results_payment|succesful_payment|successful_payment|fail_payment|retry_payment|transaction_result|paymentresult|transaction_detail|pay_in_success|pending_payment|payment_search"),
    "M10.09": rx(r"payment/setting|payment_settings|paymentsettings|notification_channel|payment_reminder|schedule_payment|paymentschedule|payment_?extend|payment_extension|extendservice|ecounter_service_extend|click_payment_mgmt|payment-ultilities|view_utilities|paymentutility|billingutilities"),
    "M10.10": rx(r"paylater|bnpl"),
    "M11.01": rx(r"createorder|web/shop|shop/home|shophome|shopproduct|product-?management|product-?detail|productitem|detail-product|hisale|hifptsaleblock|sale-platform|locationbutton|body/category|webkitecommerce"),
    "M11.02": rx(r"thiet-bi-thong-minh/[a-z-]+|thiet-bi-[a-z-]+|truyen-hinh-fpt-play|internet-ca-nhan|gia-dinh|combo-internet[a-z-]*|goi-vvip\d*|shop/fpt-play|register_camera|register_tv|product-management/camera|medicare|bao-mat-f-safe|dich-vu-so"),
    "M11.03": rx(r"address-book"),
    "M11.04": rx(r"order-?history|orderhistory|order-info|order-detail|orderdetail|view_detail_order|tab_info_order|updateorder|registerinfo__submit|register-info|buying-history"),
    "M11.05": rx(r"thanh-toan-(?:dien|nuoc)|electric_bill|water_bill"),
    "M12.01": rx(r"loylaty_member|loyalty_member|current_rank|join_now|verify_now|loyalty_popup_cccd|fgold|loylaty_rules|loyalty_rules|loyaltysdk|loymainactivity|loylaty_contact"),
    "M12.02": rx(r"loylaty_promotion|loyalty_promotion|loylaty_home|loyalty_home|loalty_home|loyalty-webview|loyalty/home|uu-dai|voucher|my_gifts|promo_detail|banner_detail|header/loyalty|account/loyalty|giftcode|changetabgift"),
    "M12.03": rx(r"redeem"),
    "M13.01": rx(r"refer_?friend|referfriends|friendreferral|gtbb|refercode|referlink"),
    "M13.02": rx(r"invi?n?te_download_app|invite_download"),
    "M14.01": rx(r"scan_?qr|scanqr|qrscanner|nonftel_qr_scan|open_setting_camera"),
    "M14.02": rx(r"employee|info-staff"),
    "M14.03": rx(r"vpbank"),
    "M14.04": rx(r"sao-?may|tarot"),
    "M14.05": rx(r"nha-?tro|smart_rental|smart_?motel"),
    "M14.06": rx(r"web/game|game-checkin|loyalty/game|sandbox"),
    "M15.01": rx(r"homenotification|notification(?!_channel|settings|/settings?\b)|notidetail|detailsnoti|view_noti|noti/view_|delete_noti|mark_read_all|delete_read_noti|select_noti|delete_selected_noti|confirm_delete_noti|click_noti"),
    "M15.02": rx(r"notification/settings?$|notificationsettings|notisetting|setting_noti|click_toggle_\w*noti|noti_guest"),
    "M15.03": rx(r"sa_(?:mascot|msg|short_msg|full_msg)|smart_?assistant|popup_remind|remind_noti|skip_remind"),
    "M15.04": rx(r"widget"),
    "M16.01": rx(r"csat"),
    "M16.02": rx(r"(?:^|[/_])nps(?:$|[/_])"),
    "M16.03": rx(r"rating_in_app|rate_?star|gotorate|experiencereview|feedback|my_rating"),
    "M16.04": rx(r"survey|(?:^|[/_])fli(?:$|[/_])"),
}
# Hubs that appear in almost every journey: the Home screen and dismissing the
# Smart Assistant popup that opens on it.
HUB_SUBMODULES = {"M02.01"}
HUB_FEATURES = {"M15.03.10"}
SCOPES = ("focus", "all")
# Events that only navigate or dismiss. They say nothing about the feature, so
# such an action token is judged by the screen it happened on instead.
GENERIC_EVENT = rx(
    r"(?:^|/)(?:btn_back|back|backbutton|handleback|pop_back|adsview|view_os_noti|"
    r"popup|pop_up_close|close|show|buttons?|body|tab|more|null|unknown|<missing>|"
    r"nav_(?:home|payement|payment|profile|support|promotion)|call_api_get_action)$"
)
# Feature evidence: the literal screen/event that shows one sitemap feature.
# A feature is chosen only when its literal matches inside a token that already
# belongs to the feature's submodule. Features without an entry stay at
# submodule level; their label words are never used as evidence.
FEATURE_EVIDENCE: dict[str, re.Pattern[str]] = {
    "M01.01.01": rx(r"login_with_fid|fid_popup"),
    "M01.01.02": rx(r"login_with_otp|otp_?screen|otpverifydevice|inappauthenotp"),
    "M01.01.03": rx(r"resend_otp|otp_resend"),
    "M01.02.01": rx(r"login_with_vneid"),
    "M01.03.01": rx(r"createpin"),
    "M01.03.05": rx(r"log_?out"),
    "M01.04.01": rx(r"account_infor$|go_to_screen/profile$"),
    "M01.04.02": rx(r"full_name_block"),
    "M01.04.03": rx(r"date_birth_block"),
    "M01.04.04": rx(r"(?:fe)?male_option"),
    "M01.04.05": rx(r"edit_avatar"),
    "M01.05.01": rx(r"language"),
    "M01.05.02": rx(r"policy|terms_of_use|thoa_thuan|chinh_sach_bao_mat"),
    "M01.05.03": rx(r"gioi-thieu-ve-hi-fpt|introduce_hifpt"),
    "M01.05.04": rx(r"hifpt-fromwhere|know_hifpt"),
    "M02.01.01": rx(r"(?:^|/)(?:homevc|home|homeactivity|mainactivity|hifpt_homeviewcontroller)$|home_pull_to_refresh"),
    "M02.01.02": rx(r"(?:^|/)banner$"),
    "M02.01.04": rx(r"home_guest$|homeguestvc"),
    "M03.01.01": rx(r"contract_list$|contract_list_reload|choosecontractvc|listcontract|managercontractvc"),
    "M03.01.02": rx(r"search_contract"),
    "M03.01.03": rx(r"change_contract|choose_contract|select_contract"),
    "M03.01.04": rx(r"save_contract_link|confirm_link_contract|btn_link_contract"),
    "M03.01.06": rx(r"send_feedback"),
    "M03.02.01": rx(r"contract_management$|contract_detail_info$|click_contract_info|btn_info_contract|confirm_contract_info"),
    "M03.02.02": rx(r"contract_manage_access|decentralizemanagement"),
    "M03.02.04": rx(r"e_?invoice|ebill"),
    "M03.03.01": rx(r"change_cus_phone|changephonenumberemail|update_information_phone|changephonesuccess"),
    "M03.03.02": rx(r"update_information_email|confirm_information_email"),
    "M03.03.03": rx(r"change_cus_id|update_information_id"),
    "M03.03.04": rx(r"change_payment_method|changepm(?!address)"),
    "M03.03.05": rx(r"change_charge_address|changechargeaddress|changepmaddress"),
    "M03.03.06": rx(r"change_invoice_address"),
    "M03.03.08": rx(r"ecounter_?history|ecountertransaction|historyecounter"),
    "M03.04.01": rx(r"update_citizen_id|changevneid|updatevneid"),
    "M03.04.03": rx(r"transfer_ownership|request_transfer"),
    "M03.05.01": rx(r"change_the_location"),
    "M03.05.02": rx(r"inputnewaddress|choose_new_address"),
    "M03.07.01": rx(r"click_restore_service_tab"),
    "M03.07.03": rx(r"services_restore|restoreserviceconfirm"),
    "M03.07.04": rx(r"upgrade_service|update-package|upgrade_btn|button_upgrade"),
    "M04.01.01": rx(r"listecontract|tab_hd_plhd|account/e-contract$"),
    "M04.01.03": rx(r"econtractdetail|view_details/e_contract|click_contract_appendix|fview_details"),
    "M04.01.04": rx(r"contract_confirm|signed_successfully|success_sign_econtract"),
    "M04.01.06": rx(r"e_contract_quotation|quote_confirm|confirm_quote|quoteandminutes|pricesecontract"),
    "M04.02.02": rx(r"confirm_cccd"),
    "M04.02.03": rx(r"econtract_with_vneid|confirm_with_vneid"),
    "M04.03.01": rx(r"inspection_report_list|acceptancereportlist|acceptance_list"),
    "M04.03.05": rx(r"inspection_report_confirm|confirmacceptance"),
    "M05.01.01": rx(r"managemodem|view_modem|modem_detail|modem_home|modem_control$|click_modem_control|modem_setting|modemmanager"),
    "M05.01.02": rx(r"rename_modem|rename_box|changenamebox|update_name_box"),
    "M05.01.03": rx(r"modem_reset|reboot_modem|restart_modem"),
    "M05.01.04": rx(r"modem_schedule|restartschedule"),
    "M05.02.01": rx(r"action_wifi(?:_[25]ghz)?$|managewifi|wifi_control$|wifi_bottom_sheet_home$|wifimanager"),
    "M05.02.02": rx(r"rename_?wifi"),
    "M05.02.03": rx(r"change_wifi_password|update_wifi_password"),
    "M05.02.04": rx(r"copy_password"),
    "M05.02.05": rx(r"share_password|sharepasswordwifi|wifisharevc"),
    "M05.02.06": rx(r"turn_on_wifi|wifi_[25]ghz_turn_on"),
    "M05.02.07": rx(r"turn_off_wifi|wifi_[25]ghz_turn_off"),
    "M05.02.08": rx(r"new_wifi_schedule|add_wifi_schedule|wificreateschedule|addoreditschedulewifi|click_wifi_schedule"),
    "M05.02.09": rx(r"update_wifi_schedule"),
    "M05.04.01": rx(r"network_model$|networkchart|visualizenetwork|click_network_model"),
    "M05.04.02": rx(r"refresh_network_model"),
    "M05.05.01": rx(r"devices_tab_all"),
    "M05.05.02": rx(r"devices_tab_connected|connected_devices"),
    "M05.05.03": rx(r"devices_tab_blocking"),
    "M05.05.04": rx(r"devices_searching|search_device"),
    "M05.05.05": rx(r"filter_device"),
    "M05.05.06": rx(r"devices_details$|device_information$|devicedetailvc|device_detail$|view_device"),
    "M05.05.07": rx(r"rename_device|change_device_name|update_new_device_name|changeinfodevice"),
    "M05.05.09": rx(r"(?<!un_)locking_internet|confirm_block_device"),
    "M05.05.10": rx(r"un_locking_internet|unblock_device"),
    "M05.05.11": rx(r"delete_devices"),
    "M05.06.01": rx(r"identify_devices_toggle|click_on_device_detection"),
    "M05.06.03": rx(r"device_detection_screen"),
    "M05.06.04": rx(r"trusted_devices"),
    "M05.06.05": rx(r"history_access|fprotecthistory"),
    "M05.07.02": rx(r"accesspointmanager|wifi_router_screen$|router_contact"),
    "M05.07.03": rx(r"ap_detail|router_detail$|view_router_item|view_ap_infor"),
    "M05.07.04": rx(r"accesspointdevicechangename"),
    "M05.07.05": rx(r"reboot_ap"),
    "M06.01.01": rx(r"fshomevc|internet_f?protect_screen$|fsafe_home|fsafe/home|gotohomefsafe"),
    "M06.01.03": rx(r"fsafe_setting$|setting_fprotect"),
    "M06.02.01": rx(r"profiles_devices(?:_detail)?$|home_fsafe/profiles$|profile_detail$|pcuserdetails|userlistcontrol"),
    "M06.02.02": rx(r"new_profile|pccreatenewprofile"),
    "M06.02.04": rx(r"assignuser|pcuseraddnewdevice"),
    "M06.03.01": rx(r"management_device_toggle|child_?protect|confirm_off_device_control|parental_control"),
    "M06.03.02": rx(r"contentfilter"),
    "M06.03.03": rx(r"new_time_setting|save_new_time|add_schedule|blockschedule|block_schedule|bedtime|night_timeframe"),
    "M06.03.04": rx(r"update_time|fsafeblockingtimeeditor"),
    "M06.03.05": rx(r"delete_time"),
    "M06.04.01": rx(r"block_ads|ads_block"),
    "M06.04.02": rx(r"browsing_protection|protect_browse_web"),
    "M06.04.03": rx(r"tracking_protection|prevent_tracking"),
    "M06.04.05": rx(r"safe_?search"),
    "M06.04.06": rx(r"harmful|content_block|check_block_content"),
    "M06.05.01": rx(r"allow_website_screen|allowed_websites|fsafelistallow|priority_website"),
    "M06.05.02": rx(r"add_website_allow"),
    "M06.05.03": rx(r"delete_allowed_webiste"),
    "M06.05.04": rx(r"block_website_screen|blocked_websites|fsafelistblock|website_list"),
    "M06.05.05": rx(r"add_website_block|add_blocked_websites|save_update_blocked_website|addblockwebsite"),
    "M06.05.06": rx(r"delete_website|delete_blocked_webiste"),
    "M06.05.07": rx(r"web_ev[en]+ts|transgress|violat"),
    "M06.05.09": rx(r"web_threats"),
    "M07.01.01": rx(r"scan_network_health$|scan_result_network_health$"),
    "M07.01.12": rx(r"network_health_create_request"),
    "M07.02.01": rx(r"doctorsmartvc|errorcheckingdrsmart|check_error$"),
    "M07.02.02": rx(r"cancel_scan"),
    "M07.02.03": rx(r"scan_result$|suggest_fix_result"),
    "M07.02.04": rx(r"suggest_fix_detail"),
    "M07.02.05": rx(r"(?:accept|reject)_reboot"),
    "M07.02.07": rx(r"need_staff_result|continue_request"),
    "M07.03.01": rx(r"speed_?test"),
    "M07.04.01": rx(r"coverage"),
    "M08.01.01": rx(r"camera_service_tab$|qldv/camera|other_manage_camera"),
    "M08.01.02": rx(r"detailcamera|view_detail_camera$|camera_contract_detail|hifptcameradetail"),
    "M08.02.01": rx(r"renew_cloud"),
    "M08.04.01": rx(r"tv_service_tab$|qldv/tv_box$|other_manage_fpt_play"),
    "M08.04.02": rx(r"tv_box/button_upgrade"),
    "M08.05.04": rx(r"ultra_fast_detail_service|ultra_fast_management"),
    "M08.05.05": rx(r"onoffultrafastadvance"),
    "M08.05.07": rx(r"list-game|ultra_fast_list_game|searchgame|check_game"),
    "M08.07.01": rx(r"prepaid-?wifi/?introducing|prepaid-wifiintroducing"),
    "M08.07.03": rx(r"buying-history"),
    "M09.01.05": rx(r"upload_media|selectmediasupport|imagesupportviewer"),
    "M09.01.06": rx(r"input_contact_(?:phone|name)"),
    "M09.02.02": rx(r"support_create_quick"),
    "M09.02.03": rx(r"voice_command|voicevc"),
    "M09.02.05": rx(r"intent_unknow"),
    "M09.03.01": rx(r"supp?ort_history_view$|support_request_history|listsupportrequest|call_requestlist"),
    "M09.03.02": rx(r"support_inprogress|detail_inprogress|information_inprogress"),
    "M09.03.03": rx(r"support_complete|detail_complete|information_complete"),
    "M09.03.04": rx(r"reportdetail|detailreport|report_detail|support_detail$|support__detail_status|requeststatus|reportstatus"),
    "M09.03.06": rx(r"call_technician|call_requestdetail"),
    "M09.03.07": rx(r"support_cancel|support__detail_cancel"),
    "M09.04.01": rx(r"support_confirm_select_schedule|set_appointment$"),
    "M09.04.02": rx(r"agree_appointment"),
    "M09.04.03": rx(r"change_timeslot"),
    "M09.05.01": rx(r"srconfirm/completed"),
    "M09.05.02": rx(r"srconfirm/needsupport"),
    "M09.06.01": rx(r"support_chat|chat_list_message|start_chat|go_to_screen/chat$"),
    "M09.06.02": rx(r"chatbot"),
    "M09.06.03": rx(r"faq"),
    "M09.06.04": rx(r"contactlocator|searchofficebyprovince|paylocator"),
    "M10.01.01": rx(r"paymenthomevc|payment_home|^(?:android/)?home/payment$|bill_home|bill_icon"),
    "M10.01.02": rx(r"select_bill|action_tap_bill"),
    "M10.01.03": rx(r"payment_infor$"),
    "M10.01.04": rx(r"confirm_payment$|checkout|payment/pay$|pay_back_immediately"),
    "M10.01.05": rx(r"qr_payment|payment_qr|paymentcontractqr"),
    "M10.02.01": rx(r"search_contract|bill_on_search_behalf|choose_behalf"),
    "M10.02.02": rx(r"behalfpayment|pay_on_behalf"),
    "M10.02.03": rx(r"guestbillingtab|nonloginpaymenthomevc"),
    "M10.02.05": rx(r"guestpaymenthistory|nonloginpaymenthistory|click_view_history"),
    "M10.03.01": rx(r"register_prepaid|click_view_prepay|prepay_icon|prepaidvc|payment/prepaid$"),
    "M10.03.04": rx(r"choose_prepaid(?:_package)?$"),
    "M10.03.05": rx(r"(?:change|update)_prepaid_package"),
    "M10.03.06": rx(r"delete_prepaid_bill"),
    "M10.03.07": rx(r"confirm_prepaid|prepaid_confirmation|transaction_result_prepay"),
    "M10.04.01": rx(r"autopay_activate|enable_autopay|on_autopay"),
    "M10.04.03": rx(r"autopay_change_payment_method"),
    "M10.04.04": rx(r"cancel_autopay|paymentautocan[c]?el"),
    "M10.04.05": rx(r"fixed_day|update_payment_day|choose_scheduled_payment_date"),
    "M10.05.01": rx(r"my_cards$|mywallet|my_wallet"),
    "M10.06.01": rx(r"payment/payment_qr|vietqr|paymentqrvc|confirm_payment/\w+_qr"),
    "M10.06.04": rx(r"select_bank_app"),
    "M10.06.09": rx(r"napas"),
    "M10.07.01": rx(r"select_vouchers|paymentvoucher"),
    "M10.07.05": rx(r"payment_information_fgold|payment_information/fgold"),
    "M10.08.01": rx(r"payment_?history$|paymenthistoryvc|historypayment|history_payment$"),
    "M10.08.02": rx(r"history_search|payment_search"),
    "M10.08.03": rx(r"payment_history_apply_filter|history_filter"),
    "M10.08.04": rx(r"transaction_detail|succesful_payment/view_detail"),
    "M10.08.05": rx(r"results_payment|succesful_payment$|fail_payment|retry_payment|transaction_result|paymentresult|pay_in_success|pending_payment"),
    "M10.09.01": rx(r"payment/setting$|payment_settings$|paymentsettings|click_payment_mgmt"),
    "M10.09.02": rx(r"notification_channel"),
    "M10.09.03": rx(r"payment_reminder|schedule_payment|paymentschedule"),
    "M10.09.04": rx(r"payment_?extend|payment_extension|extendservice|ecounter_service_extend"),
    "M11.01.01": rx(r"shop/home|shophome"),
    "M11.01.02": rx(r"product-?management|shopproduct|body/category"),
    "M11.01.04": rx(r"product-?detail(?!__)|productitem|detail-product"),
    "M11.01.05": rx(r"hisale|hifptsaleblock"),
    "M11.01.07": rx(r"locationbutton"),
    "M11.01.08": rx(r"createorder"),
    "M11.02.01": rx(r"internet-ca-nhan"),
    "M11.02.02": rx(r"combo-internet|gia-dinh"),
    "M11.02.03": rx(r"access-point"),
    "M11.02.04": rx(r"thiet-bi-camera|register_camera|product-management/camera"),
    "M11.02.05": rx(r"truyen-hinh-fpt-play|register_tv"),
    "M11.02.06": rx(r"goi-vvip|shop/fpt-play"),
    "M11.02.07": rx(r"thiet-bi-smart-home|thiet-bi-thong-minh$"),
    "M11.02.09": rx(r"medicare"),
    "M11.02.10": rx(r"bao-mat-f-safe"),
    "M11.03.01": rx(r"address-book"),
    "M11.04.01": rx(r"order-?history|orderhistory"),
    "M11.04.03": rx(r"order-?detail|orderdetail|order-info|view_detail_order|tab_info_order"),
    "M12.01.01": rx(r"join_now"),
    "M12.01.02": rx(r"current_rank|loylaty_member"),
    "M12.01.03": rx(r"fgold"),
    "M12.02.01": rx(r"loylaty_promotion_view$|loyalty_home_view$|header/loyalty|account/loyalty|loyalty-webview|loyalty/home"),
    "M12.02.02": rx(r"non-login-uu-dai|non-login/voucher"),
    "M12.02.03": rx(r"my_gifts|my-voucher"),
    "M12.02.04": rx(r"promo_detail|banner_detail|voucher/detail|detailvoucher|promotiondetailbutton"),
    "M12.03.01": rx(r"redeem"),
    "M13.01.02": rx(r"copy_refercode"),
    "M13.01.05": rx(r"share_referlink"),
    "M13.01.06": rx(r"guide_gtbb"),
    "M13.01.07": rx(r"about-gtbb"),
    "M13.02.04": rx(r"invi?n?te_download_app|invite_download"),
    "M14.01.01": rx(r"scan_?qr_screen$|scanqrvc|nonftel_qr_scan|go_to_screen/scanqr"),
    "M14.01.02": rx(r"upload_image"),
    "M14.02.01": rx(r"verify_employee|employeeqrcodeemail"),
    "M14.02.04": rx(r"view_employee|info-staff"),
    "M14.04.01": rx(r"sao-?may"),
    "M14.05.01": rx(r"nha-?tro|smart_rental"),
    "M14.06.01": rx(r"game-checkin|web/game"),
    "M15.01.01": rx(r"homenotification|home/notification$|go_to_screen/notification$|view_noti$|notificationdata"),
    "M15.01.02": rx(r"noti/view_\w+_cate|noti/view_all"),
    "M15.01.03": rx(r"notification/detail|notificationdetail|detailsnoti|notificationbuttondetail|click_noti"),
    "M15.01.05": rx(r"mark_read_all"),
    "M15.01.06": rx(r"(?<!_)delete_noti"),
    "M15.01.07": rx(r"select_noti|delete_selected_noti|select_delete"),
    "M15.01.08": rx(r"delete_read_noti"),
    "M15.02.01": rx(r"click_toggle_\w*noti|setting_noti|notification/settings?$|notificationsettings|notisetting"),
    "M15.03.10": rx(r"sa_\w*close|skip_remind"),
    "M15.04.02": rx(r"widget"),
    "M16.01.07": rx(r"csat_view_rating|my_csat_list|my_csat_history"),
    "M16.03.01": rx(r"rate_?star|rating_in_app"),
    "M16.03.04": rx(r"send_feedback|(?:^|/)feedback$"),
    "M16.03.06": rx(r"feedbackreportstatus|feedbackrequeststatus"),
    "M16.04.01": rx(r"survey"),
}


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


def evidence_text(token: str) -> str:
    """Return the part of a tracker token that says what the user did.

    ``action@screen#event`` is judged by its event unless the event only
    navigates or dismisses (back, ad view, bottom tab), in which case the
    screen it happened on is the evidence. ``view@screen`` is its screen.
    """
    if token in {"<bos>", "<eos>"}:
        return ""
    kind, separator, rest = token.partition("@")
    if not separator:
        return token
    if kind == "action" and "#" in rest:
        screen, _, event = rest.partition("#")
        rest = screen if GENERIC_EVENT.search(event) else event
    return "" if rest == "<missing>" else rest


def deepest_match(text: str, patterns: dict[str, re.Pattern[str]]) -> str | None:
    """Return the key whose literal sits deepest in a tracker path.

    Paths go from general to specific (``internet_fprotect_screen/
    management_device/.../locking_internet``), so the match that ends last
    wins; a longer literal breaks a tie and a remaining tie is ambiguous.
    """
    best: dict[str, tuple[int, int]] = {}
    for key, pattern in patterns.items():
        matches = list(pattern.finditer(text))
        if matches:
            last = max(matches, key=lambda match: (match.end(), match.end() - match.start()))
            best[key] = (last.end(), last.end() - last.start())
    if not best:
        return None
    top = max(best.values())
    winners = [key for key, value in best.items() if value == top]
    return winners[0] if len(winners) == 1 else None


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


def token_evidence(
    token: str,
    submodule_patterns: dict[str, re.Pattern[str]],
    feature_patterns: dict[str, dict[str, re.Pattern[str]]],
) -> tuple[str | None, str | None]:
    """Return the (submodule ID, feature ID) one tracker token supports."""
    text = evidence_text(token)
    submodule_id = deepest_match(text, submodule_patterns) if text else None
    if submodule_id is None:
        return None, None
    return submodule_id, deepest_match(text, feature_patterns.get(submodule_id, {}))


def analyse(
    leaves: list[TaxonomyLeaf], rows: list[dict[str, object]], min_evidence_rows: int,
    min_mass_coverage: float, min_score_share: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    leaves_by_submodule: dict[str, list[TaxonomyLeaf]] = defaultdict(list)
    for leaf in leaves:
        leaves_by_submodule[leaf.taxonomy_id[:6]].append(leaf)
    # Evidence for nodes absent from the supplied taxonomy is ignored; the
    # tables themselves are checked against the sitemap in the tests.
    submodule_patterns = {
        submodule_id: pattern for submodule_id, pattern in SUBMODULE_EVIDENCE.items()
        if submodule_id in leaves_by_submodule
    }
    leaf_ids = {leaf.taxonomy_id for leaf in leaves}
    feature_patterns: dict[str, dict[str, re.Pattern[str]]] = defaultdict(dict)
    for taxonomy_id, pattern in FEATURE_EVIDENCE.items():
        if taxonomy_id in leaf_ids:
            feature_patterns[taxonomy_id[:6]][taxonomy_id] = pattern
    module_names = {leaf.taxonomy_id[:3]: leaf.module for leaf in leaves}
    submodule_names = {
        submodule_id: (values[0].module, values[0].submodule)
        for submodule_id, values in leaves_by_submodule.items()
    }

    leaf_evidence: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    sub_evidence: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    # Scores and per-row evidence are kept twice: "focus" leaves out hub
    # tokens (see HUB_SUBMODULES), "all" keeps them for the fallback pass.
    cluster_scores = {scope: defaultdict(Counter) for scope in SCOPES}
    cluster_sub_scores = {scope: defaultdict(Counter) for scope in SCOPES}
    row_leaves: dict[str, dict[int, set[str]]] = {scope: {} for scope in SCOPES}
    row_subs: dict[str, dict[int, set[str]]] = {scope: {} for scope in SCOPES}
    cluster_rows: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    leaf_by_id = {leaf.taxonomy_id: leaf for leaf in leaves}
    taxonomy_paths = {(leaf.module, leaf.submodule) for leaf in leaves}
    unknown_reviewed_ids = {
        rule.target for rule in REVIEWED_RULES if not rule.target.startswith("@")
    } - set(leaf_by_id)
    if unknown_reviewed_ids:
        raise ValueError(f"reviewed rules reference unknown taxonomy IDs: {sorted(unknown_reviewed_ids)}")
    unknown_reviewed_paths = {
        tuple(rule.target[1:].split("|", 1))
        for rule in REVIEWED_RULES if rule.target.startswith("@")
    } - taxonomy_paths
    if unknown_reviewed_paths:
        raise ValueError(f"reviewed rules reference unknown taxonomy paths: {sorted(unknown_reviewed_paths)}")

    token_cache: dict[str, tuple[str | None, str | None]] = {}
    for row in rows:
        platform, cluster, text = str(row["platform"]), int(row["cluster"]), str(row["ngram"])
        cluster_rows[(platform, cluster)].append(row)
        subs: dict[str, set[str]] = {scope: set() for scope in SCOPES}
        details: dict[str, set[str]] = {scope: set() for scope in SCOPES}
        for token in text.split():
            if token not in token_cache:
                token_cache[token] = token_evidence(token, submodule_patterns, feature_patterns)
            submodule_id, taxonomy_id = token_cache[token]
            if submodule_id is None:
                continue
            hub = submodule_id in HUB_SUBMODULES or taxonomy_id in HUB_FEATURES
            for scope in ("all",) if hub else SCOPES:
                subs[scope].add(submodule_id)
                if taxonomy_id:
                    details[scope].add(taxonomy_id)
        weight = float(row["mass"]) * max(math.log1p(float(row["lift"])), 0.1)
        for taxonomy_id in details["all"]:
            leaf_evidence[(platform, taxonomy_id)].append(row)
        for submodule_id in subs["all"]:
            sub_evidence[(platform, submodule_id)].append(row)
        for scope in SCOPES:
            row_subs[scope][id(row)], row_leaves[scope][id(row)] = subs[scope], details[scope]
            for taxonomy_id in details[scope]:
                cluster_scores[scope][(platform, cluster)][taxonomy_id] += weight / len(details[scope])
            for submodule_id in subs[scope]:
                cluster_sub_scores[scope][(platform, cluster)][submodule_id] += weight / len(subs[scope])

    validation: list[dict[str, object]] = []
    for leaf in leaves:
        submodule_id = leaf.taxonomy_id[:6]
        patterns = [
            table[key] for table, key in (
                (SUBMODULE_EVIDENCE, submodule_id), (FEATURE_EVIDENCE, leaf.taxonomy_id),
            ) if key in table
        ]
        for platform in PLATFORMS:
            evidence = sorted(
                leaf_evidence.get((platform, leaf.taxonomy_id), []),
                key=lambda row: (-float(row["mass"]), int(row["rank"])),
            )
            parent = sub_evidence.get((platform, submodule_id), [])
            validation.append({
                "platform": platform, "taxonomy_id": leaf.taxonomy_id,
                "module": leaf.module, "submodule": leaf.submodule, "detail": leaf.detail,
                "module_observed": str(bool(parent)).lower(),
                "submodule_observed": str(bool(parent)).lower(),
                "detail_observed": str(len(evidence) >= min_evidence_rows).lower(),
                "detail_evidence_rows": len(evidence),
                "detail_cluster_count": len({int(row["cluster"]) for row in evidence}),
                "evidence_pattern": " AND ".join(pattern.pattern for pattern in patterns),
                "supporting_evidence": " || ".join(str(row["ngram"]) for row in evidence[:3]),
            })

    mappings: list[dict[str, object]] = []
    for (platform, cluster), ngram_rows in sorted(cluster_rows.items()):
        total_mass = sum(float(row["mass"]) for row in ngram_rows) or 1.0
        chosen = None
        evidence: list[dict[str, object]] = []
        module, submodule, detail = "Chưa phân loại", "Không đủ bằng chứng", ""
        source, confidence, score_share = "insufficient_observed_evidence", "low", 0.0
        scopes = SCOPES
        if cluster == -1:
            module, submodule, detail = "Chưa phân loại", "Journey hỗn hợp/nhiễu", ""
            source, confidence, score_share = "noise_cluster", "low", 0.0
            scopes = ()
        # Hub tokens sit in nearly every journey. They may name a cluster only
        # when it shows no other evidence at all.
        if any(row_subs["focus"][id(row)] for row in ngram_rows):
            scopes = scopes[:1]
        for scope in scopes:
            detail_scores = cluster_scores[scope][(platform, cluster)]
            sub_scores = cluster_sub_scores[scope][(platform, cluster)]
            if detail_scores:
                taxonomy_id, score = detail_scores.most_common(1)[0]
                candidate_evidence = [row for row in ngram_rows if taxonomy_id in row_leaves[scope][id(row)]]
                candidate_share = score / (sum(detail_scores.values()) or 1.0)
                candidate_coverage = sum(float(row["mass"]) for row in candidate_evidence) / total_mass
                # The feature must also lead at submodule level; a strong
                # feature inside a minority submodule is not the cluster's name.
                parent_share = sub_scores[taxonomy_id[:6]] / (sum(sub_scores.values()) or 1.0)
                if (
                    len(candidate_evidence) >= min_evidence_rows
                    and candidate_share >= min_score_share
                    and parent_share >= min_score_share
                    and candidate_coverage >= min_mass_coverage
                ):
                    chosen = leaf_by_id[taxonomy_id]
                    module, submodule, detail = chosen.module, chosen.submodule, chosen.detail
                    score_share, evidence = candidate_share, candidate_evidence
                    confidence = "high" if candidate_share >= .70 and candidate_coverage >= .30 else "medium"
                    source = "validated_taxonomy_detail"
                    break
            if sub_scores:
                submodule_id, score = sub_scores.most_common(1)[0]
                candidate_share = score / (sum(sub_scores.values()) or 1.0)
                candidate_evidence = [row for row in ngram_rows if submodule_id in row_subs[scope][id(row)]]
                candidate_coverage = sum(float(row["mass"]) for row in candidate_evidence) / total_mass
                if (
                    len(candidate_evidence) >= min_evidence_rows
                    and candidate_share >= min_score_share
                    and candidate_coverage >= min_mass_coverage
                ):
                    module, submodule = submodule_names[submodule_id]
                    detail, score_share, evidence = "", candidate_share, candidate_evidence
                    confidence = "high" if candidate_share >= .75 and candidate_coverage >= .35 else "medium"
                    source = "validated_taxonomy_submodule"
                    break
        if cluster != -1 and module == "Chưa phân loại":
            # Several submodules of one module (autopay, linked cards and
            # payment settings together) name the module, never a guessed
            # submodule inside it.
            module_scores: Counter[str] = Counter()
            for submodule_id, score in cluster_sub_scores["focus"][(platform, cluster)].items():
                module_scores[submodule_id[:3]] += score
            if module_scores:
                module_id, score = module_scores.most_common(1)[0]
                candidate_share = score / (sum(module_scores.values()) or 1.0)
                candidate_evidence = [
                    row for row in ngram_rows
                    if any(submodule_id[:3] == module_id for submodule_id in row_subs["focus"][id(row)])
                ]
                candidate_coverage = sum(float(row["mass"]) for row in candidate_evidence) / total_mass
                if (
                    len(candidate_evidence) >= min_evidence_rows
                    and candidate_share >= min_score_share
                    and candidate_coverage >= min_mass_coverage
                ):
                    module = module_names[module_id]
                    submodule, detail, score_share, evidence = "", "", candidate_share, candidate_evidence
                    confidence, source = "medium", "validated_taxonomy_module"
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


def preserve_reviewed_names(mappings: list[dict[str, object]], paths: list[Path]) -> int:
    """Keep every name a person set in the labeling app over the rule's name.

    The labeling app marks saved clusters with a ``business_review*``
    naming_source. Their rule evidence columns stay for comparison.
    """
    reviewed: dict[tuple[str, int], dict[str, str]] = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line, row in enumerate(csv.DictReader(handle), 2):
                if str(row.get("naming_source", "")).startswith("business_review"):
                    key = (row["platform"].strip(), parse_cluster_id(row["cluster_id"], f"{path}:{line}"))
                    reviewed[key] = row
    preserved = 0
    for mapping in mappings:
        row = reviewed.get((str(mapping["platform"]), int(mapping["cluster_id"])))
        if row is None:
            continue
        for column in (
            "taxonomy_id", "cluster_name", "business_family", "business_submodule",
            "business_detail", "naming_confidence", "naming_source", "needs_review",
        ):
            mapping[column] = row.get(column, "")
        mapping["module"] = row.get("business_family", "")
        mapping["submodule"] = row.get("business_submodule", "")
        mapping["detail"] = row.get("business_detail", "")
        preserved += 1
    return preserved


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
    parser.add_argument(
        "--preserve-names", type=Path, action="append", default=[],
        help="named_clusters CSV exported from the labeling app; clusters a person "
        "named (naming_source business_review*) keep that name (repeatable)",
    )
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
    preserved = preserve_reviewed_names(mappings, [path.resolve() for path in args.preserve_names])
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
        "preserved_business_review_clusters": preserved,
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
