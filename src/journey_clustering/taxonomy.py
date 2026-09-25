"""Controlled vocabulary for the semantic enrichment stage.

This module is *data*, not logic: it is the single place where somebody who
knows the product - not the pipeline - can change how a screen path is read.
`semantics.py` holds the matching algorithm and imports everything it knows
about the business from here.

Four normalisation tables, applied in this order: => Normalization Steps for words from canonization -> checkword -> phrase -> infrastructure word:

    DYNAMIC_SEGMENT_PATTERNS  path segments that are instance identifiers
    WORD_ALIASES              spelling / abbreviation -> canonical word(s)
    PHRASE_ALIASES            multi-word rewrites (``wi fi`` -> ``wifi``)
    INFRASTRUCTURE_WORDS      words carrying no business meaning at all

and three taxonomies matched against the normalised word stream:

    FAMILY_MODULES     (family, module) - *where in the product* the user is
    OBJECT_PHRASES     business object  - *what* the interaction is about
    OPERATION_PHRASES  operation        - *what the user did to it*

Every phrase is written as an underscore-joined string and split into words on
import, so ``"management_device"`` matches the Android path segment
``management_device`` *and* the iOS screen ``ManagementDeviceVC`` - the two
platforms disagree on spelling, never on meaning. `semantics` validates the
tables at import time: a phrase that collides with another entry in the same
taxonomy, that contains an infrastructure word, or that is not already in
alias-canonical form raises immediately, so a broken taxonomy fails loudly
instead of silently mislabelling half the corpus.
"""

from __future__ import annotations

import re

UNKNOWN = "unknown"
# Module used when a family is identified but no sub-area is. It is a real
# label, not a missing value: "the user is in payment, unspecified area".
GENERAL = "general"


# --------------------------------------------------------------------------
# Normalisation tables
# --------------------------------------------------------------------------
# A path segment matching any of these is an instance identifier, not a name.
# `{id}` / `{uuid}` / `{code}` are produced upstream by `canonize._mask_segment`.
DYNAMIC_SEGMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\{[a-z]+\}$"),          # {id} {uuid} {code}
    re.compile(r"^-?\d+$"),               # 0  1  -1  1671
    re.compile(r"^[0-9a-f]{8,}$", re.I),  # hex ids
    re.compile(r"^v\d+$", re.I),          # v920 - build/version suffix
    re.compile(r"^(null|none|nil|undefined|nan)$", re.I),
)

# Spelling variants, abbreviations and product code names. The value is a tuple
# of canonical words, so one abbreviation may expand to several. Identity
# entries are pointless and forbidden; so is an entry whose output is itself an
# alias key (that would make normalisation order-dependent).
WORD_ALIASES: dict[str, tuple[str, ...]] = {
    # the same product, spelled differently by the two apps
    "fprotect": ("fsafe",),
    "protect": ("fsafe",),
    "fs": ("fsafe",),
    "pc": ("fsafe",),           # PCContentFilterVC = parental control
    "noti": ("notification",),
    "notifications": ("notification",),
    "infor": ("info",),
    "information": ("info",),
    # plurals and inflections
    "details": ("detail",),
    "devices": ("device",),
    "profiles": ("profile",),
    "contracts": ("contract",),
    "vouchers": ("voucher",),
    "cards": ("card",),
    "modems": ("modem",),
    "locking": ("lock",),
    "blocking": ("block",),
    "blocked": ("block",),
    "searching": ("search",),
    # typos that ship in production event names
    "payement": ("payment",),
    "loylaty": ("loyalty",),
    "succesful": ("success",),
    "chossen": ("chosen",),
    # abbreviations and code names
    "ap": ("access", "point"),
    "dkol": ("sale", "online"),
    "csat": ("survey",),
    "uf": ("ultra", "fast"),
    "cccd": ("vneid",),
    "napas": ("card",),
    "vietqr": ("qr",),
    "adsview": ("ads", "view"),
    "dr": ("doctor",),
    "loy": ("loyalty",),
}

# Rewrites applied to the word stream after WORD_ALIASES, for names the two
# platforms split differently (`WiFi` -> `wi`,`fi` on iOS, `wifi` on Android)
# and for multi-word idioms that mean one thing.
PHRASE_ALIASES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("wi", "fi"): ("wifi",),
    ("e", "contract"): ("econtract",),
    ("e", "counter"): ("ecounter",),
    ("e", "bill"): ("bill",),
    ("access", "point"): ("modem",),
    ("go", "to"): ("goto",),
    ("pull", "to", "refresh"): ("refresh",),
    ("turn", "on", "off"): ("toggle",),
    ("do", "action"): ("doaction",),
    ("un", "lock"): ("unlock",),
    ("pre", "paid"): ("prepaid",),
    ("post", "paid"): ("postpaid",),
}

# Words that describe the delivery mechanism, not the business. They are
# removed from the stream before matching, so no taxonomy phrase may use one.
INFRASTRUCTURE_WORDS: frozenset[str] = frozenset(
    {
        # platform / host / transport
        "android", "ios", "app", "hi", "fpt", "vn", "com", "www", "uri",
        "http", "https", "url", "web", "webview", "webpage", "webkit", "api", "sm",
        # UI framework containers and their vendor prefixes
        "vc", "controller", "activity", "fragment", "screen", "page", "host",
        "hosting", "ui", "uikit", "sdk", "base", "main", "st", "sf", "platform",
        "introspection", "anchor", "remote", "browser", "safari", "authentication",
        "navigation", "tabbar", "keyboard", "overlay", "toast", "loading",
        "dexter", "redirect", "receiver", "tracking", "element", "system",
        "alternate", "application", "icons", "with", "inui", "voice",
        "shortcut", "hidden", "input", "basic", "pdf", "non", "interaction",
        # generic structural nouns that appear in every other event path
        "header", "footer", "body", "item", "icon", "bottom", "sheet", "button",
        "custom", "size", "big", "sa", "os", "data", "default", "action",
        "btn", "v2", "hifpt", "ftel",
    }
)


# --------------------------------------------------------------------------
# Taxonomy - where in the product the user is
# --------------------------------------------------------------------------
# family -> module -> phrases. `GENERAL` holds phrases that identify the family
# without pinning a sub-area; a phrase under any other module is "specific" and
# outranks a GENERAL phrase during resolution (see `semantics.resolve_section`).
FAMILY_MODULES: dict[str, dict[str, tuple[str, ...]]] = {
    "internet": {
        GENERAL: (
            "internet", "service", "internet_service", "service_management",
            "service_manage", "manage_service", "network",
        ),
        "modem": ("modem", "router", "wifi_router", "network_model", "rename_box", "manage_box"),
        "wifi": ("wifi", "ssid", "coverage", "band"),
        "device": (
            "device", "connected_device", "management_device", "device_control",
            "device_manager", "device_detection", "history_access", "poor_connect",
        ),
        "parental_control": (
            "fsafe", "f_safe", "parental", "website", "threat",
            "internet_break", "harmful_content", "block_content",
            "content_filter", "safe_search", "trusted_device",
        ),
        "diagnostics": (
            "speedtest", "network_chart", "visualize_network", "doctor_smart",
            "health_net", "check_game", "evaluate_coverage",
        ),
        "schedule": (
            "block_schedule", "modem_schedule", "wifi_schedule", "schedule_wifi",
            "time_setting", "access_block", "restart_schedule",
        ),
    },
    "payment": {
        GENERAL: ("payment", "pay", "bill", "billing", "invoice", "cash_in", "pay_later", "wallet"),
        "prepaid": ("prepaid", "postpaid"),
        "autopay": ("autopay", "auto_pay", "payment_schedule", "schedule_payment", "list_auto_pay"),
        "method": ("payment_method", "my_card", "card", "qr_payment", "payment_qr", "paylocator"),
        "history": ("payment_history", "history_payment", "transaction_history", "payment_extension"),
        "checkout": (
            "checkout", "confirm_payment", "payment_info", "info_payment",
            "offer_payment", "payment_result", "re_payment",
        ),
        "behalf": ("behalf", "on_behalf"),
        "gold": ("fgold", "f_gold"),
    },
    "account": {
        GENERAL: ("account", "personal", "profile", "manage_account", "auth_account"),
        "contract": ("contract", "choose_contract", "contract_management", "manage_contract", "no_contract"),
        "econtract": ("econtract", "quote_minutes", "acceptance", "signed_success"),
        "info_change": (
            "ecounter", "change_info", "info_change", "change_address", "change_phone",
            "change_charge_address", "change_invoice_address", "kyc", "ocr",
            "read_card", "vneid", "scan_document",
        ),
        "settings": ("account_setting", "setting_account", "policy", "manage_info"),
        "permission": ("decentralize", "authorization", "assign_user", "verify_employee"),
    },
    "support": {
        GENERAL: ("support", "help"),
        "request": (
            "support_create", "create_request", "request_status", "support_request",
            "list_support", "select_type_support", "select_content_support",
        ),
        "chat": ("chatbot", "chat", "support_chat"),
        "report": ("report", "feedback", "question_report", "filter_report", "choose_time_report"),
        "survey": ("survey", "rating", "form_survey"),
    },
    "notification": {
        GENERAL: ("notification",),
        "settings": (
            "setting_notification", "notification_setting", "notification_channel",
            "popup_remind", "remind_notification",
        ),
    },
    "loyalty": {
        GENERAL: ("loyalty", "member", "point"),
        "voucher": ("voucher", "gift", "giftcode", "redeem"),
        "game": ("game", "checkin", "tarot", "shaking", "shake", "fortune", "ultra_fast"),
        "promotion": ("promotion", "promo"),
        "referral": ("refer", "referral", "refer_friends", "my_qr"),
    },
    "shop": {
        GENERAL: ("shop", "ecommerce", "store"),
        "catalog": ("product", "product_management", "product_detail", "category"),
        "order": ("order", "order_history", "order_info", "order_detail", "cart", "address_book", "vat_info"),
        "sale": ("sale", "sale_online", "register_info", "select_contract"),
    },
    "auth": {
        GENERAL: ("login", "logout", "sign_in", "sign_out", "guest", "otp", "explore"),
    },
    "camera": {
        GENERAL: ("camera",),
    },
    "tv": {
        GENERAL: ("tv",),
    },
    "marketing": {
        "ads": ("ads", "banner", "mascot"),
        "message": ("short_msg", "full_msg", "msg", "message"),
    },
}

# `home` is the app hub. It prefixes almost every Android path
# (`android/home/payment/...`), so it may only decide the family when *nothing*
# else in the path does - otherwise every screen in the app would be "home".
WEAK_FAMILY_MODULES: dict[str, dict[str, tuple[str, ...]]] = {
    "home": {
        GENERAL: ("home",),
        "navigation": ("nav",),
        "favourite": ("fav", "favorite", "favourite", "favorite_feature", "fav_function"),
        "customize": ("customize", "customize_function", "change_logo"),
    },
}

# Navigation containers, popups and app-launch screens: emitted by the
# framework, never chosen by the user. `chrome` is a real answer, not a guess -
# "the user was in UI plumbing" is exactly what happened - but it is consulted
# last, so a popup *inside* the payment flow stays payment. A path whose words
# are all stripped, or all bare verbs, also lands in `chrome/container`.
FALLBACK_FAMILY_MODULES: dict[str, dict[str, tuple[str, ...]]] = {
    "chrome": {
        "container": ("tab_bar", "alert", "scene", "window", "password_saving"),
        "popup": ("popup",),
        "boot": ("splash", "launch", "welcome"),
    },
}
CONTAINER_SECTION: tuple[str, str] = ("chrome", "container")


# --------------------------------------------------------------------------
# Taxonomy - what the interaction is about
# --------------------------------------------------------------------------
OBJECT_PHRASES: dict[str, tuple[str, ...]] = {
    "device": ("device", "connected_device", "management_device"),
    "modem": ("modem", "router", "box", "network_model"),
    "wifi": ("wifi", "ssid", "band"),
    "profile": ("profile",),
    "contract": ("contract",),
    "econtract": ("econtract", "quote_minutes"),
    "bill": ("bill", "billing", "invoice"),
    "payment": ("payment", "checkout"),
    "card": ("card",),
    "voucher": ("voucher", "gift", "giftcode"),
    "game": ("game", "tarot", "checkin"),
    "promotion": ("promotion", "promo"),
    "notification": ("notification",),
    "schedule": ("schedule", "time_setting"),
    "request": ("request", "ticket"),
    "report": ("report", "feedback"),
    "survey": ("survey", "rating"),
    "website": ("website", "threat", "harmful_content", "content"),
    "product": ("product",),
    "order": ("order", "cart"),
    "account": ("account", "personal"),
    "otp": ("otp",),
    "password": ("password", "passwd"),
    "name": ("name",),
    "address": ("address",),
    "phone": ("phone",),
    "email": ("email",),
    "camera": ("camera",),
    "tv": ("tv",),
    "package": ("package", "prepaid", "postpaid"),
    "network": ("network",),
    "message": ("msg", "message", "chat"),
    "popup": ("popup", "banner", "ads", "mascot"),
    "qr": ("qr",),
    "history": ("history",),
}


# --------------------------------------------------------------------------
# Taxonomy - what the user did
# --------------------------------------------------------------------------
OPERATION_PHRASES: dict[str, tuple[str, ...]] = {
    "open": ("click", "open", "enter", "goto", "nav", "show", "doaction", "tap", "popup"),
    "view": ("view", "detail", "info", "read"),
    "list": ("list", "all", "tab_all"),
    "select": ("select", "choose", "chosen", "pick", "assign", "check"),
    "search": ("search",),
    "filter": ("filter", "tab_connected", "tab_block", "sort"),
    "refresh": ("refresh", "reload", "sync"),
    "create": ("create", "add", "new", "register"),
    "update": ("update", "edit", "change", "set", "setting", "config", "configure"),
    "rename": ("rename", "change_name", "set_name"),
    "delete": ("delete", "remove"),
    "block": ("block", "lock", "restrict"),
    "unblock": ("unblock", "unlock", "allow", "trust"),
    "enable": ("turn_on", "activate", "enable"),
    "disable": ("turn_off", "deactivate", "disable"),
    "toggle": ("toggle", "switch"),
    "restart": ("reboot", "restart", "reset"),
    "schedule": ("schedule",),
    "confirm": ("confirm", "accept", "agree", "verify", "ok"),
    "submit": ("submit", "send", "save", "apply", "upload"),
    "pay": ("pay", "checkout", "purchase", "buy"),
    "cancel": ("cancel",),
    "back": ("back", "backbutton", "handleback", "goback"),
    "close": ("close", "dismiss", "skip", "exit"),
    "login": ("login", "sign_in"),
    "logout": ("logout", "log_out", "sign_out"),
    "share": ("share",),
    "scan": ("scan", "speedtest", "health_net"),
    "download": ("download", "export"),
    "rate": ("rate", "rating", "vote"),
}

# Coarse funnel phase for each operation. This is its own clustering channel:
# the shape of a journey in these six symbols is what separates "browsed and
# left" from "configured and committed".
OPERATION_STAGES: dict[str, str] = {
    "open": "entry",
    "view": "entry",
    "list": "browse",
    "select": "browse",
    "search": "browse",
    "filter": "browse",
    "refresh": "browse",
    "create": "configure",
    "update": "configure",
    "rename": "configure",
    "schedule": "configure",
    "enable": "configure",
    "disable": "configure",
    "toggle": "configure",
    "confirm": "commit",
    "submit": "commit",
    "pay": "commit",
    "delete": "commit",
    "block": "commit",
    "unblock": "commit",
    "restart": "commit",
    "login": "commit",
    "logout": "commit",
    "share": "commit",
    "scan": "commit",
    "download": "commit",
    "rate": "commit",
    "back": "abort",
    "close": "abort",
    "cancel": "abort",
}

# Action targets that describe the *gesture*, not the destination: they carry an
# operation but no business meaning of their own, so family, module and object
# must be inherited from the screen the user was on. `btn_back` on the modem
# screen is "left the modem screen", not a fifth kind of event. Written in
# normalised form - `btn_back` reaches the matcher as ``("back",)``.
GENERIC_ACTION_PHRASES: frozenset[tuple[str, ...]] = frozenset(
    {
        ("back",),
        ("backbutton",),
        ("handleback",),
        ("goback",),
        ("popup",),
        ("confirm",),
        ("cancel",),
        ("close",),
        ("dismiss",),
        ("skip",),
        ("ok",),
        ("submit",),
        ("send",),
        ("save",),
        ("next",),
        ("continue",),
        ("refresh",),
        ("reload",),
        ("select",),
        ("view",),
    }
)
