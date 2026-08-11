"""Build a readable, evidence-backed cluster catalog for external reporting.

The cluster ids are specific to the current production run. Names are inferred
from the current medoid path, entry/exit tokens, and the highest-ranked n-grams
for each cluster; the source evidence is preserved in the resulting JSON.

Usage:
    python scripts/build_shareholder_cluster_catalog.py
"""

from __future__ import annotations

import csv
import copy
import json
import re
import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPO_ROOT / "output" / "journey_runs" / "EXACT_ch-c45i30o20_ng1-3_svd64_fdf3_mf20000_nw0p35_mcs100_ms5_sel-eom_gap90_jmin4_tdf3_ent0_chr1_boot0_test0p2"
OUTPUT_PATH = CLUSTER_DIR / "shareholder_cluster_catalog.json"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value: Any) -> int:
    return int(float(value))


def as_float(value: Any) -> float:
    return float(value)


def clean_ngram(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("<bos>", "START").replace("<eos>", "END")).strip()


def token_text(token: str) -> str:
    return token.split("@", 1)[-1].lower()


def signal_text(record: dict[str, Any], ngrams: list[dict[str, str]]) -> str:
    evidence = [
        str(record.get("medoid_path", "")),
        str(record.get("top_entry_token", "")),
        str(record.get("top_exit_token", "")),
    ]
    evidence.extend(row.get("ngram", "") for row in ngrams)
    return " ".join(evidence).lower()


def ngram_signal_text(ngrams: list[dict[str, str]]) -> str:
    """Return the displayed top-three n-grams used for subtype names.

    Lower-ranked evidence is still retained in the source CSV, but must not
    create a specific public label that the top evidence does not support.
    """
    ordered = top_ngram_rows(ngrams)
    return " ".join(str(row.get("ngram", "")) for row in ordered).lower()


def top_ngram_rows(ngrams: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the same top-three evidence rows used for public labels."""

    return sorted(ngrams, key=lambda row: as_int(row.get("rank", 0)))[:3]


def medoid_tokens(path: str) -> list[str]:
    return [token.strip().lower() for token in path.split("->") if token.strip()]


def token_is_view(token: str) -> bool:
    return token.startswith("view@")


def home_view_count(tokens: list[str]) -> int:
    generic_home_views = (
        "view@home",
        "view@android/home",
        "view@homevc",
        "view@maintabbarcontroller",
        "view@mainappactivity",
        "view@splashactivity",
        "view@splash",
    )
    return sum(token in generic_home_views for token in tokens)


def marker_token_counts(tokens: list[str], markers: tuple[str, ...]) -> tuple[int, int, int]:
    """Count all, view, and action marker hits in the medoid.

    A marker in one terminal action is weaker than a marker in a view or a
    repeated marker. This prevents paths such as Home -> Home -> Pay from
    being reported as payment journeys when the cluster is really a home
    landing/browse pattern.
    """

    all_hits = [token for token in tokens if any(marker in token for marker in markers)]
    view_hits = [token for token in all_hits if token_is_view(token)]
    action_hits = [token for token in all_hits if token.startswith("action@")]
    return len(all_hits), len(view_hits), len(action_hits)


def marker_ngram_row_count(ngrams: list[dict[str, str]], markers: tuple[str, ...]) -> int:
    return sum(
        any(marker in str(row.get("ngram", "")).lower() for marker in markers)
        for row in top_ngram_rows(ngrams)
    )


def dominant_signal_score(
    tokens: list[str],
    ngrams: list[dict[str, str]],
    markers: tuple[str, ...],
) -> float:
    """Score whether a theme dominates both the medoid and ranked n-grams.

    Views receive more weight than actions because they represent the surface
    actually occupied by the journey. Each matching top-three n-gram adds a
    rank-discounted vote, preventing an incidental early action from beating a
    repeated journey theme.
    """

    all_count, view_count, action_count = marker_token_counts(tokens, markers)
    score = (view_count * 2.0) + action_count + max(0, all_count - view_count - action_count)
    for row in top_ngram_rows(ngrams):
        row_text = str(row.get("ngram", "")).lower()
        if any(marker in row_text for marker in markers):
            rank = max(1, as_int(row.get("rank", 1)))
            score += max(1.0, 4.0 - rank) * 2.0
    return score


def is_incidental_tail_signal(
    tokens: list[str],
    ngrams: list[dict[str, str]],
    markers: tuple[str, ...],
) -> bool:
    """Identify a business marker that is only a weak terminal event.

    The signal is suppressed only when the medoid is visibly home-dominant,
    has no matching view, and has at most one matching action supported by at
    most one of the displayed n-grams. A real payment journey with a payment
    screen or repeated payment n-grams remains eligible for a payment label.
    """

    all_count, view_count, action_count = marker_token_counts(tokens, markers)
    return (
        home_view_count(tokens) >= 2
        and all_count == 1
        and view_count == 0
        and action_count <= 1
        and marker_ngram_row_count(ngrams, markers) <= 1
    )


def is_single_action_only_signal(
    tokens: list[str],
    ngrams: list[dict[str, str]],
    markers: tuple[str, ...],
) -> bool:
    """Return whether a business intent is supported by one action only.

    A single navigation action can describe where the user tapped without
    proving that the journey reached or completed that destination.  Do not
    name a whole cluster after that action unless a matching destination view
    is present either in the medoid or in the displayed top n-grams.
    """

    unique_actions = {token for token in tokens if token.startswith("action@")}
    matching_actions = {
        token for token in unique_actions if any(marker in token for marker in markers)
    }
    matching_views = {
        token
        for token in tokens
        if token_is_view(token) and any(marker in token for marker in markers)
    }
    ngram_has_matching_view = any(
        any(
            token.startswith("view@") and any(marker in token for marker in markers)
            for token in str(row.get("ngram", "")).lower().split()
        )
        for row in top_ngram_rows(ngrams)
    )
    return (
        len(unique_actions) == 1
        and len(matching_actions) == 1
        and not matching_views
        and not ngram_has_matching_view
    )


def weighted_signal_score(
    primary_text: str,
    ngrams: list[dict[str, str]],
    markers: tuple[str, ...],
) -> float:
    """Score a business signal using the medoid and displayed top n-grams.

    The medoid contributes a small prior. Ranked n-grams contribute their
    cluster mass discounted by rank, so a frequent login pattern beats one
    incidental payment transition in a mixed cluster.
    """
    primary_hits = sum(primary_text.count(marker) for marker in markers)
    score = min(3.0, primary_hits * 0.75)
    ordered = top_ngram_rows(ngrams)
    for row in ordered:
        text = str(row.get("ngram", "")).lower()
        if not any(marker in text for marker in markers):
            continue
        rank = max(1, as_int(row.get("rank", 1)))
        mass = max(0.1, as_float(row.get("cluster_mass", 1.0)))
        score += mass / rank
    return score


def has_econtract_marker(value: str) -> bool:
    """Recognize EContract screens without mistaking ChooseContract for them."""

    return bool(re.search(r"(?<![A-Za-z])[eE][-\_]?contract(?:[A-Z]|\b)", value)) or any(
        term in value.lower() for term in ("fe_contract", "contract_confirm")
    )


def classify(record: dict[str, Any], ngrams: list[dict[str, str]]) -> tuple[str, str, list[str], str]:
    """Return family, readable name, evidence signals, and confidence."""

    cluster_id = as_int(record["cluster"])
    if cluster_id == -1 or record.get("label_kind") == "noise":
        return (
            "unknown",
            "Unclassified / mixed journeys",
            ["noise cluster"],
            "not_applicable",
        )

    # The medoid is useful for the broad business family. Specific subtype names
    # must also be present in ranked cluster n-grams; otherwise a single medoid
    # can overfit to an incidental screen (for example, calling a cluster
    # "VietQR payment" when the cluster's discriminative n-grams never contain
    # VietQR).
    raw_primary = " ".join(
        [
            str(record.get("medoid_path", "")),
            str(record.get("top_entry_token", "")),
            str(record.get("top_exit_token", "")),
        ]
    )
    text = raw_primary.lower()
    tokens = medoid_tokens(str(record.get("medoid_path", "")))
    ngram_text = ngram_signal_text(ngrams)
    payment_markers = ("payment", "payaction", "pay_action", "prepaid", "bill", "checkout", "vietqr")
    auth_markers = ("login", "oauth", "authorization", "authentication", "otp", "sfauthentication", "guest_login")
    wifi_markers = ("wifi",)
    notification_markers = ("notification", "noti", "view_os_noti")
    payment_score = weighted_signal_score(text, ngrams, payment_markers)
    auth_score = weighted_signal_score(text, ngrams, auth_markers)
    wifi_dominance = dominant_signal_score(tokens, ngrams, wifi_markers)
    notification_dominance = dominant_signal_score(tokens, ngrams, notification_markers)
    suppressed_single_actions: list[str] = []

    def allow_action_signal(markers: tuple[str, ...], label: str) -> bool:
        if is_single_action_only_signal(tokens, ngrams, markers):
            suppressed_single_actions.append(label)
            return False
        return True

    def in_ngrams(*terms: str) -> bool:
        return any(term in ngram_text for term in terms)

    def in_primary(*terms: str) -> bool:
        return any(term in text for term in terms)

    def in_evidence(*terms: str) -> bool:
        return any(term in text or term in ngram_text for term in terms)

    # Rules are ordered from specific feature journeys to broader surfaces.
    if ("modem_reset" in text or "mode_reset" in text) and allow_action_signal(
        ("modem_reset", "mode_reset"), "modem reset"
    ):
        return "device_management", "Router reset / modem reboot", ["modem reset"], "high"

    if in_primary("scanqr", "scan_qr", "scan qr") and allow_action_signal(
        ("scanqr", "scan_qr", "scan qr"), "QR scan"
    ):
        return "utility", "QR scanning", ["QR scan"], "high"

    if any(term in text for term in ("csat", "rating", "survey")) and allow_action_signal(
        ("csat", "rating", "survey"), "CSAT or survey"
    ):
        return "feedback", "Customer feedback / survey", ["CSAT or survey"], "high"

    has_econtract = has_econtract_marker(raw_primary)
    if has_econtract and in_evidence("schedule_payment", "payment") and in_evidence("profile", "account"):
        return "contracts", "Profile, payment reminders and e-contract", ["profile + payment reminder + e-contract"], "high"

    if has_econtract:
        if in_ngrams("sign", "confirm", "vneid", "cccd", "successsign"):
            name = "E-contract signing / confirmation"
        else:
            name = "E-contract review"
        return "contracts", name, ["e-contract"], "high"

    support_markers = ("support", "chatbot", "supportcreator", "support_request")
    if any(term in text for term in support_markers) and allow_action_signal(
        support_markers, "support"
    ):
        if any(term in text for term in ("send", "confirm", "submit")):
            name = "Support request submission"
        elif "chat" in text or "chatbot" in text:
            name = "Support chat and request"
        else:
            name = "Customer support journey"
        return "support", name, ["support journey"], "high"

    if any(term in text for term in ("networkchart", "networkchart", "network_model", "networkmodel", "manageap", "view_ap")):
        return "device_management", "Network performance and access-point management", ["network / access point controls"], "high"

    if any(term in text for term in ("fprotect", "f-safe", "fsafe", "connecteddevice", "connected_device", "device_control", "device_detail", "block_harmful", "access_block")):
        return "device_management", "Connected-device protection and controls", ["safe internet / connected devices"], "high"

    # A few setup actions at the beginning of a medoid must not override the
    # surface that occupies most of the journey. Require Notification to be
    # present repeatedly in both the medoid and at least two top n-grams before
    # it can outrank a Wi-Fi action seen earlier in the path.
    notification_token_count, _, _ = marker_token_counts(tokens, notification_markers)
    if (
        notification_token_count >= 2
        and marker_ngram_row_count(ngrams, notification_markers) >= 2
        and notification_dominance > wifi_dominance
    ):
        return (
            "engagement",
            "Notifications and alerts",
            ["notification dominates medoid and top n-grams"],
            "high",
        )

    if "wifi" in text and allow_action_signal(("wifi",), "Wi-Fi"):
        if "schedule" in text:
            name = "Wi-Fi scheduling"
        elif "guest" in text:
            name = "Guest Wi-Fi settings"
        else:
            band = "5 GHz" if "5ghz" in text or "5ghz" in text else "2 GHz" if "2ghz" in text else ""
            state = "on/off" if "turn_on" in text and "turn_off" in text else "on" if "turn_on" in text else "off" if "turn_off" in text else "settings"
            name = f"Wi-Fi control — {band + ' ' if band else ''}{state}".strip()
        return "device_management", name, ["Wi-Fi controls"], "high"

    if ("modem" in text or "managemodem" in text) and allow_action_signal(
        ("modem", "managemodem"), "modem"
    ):
        return "device_management", "Modem management", ["modem controls"], "high"

    # The business family must be visible in the representative evidence
    # (medoid + entry/exit). Ranked n-grams may refine the subtype, but must
    # not relabel an account journey as payment merely because one n-gram
    # contains a payment reminder.
    # One terminal action such as Action@Pay must not override repeated Home
    # views. The n-grams form a second gate, so genuine payment journeys with
    # a payment screen or repeated payment evidence still receive a payment
    # label.
    payment_tail_only = is_incidental_tail_signal(tokens, ngrams, payment_markers)
    primary_payment = (
        in_primary(*payment_markers)
        and not payment_tail_only
        and allow_action_signal(payment_markers, "payment")
    )
    primary_auth = in_primary(*auth_markers) and allow_action_signal(
        auth_markers, "authentication"
    )
    if primary_payment and payment_score >= auth_score:
        if in_ngrams("vietqr"):
            name = "VietQR payment"
        elif in_ngrams("prepaid"):
            name = "Prepaid payment"
        elif in_ngrams("checkout", "payaction", "pay_action"):
            name = "Bill payment and checkout"
        else:
            name = "Payments and billing"
        return "payments", name, ["payment or billing"], "high"

    if any(term in text for term in ("logout", "log_out")) and allow_action_signal(
        ("logout", "log_out"), "logout"
    ):
        return "authentication", "Sign-out / login reset", ["logout"], "high"

    if primary_auth and auth_score > 0.0:
        if in_ngrams("oauth", "authorization"):
            name = "External / OAuth login"
        elif in_ngrams("otp"):
            name = "OTP login"
        elif in_ngrams("guest"):
            name = "Guest login"
        else:
            name = "Login and authentication"
        return "authentication", name, ["authentication"], "high"

    commerce_markers = ("shop", "product", "order", "ecommerce", "e-commerce", "product-detail")
    if any(term in text for term in commerce_markers) and allow_action_signal(
        commerce_markers, "shop / product / order"
    ):
        if "order" in text or "checkout" in text:
            name = "Shop and order journey"
        else:
            name = "Shop browsing"
        return "commerce", name, ["shop / product / order"], "high"

    engagement_markers = ("loyalty", "promotion", "adsview", "ads_view", "game-checkin")
    if any(term in text for term in engagement_markers) and allow_action_signal(
        engagement_markers, "promotion / loyalty / ads"
    ):
        return "engagement", "Promotions, loyalty and ad exposure", ["promotion / loyalty / ads"], "high"

    if any(term in text for term in notification_markers) and allow_action_signal(
        notification_markers, "notification"
    ):
        return "engagement", "Notifications and alerts", ["notifications"], "high"

    contract_markers = ("contract", "choosecontract", "choose_contract", "managercontract", "change_contract")
    if any(term in text for term in contract_markers) and allow_action_signal(
        contract_markers, "contract"
    ):
        if "share" in text or "permission" in text:
            name = "Contract sharing and permissions"
        elif "choose" in text or "change_contract" in text or "contract_list" in text:
            name = "Contract selection and switching"
        else:
            name = "Contract management"
        return "contracts", name, ["contract management"], "high"

    account_markers = ("profile", "account", "personal", "nav_profile")
    if any(term in text for term in account_markers) and allow_action_signal(
        account_markers, "profile / account"
    ):
        return "account", "Profile and account journey", ["profile / account"], "high"

    service_markers = ("update-package", "update_package", "package", "service_manage", "servicemanage", "other_manage_internet", "internet_service_management")
    if any(term in text for term in service_markers) and allow_action_signal(
        service_markers, "service management"
    ):
        if "package" in text or "update-package" in text:
            name = "Service package browsing and upgrade"
        else:
            name = "Internet service management"
        return "service_management", name, ["internet service management"], "high"

    popup_markers = ("popup", "remind", "invite")
    if any(term in text for term in popup_markers) and allow_action_signal(
        popup_markers, "popup / interruption"
    ):
        return "engagement", "Popup / interruption handling", ["popup or invite"], "medium"

    if any(term in text for term in ("mainappactivity", "maintabbarcontroller", "splashactivity", "splashvc", "homevc", "view@home", "view@homevc")):
        signals = ["app launch / home"]
        if payment_tail_only:
            signals.append("terminal payment-like action suppressed")
        if suppressed_single_actions:
            signals.append(
                "single unique action signal suppressed: "
                + ", ".join(dict.fromkeys(suppressed_single_actions))
            )
        return "navigation", "App launch and home browsing", signals, "medium"

    signals = ["no dominant business signal"]
    if suppressed_single_actions:
        signals.append(
            "single unique action signal suppressed: "
            + ", ".join(dict.fromkeys(suppressed_single_actions))
        )
    return "other", "Other app journey", signals, "medium"


def readable_token(token: str) -> str:
    raw = token.strip()
    if not raw:
        return ""
    lower = raw.lower()
    body = token_text(raw)

    if lower in {"<rare>", "rare"}:
        return "Other / rare event"
    if "modem_reset" in lower:
        return "Reset modem"
    if "turn_on_wifi_5ghz" in lower:
        return "Turn on 5 GHz Wi-Fi"
    if "turn_off_wifi_5ghz" in lower:
        return "Turn off 5 GHz Wi-Fi"
    if "turn_on_wifi_2ghz" in lower:
        return "Turn on 2 GHz Wi-Fi"
    if "turn_off_wifi_2ghz" in lower:
        return "Turn off 2 GHz Wi-Fi"
    if "wifi_guest" in lower:
        return "Manage guest Wi-Fi"
    if "wifi" in lower and "schedule" in lower:
        return "Manage Wi-Fi schedule"
    if "wifi" in lower:
        return "Open Wi-Fi controls"
    if "modem" in lower or "managemodem" in lower:
        return "Open modem controls"
    if "fprotect" in lower or "fsafe" in lower or "connecteddevice" in lower or "device_control" in lower:
        return "Manage protected / connected devices"
    if "network" in lower or "manageap" in lower or "view_ap" in lower:
        return "View network performance"
    if "scanqr" in lower or "scan_qr" in lower:
        return "Scan QR code"
    if "support" in lower or "chatbot" in lower:
        return "Open customer support"
    if has_econtract_marker(raw):
        return "View or confirm e-contract"
    if "contract" in lower or "choosecontract" in lower or "choose_contract" in lower:
        return "Select or manage contract"
    if "payment" in lower or "payaction" in lower or "prepaid" in lower or "vietqr" in lower or "checkout" in lower:
        return "Open payment flow"
    if "notification" in lower or "noti" in lower or "view_os_noti" in lower:
        return "View notifications"
    if "logout" in lower or "log_out" in lower:
        return "Sign out"
    if "login" in lower or "oauth" in lower or "authorization" in lower or "otp" in lower:
        return "Authenticate / log in"
    if "profile" in lower or "personal" in lower or "account" in lower:
        return "Open profile or account"
    if "shop" in lower or "product" in lower or "order" in lower or "ecommerce" in lower:
        return "Browse shop or order"
    if "adsview" in lower or "promotion" in lower or "loyalty" in lower:
        return "View promotion or ad"
    if "popup" in lower or "remind" in lower or "invite" in lower:
        return "Handle popup"
    if "csat" in lower or "rating" in lower or "survey" in lower:
        return "Provide feedback"
    if "btn_back" in lower or lower.endswith("/back") or "backbutton" in lower:
        return "Go back"
    if lower in {"view@home", "home", "homevc", "view@homevc", "view@android/home"} or body in {"home", "homevc"}:
        return "Home"
    if "mainappactivity" in lower or "maintabbarcontroller" in lower or "splashactivity" in lower or "splashvc" in lower or "basenavigation" in lower:
        return "Open app navigation"

    # Keep fallback labels short while retaining enough of the original event
    # to be useful to a reader who needs to audit the raw path.
    compact = re.sub(r"\{[^}]+\}", "parameter", body)
    compact = re.sub(r"[^a-z0-9]+", " ", compact).strip()
    words = compact.split()
    label = " ".join(words[-5:]) if words else "Other event"
    return label.capitalize()


def readable_path(path: str) -> tuple[list[str], bool]:
    tokens = [token.strip() for token in path.split("->") if token.strip()]
    labels = [readable_token(token) for token in tokens]
    if len(labels) <= 16:
        return labels, False
    return labels[:12] + ["… additional steps …"] + labels[-2:], True


def evidence_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ordered = top_ngram_rows(rows)
    result: list[dict[str, Any]] = []
    for row in ordered[:3]:
        result.append(
            {
                "rank": as_int(row["rank"]),
                "ngram": clean_ngram(row["ngram"]),
                "lift": as_float(row["lift"]),
                "cluster_mass": as_float(row["cluster_mass"]),
            }
        )
    return result


def cluster_entry(record: dict[str, Any], ngrams: list[dict[str, str]], platform: str) -> dict[str, Any]:
    family, name, signals, confidence = classify(record, ngrams)
    display_steps, truncated = readable_path(str(record.get("medoid_path", "")))
    cluster_id = as_int(record["cluster"])

    return {
        "cluster_id": cluster_id,
        "cluster_name": name,
        "business_family": family,
        "naming_confidence": confidence,
        "naming_basis": {
            "signals": signals,
            "source": "medoid path + entry/exit tokens + ranked cluster n-grams",
        },
        "representative_journey": {
            "journey_id": record.get("medoid_journey_id"),
            "event_count": as_int(record["medoid_length"]),
            "readable_steps": display_steps,
            "display_path_truncated": truncated,
            "raw_path": record.get("medoid_path"),
        },
        "metadata": {
            "journey_count": as_int(record["size"]),
            "journey_share": as_float(record["share"]),
            "session_count": as_int(record["n_sessions"]),
            "device_count": as_int(record["n_devices"]),
            "median_journey_length": as_float(record["median_length"]),
            "median_duration_seconds": as_float(record["median_span_s"]),
            "mean_action_ratio": as_float(record["mean_action_ratio"]),
            "mean_back_rate": as_float(record["mean_back_rate"]),
            "mean_revisit_ratio": as_float(record["mean_revisit_ratio"]),
            "loop_journey_share": as_float(record["loop_journey_share"]),
            "top_entry_token": record.get("top_entry_token"),
            "top_exit_token": record.get("top_exit_token"),
        },
        "evidence": {
            "top_ngrams": evidence_rows(ngrams),
            "platform": platform,
        },
    }


def platform_entry(platform: str) -> dict[str, Any]:
    catalog_name = f"{platform}_cluster_catalog.json"
    ngram_name = f"{platform}_cluster_ngrams.csv"
    records = read_json(CLUSTER_DIR / catalog_name)
    ngram_rows = read_csv(CLUSTER_DIR / ngram_name)
    ngrams_by_cluster: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in ngram_rows:
        ngrams_by_cluster[as_int(row["cluster"])].append(row)

    clusters = [
        cluster_entry(record, ngrams_by_cluster[as_int(record["cluster"])], platform)
        for record in records
    ]
    clusters.sort(key=lambda item: (-item["metadata"]["journey_count"], item["cluster_id"]))
    noise = [item for item in clusters if item["cluster_id"] == -1]
    total_journeys = sum(item["metadata"]["journey_count"] for item in clusters)
    source_dir = CLUSTER_DIR.relative_to(REPO_ROOT).as_posix()

    return {
        "platform": platform,
        "source_files": {
            "cluster_catalog": f"{source_dir}/{catalog_name}",
            "cluster_ngrams": f"{source_dir}/{ngram_name}",
        },
        "summary": {
            "cluster_count": len(clusters),
            "named_cluster_count": len(clusters) - len(noise),
            "journey_count_in_catalog": total_journeys,
            "unclassified_journey_count": sum(item["metadata"]["journey_count"] for item in noise),
            "unclassified_journey_share": noise[0]["metadata"]["journey_share"] if noise else 0.0,
            "largest_clusters": [
                {
                    "cluster_id": item["cluster_id"],
                    "cluster_name": item["cluster_name"],
                    "journey_count": item["metadata"]["journey_count"],
                    "journey_share": item["metadata"]["journey_share"],
                }
                for item in clusters[:10]
            ],
        },
        "clusters": clusters,
    }


BUSINESS_FAMILY_VI = {
    "unknown": "chưa phân loại",
    "utility": "tiện ích",
    "feedback": "phản hồi",
    "contracts": "hợp đồng",
    "support": "hỗ trợ",
    "device_management": "quản lý thiết bị",
    "payments": "thanh toán",
    "authentication": "xác thực",
    "engagement": "tương tác",
    "commerce": "thương mại",
    "navigation": "điều hướng",
    "service_management": "quản lý dịch vụ",
    "account": "tài khoản",
    "other": "khác",
}


CLUSTER_NAME_VI = {
    "Unclassified / mixed journeys": "Hành trình chưa phân loại / hỗn hợp",
    "Notifications and alerts": "Thông báo và cảnh báo",
    "Profile, payment reminders and e-contract": "Hồ sơ, nhắc thanh toán và hợp đồng điện tử",
    "Prepaid payment": "Thanh toán trả trước",
    "App launch and home browsing": "Mở ứng dụng và duyệt trang chính",
    "Router reset / modem reboot": "Đặt lại bộ định tuyến / khởi động lại modem",
    "Support request submission": "Gửi yêu cầu hỗ trợ",
    "Contract selection and switching": "Chọn và chuyển đổi hợp đồng",
    "Customer feedback / survey": "Phản hồi khách hàng / khảo sát",
    "Guest login": "Đăng nhập khách",
    "Promotions, loyalty and ad exposure": "Khuyến mãi, khách hàng thân thiết và quảng cáo",
    "Connected-device protection and controls": "Bảo vệ và quản lý thiết bị kết nối",
    "VietQR payment": "Thanh toán VietQR",
    "Internet service management": "Quản lý dịch vụ Internet",
    "QR scanning": "Quét mã QR",
    "Shop browsing": "Duyệt cửa hàng",
    "Shop and order journey": "Duyệt cửa hàng và đơn hàng",
    "Popup / interruption handling": "Xử lý popup / gián đoạn",
    "Network performance and access-point management": "Hiệu năng mạng và quản lý điểm truy cập",
    "Modem management": "Quản lý modem",
    "Wi-Fi scheduling": "Lập lịch Wi-Fi",
    "Payments and billing": "Thanh toán và hóa đơn",
    "Bill payment and checkout": "Thanh toán hóa đơn và checkout",
    "External / OAuth login": "Đăng nhập ngoài ứng dụng / OAuth",
    "Login and authentication": "Đăng nhập và xác thực",
    "Support chat and request": "Chat và yêu cầu hỗ trợ",
    "Customer support journey": "Hành trình hỗ trợ khách hàng",
    "Profile and account journey": "Hành trình hồ sơ và tài khoản",
    "Service package browsing and upgrade": "Duyệt và nâng cấp gói dịch vụ",
    "E-contract signing / confirmation": "Ký / xác nhận hợp đồng điện tử",
    "E-contract review": "Xem hợp đồng điện tử",
    "Contract sharing and permissions": "Chia sẻ hợp đồng và phân quyền",
    "Contract management": "Quản lý hợp đồng",
    "Sign-out / login reset": "Đăng xuất / đặt lại đăng nhập",
    "Other app journey": "Hành trình ứng dụng khác",
    "Guest Wi-Fi settings": "Cài đặt Wi-Fi khách",
}


STEP_VI = {
    "Open app navigation": "Mở điều hướng ứng dụng",
    "Home": "Trang chính",
    "Reset modem": "Đặt lại modem",
    "Turn on 5 GHz Wi-Fi": "Bật Wi-Fi 5 GHz",
    "Turn off 5 GHz Wi-Fi": "Tắt Wi-Fi 5 GHz",
    "Turn on 2 GHz Wi-Fi": "Bật Wi-Fi 2 GHz",
    "Turn off 2 GHz Wi-Fi": "Tắt Wi-Fi 2 GHz",
    "Manage guest Wi-Fi": "Quản lý Wi-Fi khách",
    "Manage Wi-Fi schedule": "Quản lý lịch Wi-Fi",
    "Open Wi-Fi controls": "Mở điều khiển Wi-Fi",
    "Open modem controls": "Mở điều khiển modem",
    "Manage protected / connected devices": "Quản lý thiết bị được bảo vệ / thiết bị kết nối",
    "View network performance": "Xem hiệu năng mạng",
    "Scan QR code": "Quét mã QR",
    "Open customer support": "Mở hỗ trợ khách hàng",
    "View or confirm e-contract": "Xem hoặc xác nhận hợp đồng điện tử",
    "Select or manage contract": "Chọn hoặc quản lý hợp đồng",
    "Open payment flow": "Mở luồng thanh toán",
    "View notifications": "Xem thông báo",
    "Sign out": "Đăng xuất",
    "Authenticate / log in": "Xác thực / đăng nhập",
    "Open profile or account": "Mở hồ sơ hoặc tài khoản",
    "Browse shop or order": "Duyệt cửa hàng hoặc đơn hàng",
    "View promotion or ad": "Xem khuyến mãi hoặc quảng cáo",
    "Handle popup": "Xử lý popup",
    "Provide feedback": "Gửi phản hồi",
    "Go back": "Quay lại",
    "Other / rare event": "Sự kiện khác / hiếm",
    "… additional steps …": "… các bước tiếp theo …",
}


SIGNAL_VI = {
    "noise cluster": "cụm nhiễu",
    "payment or billing": "thanh toán hoặc hóa đơn",
    "profile + payment reminder + e-contract": "hồ sơ + nhắc thanh toán + hợp đồng điện tử",
    "Wi-Fi controls": "điều khiển Wi-Fi",
    "modem controls": "điều khiển modem",
    "safe internet / connected devices": "Internet an toàn / thiết bị kết nối",
    "network / access point controls": "điều khiển mạng / điểm truy cập",
    "support journey": "hành trình hỗ trợ",
    "support": "hỗ trợ",
    "notifications": "thông báo",
    "notification dominates medoid and top n-grams": "thông báo chi phối medoid và các n-gram hàng đầu",
    "authentication": "xác thực",
    "logout": "đăng xuất",
    "promotion / loyalty / ads": "khuyến mãi / khách hàng thân thiết / quảng cáo",
    "shop / product / order": "cửa hàng / sản phẩm / đơn hàng",
    "contract management": "quản lý hợp đồng",
    "QR scan": "quét mã QR",
    "CSAT or survey": "CSAT hoặc khảo sát",
    "popup or invite": "popup hoặc lời mời",
    "app launch / home": "mở ứng dụng / trang chính",
    "terminal payment-like action suppressed": "đã loại tín hiệu thanh toán chỉ xuất hiện ở action cuối",
    "no dominant business signal": "không có tín hiệu nghiệp vụ nổi trội",
}


def translate_step(step: str) -> str:
    if step in STEP_VI:
        return STEP_VI[step]
    replacements = {
        "Home": "Trang chính",
        "other": "khác",
        "manage": "quản lý",
        "internet": "Internet",
        "service": "dịch vụ",
        "view": "xem",
        "open": "mở",
        "select": "chọn",
        "contract": "hợp đồng",
        "payment": "thanh toán",
    }
    translated = step
    for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
        translated = re.sub(rf"\b{re.escape(source)}\b", target, translated, flags=re.IGNORECASE)
    return translated


def translate_cluster_name(name: str) -> str:
    if name in CLUSTER_NAME_VI:
        return CLUSTER_NAME_VI[name]
    if name.startswith("Wi-Fi control"):
        suffix = name.split("—", 1)[-1].strip()
        suffix = re.sub(r"\bon/off\b", "bật/tắt", suffix)
        suffix = re.sub(r"\bsettings\b", "cài đặt", suffix)
        suffix = re.sub(r"\bon\b", "bật", suffix)
        suffix = re.sub(r"\boff\b", "tắt", suffix)
        return f"Điều khiển Wi-Fi — {suffix}"
    return name


def vietnamese_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["language"] = "vi-VN"
    result["title"] = "Danh mục các cụm hành trình sản phẩm trong dữ liệu production"
    result["audience"] = "Báo cáo cho cổ đông và lãnh đạo"
    result["interpretation_notes"] = [
        "Tên cụm là diễn giải dễ đọc từ hành vi production, không phải nhãn gốc của hệ thống.",
        "Hành trình đại diện là medoid của cụm; đường dẫn kỹ thuật gốc được giữ lại để đối soát.",
        "Tỷ trọng hành trình và mã cụm chỉ áp dụng cho lần chạy production này; mã cụm có thể thay đổi sau khi huấn luyện lại.",
        "Cụm nhiễu được giữ là chưa phân loại thay vì ép vào một nhóm nghiệp vụ.",
        "Các n-gram trong trường `ngrams` được ngăn cách bằng dấu phẩy; đường đi đại diện dùng mũi tên `→` giữa các sự kiện.",
    ]

    for platform in result["platforms"]:
        platform["platform_label"] = "Android" if platform["platform"] == "android" else "iOS"
        for item in platform["summary"]["largest_clusters"]:
            item["cluster_name"] = translate_cluster_name(item["cluster_name"])
        for cluster in platform["clusters"]:
            cluster["cluster_name"] = translate_cluster_name(cluster["cluster_name"])
            cluster["business_family"] = BUSINESS_FAMILY_VI.get(cluster["business_family"], cluster["business_family"])
            basis = cluster["naming_basis"]
            basis["signals"] = [SIGNAL_VI.get(signal, signal) for signal in basis["signals"]]
            basis["source"] = "đường dẫn medoid + token vào/ra + n-gram xếp hạng của cụm"

            journey = cluster["representative_journey"]
            translated_steps = [translate_step(step) for step in journey["readable_steps"]]
            arrow_path = " → ".join(translated_steps)
            journey["step_list"] = translated_steps
            journey["readable_steps"] = arrow_path
            journey["display_path"] = arrow_path
            journey["raw_path_arrow"] = str(journey["raw_path"]).replace(" -> ", " → ")

            evidence = cluster["evidence"]
            details = evidence.pop("top_ngrams")
            evidence["ngrams"] = ", ".join(item["ngram"] for item in details)
            evidence["ngram_details"] = details

    return result


def mapping_rows(payload: dict[str, Any], vietnamese: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the catalog into a platform-aware cluster-name lookup table."""

    vi_by_platform = {item["platform"]: item for item in vietnamese["platforms"]}
    rows: list[dict[str, Any]] = []
    for platform in payload["platforms"]:
        vi_clusters = {
            int(item["cluster_id"]): item
            for item in vi_by_platform[platform["platform"]]["clusters"]
        }
        for item in platform["clusters"]:
            cluster_id = int(item["cluster_id"])
            vi_item = vi_clusters[cluster_id]
            rows.append(
                {
                    "platform": platform["platform"],
                    "cluster_id": cluster_id,
                    "cluster_name": item["cluster_name"],
                    "cluster_name_vi": vi_item["cluster_name"],
                    "business_family": item["business_family"],
                    "business_family_vi": vi_item["business_family"],
                    "naming_confidence": item["naming_confidence"],
                    "journey_count": item["metadata"]["journey_count"],
                    "journey_share": item["metadata"]["journey_share"],
                    "medoid_journey_id": item["representative_journey"]["journey_id"],
                    "medoid_length": item["representative_journey"]["event_count"],
                    "top_entry_token": item["metadata"]["top_entry_token"],
                    "top_exit_token": item["metadata"]["top_exit_token"],
                    "ngrams": ", ".join(detail["ngram"] for detail in item["evidence"]["top_ngrams"]),
                }
            )
    return sorted(rows, key=lambda row: (row["platform"], row["cluster_id"]))


def write_mapping_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster-dir",
        default="output/journey_runs/EXACT_ch-c45i30o20_ng1-3_svd64_fdf3_mf20000_nw0p35_mcs100_ms5_sel-eom_gap90_jmin4_tdf3_ent0_chr1_boot0_test0p2",
        help="directory containing <platform>_cluster_catalog.json and ngrams",
    )
    args = parser.parse_args()

    global CLUSTER_DIR, OUTPUT_PATH
    CLUSTER_DIR = Path(args.cluster_dir)
    if not CLUSTER_DIR.is_absolute():
        CLUSTER_DIR = REPO_ROOT / CLUSTER_DIR
    OUTPUT_PATH = CLUSTER_DIR / "shareholder_cluster_catalog.json"

    payload = {
        "schema_version": "1.0",
        "title": "Production Journey Cluster Catalog",
        "audience": "Shareholder and executive reporting",
        "generated_on": date.today().isoformat(),
        "interpretation_notes": [
            "Cluster names are readable interpretations of production behavior, not source-system labels.",
            "A representative journey is the production medoid path for that cluster; the raw technical path is retained for auditability.",
            "Journey shares and cluster ids are specific to this production run. Cluster ids should not be treated as stable identifiers after refitting.",
            "The noise cluster is intentionally named unclassified rather than forced into a business journey type.",
        ],
        "platforms": [platform_entry("android"), platform_entry("ios")],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    vietnamese_output = OUTPUT_PATH.with_name("shareholder_cluster_catalog_vi.json")
    vietnamese = vietnamese_payload(payload)
    vietnamese_output.write_text(
        json.dumps(vietnamese, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = mapping_rows(payload, vietnamese)
    combined_mapping = CLUSTER_DIR / "cluster_name_mapping.csv"
    write_mapping_csv(combined_mapping, rows)
    for platform in ("android", "ios"):
        write_mapping_csv(
            CLUSTER_DIR / f"{platform}_cluster_name_mapping.csv",
            [row for row in rows if row["platform"] == platform],
        )
    total = sum(platform["summary"]["cluster_count"] for platform in payload["platforms"])
    print(f"wrote {total} cluster entries -> {OUTPUT_PATH}")
    print(f"wrote {total} Vietnamese cluster entries -> {vietnamese_output}")
    print(f"wrote {len(rows)} cluster-name mappings -> {combined_mapping}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
