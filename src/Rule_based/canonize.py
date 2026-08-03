"""Stage 1 - Canonicalisation.

The single most important fact about this dataset is that the *semantic role of
the columns flips with `event_type`*:

    event_type == "View"    -> segment_name IS the screen, screen_name is NULL (100%)
    event_type == "Action"  -> screen_name  IS the screen, segment_name is the
                               action path within that screen (98.4% populated)

Concatenating (event_type, segment_name, screen_name) as a flat triple therefore
mixes two different field meanings into one vocabulary and, worse, puts View
tokens and Action tokens into *disjoint namespaces* — the model can never learn
that `View HomeVC` and `Action Home/Nav_profile @ HomeVC` happen on the same
screen. This module fixes that by producing a role-correct canonical form:

    (event_type, screen, target)

Canonicalisation here is strictly *structural*. No semantic merging is done:
`HOME`, `android/Home` and `HomeVC` stay three distinct screens, and screens are
additionally namespaced by OS so identical strings on different platforms can
never collide.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, parse_qsl

import numpy as np
import pandas as pd

from .config import (
    BACK_ACTION_MARKERS,
    BOOT_SCREENS,
    CHROME_SCREENS,
    CanonizeConfig,
    ROOT_SCREENS,
    SEMANTIC_QUERY_KEYS,
)

MISSING = "<none>"

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_HEX24_RE = re.compile(r"^[0-9a-f]{16,}$", re.I)
_DIGIT_RE = re.compile(r"^\d+$")
# Contract/order codes seen in this app: SGABP2073, HNIJH0026HQV, AGAAF1671 ...
_CODE_RE = re.compile(r"^(?=.*\d)[A-Z0-9]{6,}$")


def _mask_segment(seg: str) -> str:
    """Replace an instance identifier with a type placeholder."""
    if not seg:
        return seg
    if _DIGIT_RE.match(seg):
        return "{id}"
    if _UUID_RE.match(seg) or _HEX24_RE.match(seg):
        return "{uuid}"
    if _CODE_RE.match(seg):
        return "{code}"
    return seg


def canonize_url(raw: str, *, mask_ids: bool = True) -> str:
    """Collapse a webview URL to `host/path` plus semantic query keys only.

    575 of the 821 distinct View screens are URLs, and their long tail is almost
    entirely `contractNo=`/`orderId=`/`timestamp=` variation. Dropping those
    params is the single largest, and safest, vocabulary reduction available:
    it removes identifiers, not behaviour.
    """
    parts = urlsplit(raw)
    host = parts.netloc or ""
    path = parts.path or "/"

    if mask_ids: # Turn IDs, Digits and Code into type instead of preserving the identifiers.
        path = "/".join(_mask_segment(seg) for seg in path.split("/"))
    path = path.rstrip("/") or "/"

    kept = sorted(
        (k.lower(), v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() in SEMANTIC_QUERY_KEYS and v
    )
    suffix = ("?" + "&".join(f"{k}={v}" for k, v in kept)) if kept else ""
    return f"{host}{path}{suffix}"


def canonize_name(raw: object, *, cfg: CanonizeConfig) -> str:
    """Canonicalise one screen/segment string without semantic merging."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return MISSING
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return MISSING

    if cfg.canonize_urls and "://" in text:
        return canonize_url(text, mask_ids=cfg.mask_id_segments)

    if cfg.mask_id_segments and "/" in text:
        return "/".join(_mask_segment(seg) for seg in text.split("/"))
    return text


def classify_screen(bare_screen: str) -> str:
    """Label a screen as ux / chrome / boot so noise can be filtered by policy."""
    if bare_screen in BOOT_SCREENS:
        return "boot"
    if bare_screen in CHROME_SCREENS:
        return "chrome"
    return "ux"


def is_back_action(target: str) -> bool:
    low = target.lower()
    return any(marker in low for marker in BACK_ACTION_MARKERS)


def canonize_events(df: pd.DataFrame, cfg: CanonizeConfig) -> pd.DataFrame:
    """Raw event rows -> role-correct canonical events, sorted within session.

    Input columns required:
        record_id, session_id, device_id, customer_id, created_at, timestamp,
        event_type, OS, segment_name, screen_name, duration

    Output adds:
        ts, screen_bare, screen, target, screen_class, is_back,
        duration_clip, gap_prev_s
    """
    required = {
        "record_id",
        "session_id",
        "created_at",
        "key",
        "segmentation.segment",
        "segmentation.name",
        "segmentation.screen_id",
        "duration",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["key"] = out["key"].astype(str).str.strip().str.title()
    out["ts"] = pd.to_datetime(out["created_at"], utc=True, format="mixed")

    is_view = out["key"].eq("View")

    # --- the role fix -----------------------------------------------------
    # View : screen comes from segment_name, there is no target
    # Action: screen comes from screen_name, target is the action path
    screen_raw = out["segmentation.screen_id"].where(~is_view, out["segmentation.name"])
    target_raw = out["segmentation.name"].where(~is_view, other=np.nan)

    out["screen_bare"] = [canonize_name(x, cfg=cfg) for x in screen_raw]
    out["target"] = [canonize_name(x, cfg=cfg) for x in target_raw]

    out["screen_class"] = [classify_screen(s) for s in out["screen_bare"]]

    if cfg.namespace_by_os:
        os_series = out["segmentation.segment"].fillna("unk").astype(str)
        out["screen"] = os_series + "::" + out["screen_bare"]
    else:
        out["screen"] = out["screen_bare"]

    out["is_back"] = [
        (t != MISSING) and is_back_action(t) for t in out["target"]
    ]

    # duration is emitted on View rows only (Action duration is always 0) and
    # carries multi-hour instrumentation artefacts; clip rather than drop.
    out["duration_clip"] = (
        pd.to_numeric(out["duration"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=cfg.duration_clip_seconds)
    )

    out = out.sort_values(["session_id", "ts", "record_id"], kind="mergesort").reset_index(drop=True)
    out["gap_prev_s"] = (
        out.groupby("session_id", sort=False)["ts"].diff().dt.total_seconds().fillna(0.0)
    )
    return out


def canonization_report(raw: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    """Before/after vocabulary counts — evidence that canonisation paid off."""
    raw_triples = (
        raw["event_type"].astype(str)
        + "|"
        + raw["segment_name"].astype(str)
        + "|"
        + raw["screen_name"].astype(str)
    )
    canon_triples = canon["event_type"] + "|" + canon["screen"] + "|" + canon["target"]

    def _stats(series: pd.Series, label: str) -> dict[str, object]:
        vc = series.value_counts()
        return {
            "stage": label,
            "vocabulary": int(len(vc)),
            "singletons": int((vc == 1).sum()),
            "singleton_share": round(float((vc == 1).mean()), 4),
            "df_lt_5": int((vc < 5).sum()),
            "top100_coverage": round(float(vc.head(100).sum() / vc.sum()), 4),
        }

    return pd.DataFrame(
        [
            _stats(raw_triples, "raw exact triple"),
            _stats(canon_triples, "canonical (type, screen, target)"),
            _stats(canon["screen"], "canonical screen axis only"),
        ]
    )

if __name__ == "__main__":
    import os
    canon_config = CanonizeConfig() #use default.
    raw_android_path = "data/clean_july_events_android.csv"
    raw_ios_path = "data/clean_july_events_ios.csv"
    canon_android_df = canonize_events(pd.read_csv(raw_android_path), canon_config)
    canon_ios_df = canonize_events(pd.read_csv(raw_ios_path), canon_config)
    output_path = "outputs/journeys"
    os.makedirs(output_path, exist_ok=True)
    canon_android_df.to_csv(os.path.join(output_path, "canonized_events_android.csv"), index=False)
    canon_ios_df.to_csv(os.path.join(output_path, "canonized_events_ios.csv"), index=False)