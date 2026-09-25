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


def canonize_name(raw: object, *, cfg: CanonizeConfig, missing: str = MISSING) -> str:
    """Canonicalise one screen/segment string without semantic merging."""
    if raw is None or pd.isna(raw):
        return missing
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return missing

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
    """Return canonical production events, accepting raw or canonical input."""
    if set(("event_time", "event_type", "segment_name", "event_token")).issubset(df.columns):
        out = df.copy()
        out["event_time"] = pd.to_datetime(out["event_time"], utc=True, format="mixed", errors="coerce")
        return out.sort_values(
            ["platform", "session_id", "event_time", "source_file", "source_row_number"], kind="mergesort"
        ).reset_index(drop=True)
    from .preprocessing import canonicalize_frame

    return canonicalize_frame(df, cfg=cfg)[0]


def canonization_report(raw: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    """Before/after vocabulary counts — evidence that canonisation paid off."""
    raw_triples = raw["key"].astype(str) + "|" + raw["segmentation_name"].astype(str)
    canon_triples = canon["event_token"]

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
            _stats(canon_triples, "canonical event_type@segment_name"),
            _stats(canon["segment_name"], "canonical segment_name"),
        ]
    )
