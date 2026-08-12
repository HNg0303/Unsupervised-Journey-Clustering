"""Shared data loading, i18n and formatting helpers for the journey-clustering dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "output" / "dashboard_cache"
RUN_DIR = (
    ROOT
    / "output"
    / "journey_runs"
    / "EXACT_ch-c45i30o20_ng1-3_svd64_fdf3_mf20000_nw0p35_mcs100_ms5_sel-eom_gap90_jmin4_tdf3_ent0_chr1_boot0_test0p2"
)
TEST_DIR = ROOT / "output" / "test"

PLATFORMS = ["android", "ios"]

# Plot palette (colour-blind safe, works on light and dark themes)
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]


# ======================================================================================
# i18n
# ======================================================================================
def lang() -> str:
    """Current UI language: 'EN' or 'VI'."""
    return st.session_state.get("lang", "EN")


def is_vi() -> bool:
    return lang() == "VI"


def t(en: str, vi: str) -> str:
    """Pick the string for the active language. Technical terms stay in English in VI copy."""
    return vi if is_vi() else en


# Business families are stored in English snake_case (cluster_name_mapping.csv) but the
# same file also carries Vietnamese labels; these maps cover both directions.
FAMILY_EN = {
    "quản lý thiết bị": "device management",
    "thanh toán": "payment",
    "hợp đồng": "contract",
    "tương tác": "engagement",
    "điều hướng": "navigation",
    "hỗ trợ": "support",
    "thương mại": "commerce",
    "chưa phân loại": "unclassified",
    "tài khoản": "account",
    "xác thực": "authentication",
    "giải trí": "entertainment",
    "khuyến mãi": "promotion",
}

FAMILY_VI = {
    "device management": "quản lý thiết bị",
    "device_management": "quản lý thiết bị",
    "payment": "thanh toán",
    "payments": "thanh toán",
    "contract": "hợp đồng",
    "contracts": "hợp đồng",
    "engagement": "tương tác",
    "navigation": "điều hướng",
    "support": "hỗ trợ",
    "commerce": "thương mại",
    "unclassified": "chưa phân loại",
    "unknown": "chưa phân loại",
    "novel": "hành vi mới",
    "account": "tài khoản",
    "authentication": "xác thực",
    "entertainment": "giải trí",
    "promotion": "khuyến mãi",
    "service management": "quản lý dịch vụ",
    "service_management": "quản lý dịch vụ",
    "feedback": "phản hồi",
}


def pretty_family(value) -> str:
    """Render a business family in the active language."""
    v = str(value)
    if is_vi():
        key = FAMILY_EN.get(v, v).replace("_", " ")
        return FAMILY_VI.get(key, FAMILY_VI.get(v, key))
    return FAMILY_EN.get(v, v).replace("_", " ")


FRICTION_EXPLAIN_EN = {
    "improbable_transitions": "Steps the trained transition model rates as unlikely — user went somewhere the flow does not normally lead.",
    "screen_thrash": "Rapid back-and-forth between the same screens — the user could not find what they wanted.",
    "excessive_back": "High share of back taps — the flow sent the user down a wrong path.",
    "navigation_loop": "The same short screen cycle repeats — a dead end in the UI.",
    "slow_journey": "Journey took far longer than the norm for its type.",
    "unknown_archetype": "Behaviour does not match any learned journey type — new or broken flow.",
}

FRICTION_EXPLAIN_VI = {
    "improbable_transitions": "Các bước mà Markov model đánh giá là ít có khả năng xảy ra — user đi tới nơi mà flow bình thường không dẫn tới.",
    "screen_thrash": "Nhảy qua lại liên tục giữa vài screen — user không tìm được thứ mình cần.",
    "excessive_back": "Tỷ lệ bấm back cao — flow đã đưa user đi sai hướng.",
    "navigation_loop": "Một vòng lặp screen ngắn lặp đi lặp lại — ngõ cụt trong UI.",
    "slow_journey": "Journey kéo dài hơn hẳn mức bình thường của cùng loại journey.",
    "unknown_archetype": "Hành vi không khớp journey type nào đã học — flow mới hoặc đang lỗi.",
}


def friction_explain() -> dict:
    return FRICTION_EXPLAIN_VI if is_vi() else FRICTION_EXPLAIN_EN


# ======================================================================================
# loaders
# ======================================================================================
def missing_cache_notice(what: str) -> None:
    st.error(
        t(
            f"Cache for **{what}** not found. Build it first:",
            f"Chưa có cache cho **{what}**. Hãy build trước:",
        )
        + "\n\n```bash\npython dashboard/prepare_dashboard_data.py\n```"
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_eda() -> dict:
    path = CACHE_DIR / "eda.json"
    if not path.exists():
        missing_cache_notice(t("raw-data EDA", "EDA dữ liệu thô"))
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_train_run() -> dict:
    path = CACHE_DIR / "train_run.json"
    if not path.exists():
        missing_cache_notice(t("training run", "training run"))
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_inference(platform: str) -> pd.DataFrame:
    path = CACHE_DIR / f"inference_{platform}.parquet"
    if not path.exists():
        missing_cache_notice(f"inference ({platform})")
    df = pd.read_parquet(path)
    df["start_ts"] = pd.to_datetime(df["start_ts"], errors="coerce", utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_run_csv(name: str) -> pd.DataFrame:
    path = RUN_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


@st.cache_data(show_spinner=False)
def load_cluster_names() -> pd.DataFrame:
    df = load_run_csv("cluster_name_mapping.csv")
    if df.empty:
        missing_cache_notice(t("cluster name mapping", "bảng tên cluster"))
    return df


@st.cache_data(show_spinner=False)
def load_shareholder_catalog() -> dict:
    path = RUN_DIR / "shareholder_cluster_catalog.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_shareholder_catalog_vi() -> dict:
    path = RUN_DIR / "shareholder_cluster_catalog_vi.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ======================================================================================
# formatting
# ======================================================================================
def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """items = [(label, value, help_text)]"""
    cols = st.columns(len(items))
    for col, (label, value, helptext) in zip(cols, items):
        col.metric(label, value, help=helptext)


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(x, digits: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def hist_to_df(h: dict, value_name: str = "count") -> pd.DataFrame:
    edges = h["edges"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    return pd.DataFrame({"bin": centers, value_name: h["counts"]})


def quantile_table(qdict: dict, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"quantile": list(qdict.keys()), label: list(qdict.values())}
    )


def explode_flags(series: pd.Series) -> pd.Series:
    """friction_flags is a '|'-joined string; return exploded flag counts."""
    s = series.fillna("").astype(str)
    s = s[s != ""]
    if s.empty:
        return pd.Series(dtype="int64")
    return s.str.split("|").explode().str.strip().value_counts()
