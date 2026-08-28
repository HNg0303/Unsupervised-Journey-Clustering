"""Streamlit dashboard for the unsupervised journey-clustering results.

Run:
    streamlit run dashboard/app.py

The dashboard reads a precomputed cache (output/dashboard_cache). Build it once with:
    python dashboard/prepare_dashboard_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.lib import RUN_DIR, SUMMARY_DIR, TRAIN_RUN_DIR, inject_css, t  # noqa: E402
from dashboard.views import (  # noqa: E402
    page_clusters,
    page_eda,
    page_inference,
    page_overview,
    page_customer_based,
    page_training,
)

st.set_page_config(
    page_title="Journey Clustering — production results",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# The language selector must be read before any page renders, so it is created first and
# every downstream string goes through lib.t().
with st.sidebar:
    st.title("🧭 Journey Clustering")
    st.segmented_control(
        "Language",
        ["EN", "VI"],
        default="EN",
        key="lang",
        label_visibility="collapsed",
        help="English / Tiếng Việt",
    )

# Two top-level sections. Detail pages live under the second one. All keys are
# language-stable so switching language never invalidates a selection.
SECTIONS = {
    "overview": t("Overview", "Tổng quan"),
    "details": t("Details", "Chi tiết"),
}
DETAIL_PAGES = {
    "eda": page_eda.render,
    "training": page_training.render,
    "clusters": page_clusters.render,
    "inference": page_inference.render,
    "customer_based": page_customer_based.render,
}
DETAIL_LABELS = {
    "eda": t("1 · Raw data (EDA)", "1 · Dữ liệu thô (EDA)"),
    "training": t("2 · Training outputs", "2 · Các file khi train"),
    "clusters": t("3 · Learned journey types", "3 · Journey type học được"),
    "inference": t("4 · Production results", "4 · Kết quả trên production"),
    "customer_based": t("5 · Customer-based results", "5 · Kết quả theo customer"),
}

with st.sidebar:
    st.caption(t("Unsupervised journeys from HiFPT clickstream", "Journey không giám sát từ clickstream HiFPT"))
    section = st.radio(
        t("Section", "Mục"),
        list(SECTIONS),
        format_func=lambda k: SECTIONS[k],
        label_visibility="collapsed",
        key="section",
    )
    if section == "details":
        page = st.radio(
            t("Page", "Trang"),
            list(DETAIL_PAGES),
            format_func=lambda k: DETAIL_LABELS[k],
            key="page",
        )
    st.divider()
    st.markdown(
        t(
            f"""
**Inference data**
`output/scores/pca48_ngrams12_500` — full Android + iOS scored journey partitions.

**Dashboard summaries**
`output/scores/pca48_ngrams12_500/html_dashboard_summary` — compact aggregates used by the pages.

**Training run**
`output/partitioned_runs/pca48_svd48_ngrams12_500k/latest` — fitted model + holdout artefacts.

**Preparation**
raw events → canonization + taxonomy → journey segmentation → multi-channel TF-IDF + SVD →
HDBSCAN + Markov → named journey catalogue → scoring → post-analysis summaries.
            """,
            f"""
**Dữ liệu inference**
`output/scores/pca48_ngrams12_500` — các partition journey đã score cho Android + iOS.

**Summary dashboard**
`output/scores/pca48_ngrams12_500/html_dashboard_summary` — các bảng aggregate mà dashboard sử dụng.

**Run train**
`output/partitioned_runs/pca48_svd48_ngrams12_500k/latest` — model fit và holdout.

**Chuẩn bị dữ liệu**
event thô → canonization + taxonomy → cắt journey → TF-IDF đa kênh + SVD →
HDBSCAN + Markov → catalogue journey đã đặt tên → scoring → summary post-analysis.
            """,
        )
    )
    st.divider()
    st.caption(
        t(
            "Raw EDA is opt-in. Aggregate pages read compact inference summaries; names come from authoritative `Cluster_naming.csv`.",
            "EDA raw phải chạy riêng. Các trang aggregate đọc summary inference compact; tên lấy từ `Cluster_naming.csv` authoritative.",
        )
        + "\n\n`python dashboard/prepare_dashboard_data.py`"
    )

if section == "overview":
    page_overview.render()
else:
    DETAIL_PAGES[st.session_state.get("page", "eda")]()
