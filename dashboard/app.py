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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import RUN_DIR, t  # noqa: E402
from views import page_clusters, page_eda, page_inference, page_training  # noqa: E402

st.set_page_config(
    page_title="Journey Clustering — production results",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Page keys are language-stable so switching language never invalidates the selection.
PAGES = {
    "eda": page_eda.render,
    "training": page_training.render,
    "clusters": page_clusters.render,
    "inference": page_inference.render,
}
PAGE_LABELS = {
    "eda": t("1 · Raw data (EDA)", "1 · Dữ liệu thô (EDA)"),
    "training": t("2 · Training outputs", "2 · Các file khi train"),
    "clusters": t("3 · Learned journey types", "3 · Journey type học được"),
    "inference": t("4 · Production results", "4 · Kết quả trên production"),
}

with st.sidebar:
    st.caption(t("Unsupervised journeys from HiFPT clickstream", "Journey không giám sát từ clickstream HiFPT"))
    choice = st.radio(
        t("Page", "Trang"),
        list(PAGES),
        format_func=lambda k: PAGE_LABELS[k],
        label_visibility="collapsed",
        key="page",
    )
    st.divider()
    st.markdown(
        t(
            f"""
**Data**
`data/train_data/raw_data_production/data_raw_sample` — Android + iOS, extracts T3 / T4 / T5.

**Run**
`{RUN_DIR.name}`

**Pipeline**
raw events → canonization + taxonomy → journey segmentation → multi-channel TF-IDF + SVD →
HDBSCAN + Markov → named journey catalogue → scoring.
            """,
            f"""
**Dữ liệu**
`data/train_data/raw_data_production/data_raw_sample` — Android + iOS, các extract T3 / T4 / T5.

**Run**
`{RUN_DIR.name}`

**Pipeline**
event thô → canonization + taxonomy → cắt journey → TF-IDF đa kênh + SVD →
HDBSCAN + Markov → catalogue journey đã đặt tên → scoring.
            """,
        )
    )
    st.divider()
    st.caption(
        t(
            "Aggregates come from `output/dashboard_cache`. Rebuild after a new run:",
            "Số liệu tổng hợp lấy từ `output/dashboard_cache`. Build lại sau mỗi run mới:",
        )
        + "\n\n`python dashboard/prepare_dashboard_data.py`"
    )

PAGES[choice]()
