"""Page 5 — customer-based journey and post-VNeID views."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.lib import (
    PALETTE,
    fmt_int,
    fmt_pct,
    kpi_row,
    load_customer_analysis_csv,
    t,
)


def _load(name: str) -> pd.DataFrame:
    return load_customer_analysis_csv(name, bundle_key=st.session_state.get("inference_bundle"))


def render() -> None:
    st.title(t("5 · Customer-based results", "5 · Kết quả theo customer"))
    st.caption(
        t(
            "Every customer-level number is grouped by customer_id. Post-VNeID detail stays one row per action record.",
            "Mọi thống kê customer-level đều group theo customer_id. Chi tiết sau ký VNeID giữ một dòng cho từng record hành động.",
        )
    )

    customers = _load("customer_metrics.csv")
    monthly = _load("customer_month_unique.csv")
    focus = _load("focus_customer_summary.csv")
    post = _load("vneid_post_action_summary.csv")
    if post.empty:
        # Compatibility with the separate VNeID analysis module and the earlier
        # family-only export. Both remain row-level/customer-level aggregates.
        post = _load("vneid_econtract_post_signature_behavior_summary.csv")
        if post.empty:
            post = _load("vneid_econtract_post_signature_family_summary.csv")
        if post.empty:
            post = _load("vneid_post_family_summary.csv")
        if not post.empty:
            post = post.rename(columns={
                "unique_users": "unique_customers",
                "cohort_users": "cohort_customers",
                "pct_of_cohort": "pct_of_signature_cohort",
                "bucket_label": "time_bucket",
            })
            if "time_bucket" not in post.columns and {"time_unit", "elapsed_bucket"} <= set(post.columns):
                post["time_bucket"] = post.apply(
                    lambda r: f"{r.get('time_unit', '')}_{int(r.get('elapsed_bucket', 0))}", axis=1
                )
            post["record_count"] = post.get("unique_customers", 0)
    post_records = _load("vneid_post_action_records.csv")
    loyalty_post = _load("loyalty_post_action_summary.csv")
    loyalty_records = _load("loyalty_post_action_records.csv")
    if customers.empty:
        st.warning(
            t(
                "Customer analysis is not available. Run the independent scripts under scripts/post_analysis/ first.",
                "Chưa có customer analysis. Hãy chạy các script độc lập trong scripts/post_analysis/ trước.",
            )
        )
        return

    platform_options = ["all"] + sorted({x for value in customers["platforms"].dropna().astype(str) for x in value.split(";") if x})
    selected_platform = st.selectbox(t("Platform", "Platform"), platform_options, format_func=str.title)
    scoped = customers if selected_platform == "all" else customers[customers["platforms"].str.contains(selected_platform, na=False)]
    n_customers = len(scoped)
    n_journeys = pd.to_numeric(scoped["journey_count"], errors="coerce").sum()
    medians = pd.to_numeric(scoped["median_distance_between_journeys_hours"], errors="coerce").dropna()
    kpi_row([
        (t("Unique customers", "Unique customer"), fmt_int(n_customers), None),
        (t("Journeys", "Journey"), fmt_int(n_journeys), None),
        (t("Median distance between journeys", "Median khoảng cách giữa journeys"), f"{medians.median():.1f} h" if not medians.empty else "—", t("Median across customers with at least two journeys", "Median trên các customer có ít nhất hai journey")),
        (t("Journeys per active month", "Journey mỗi tháng active"), f"{pd.to_numeric(scoped['journeys_per_active_month'], errors='coerce').median():.2f}" if not scoped.empty else "—", None),
    ])

    st.divider()
    st.header(t("Customer journey footprint", "Dấu chân journey theo customer"))
    left, right = st.columns([3, 2])
    with left:
        show = scoped[["customer_id", "journey_count", "platforms", "unique_month_count", "first_journey_ts", "last_journey_ts", "median_distance_between_journeys_hours"]].copy()
        show = show.sort_values("journey_count", ascending=False).head(100)
        st.dataframe(show, hide_index=True, width="stretch", height=420)
    with right:
        if not monthly.empty:
            m = monthly.copy()
            if selected_platform != "all":
                m = m[(m["scope"] == "platform") & (m["platform"] == selected_platform)]
            else:
                m = m[m["scope"].eq("all")]
            m["month"] = pd.to_datetime(m["month"], format="%Y-%m", errors="coerce")
            m = m.sort_values("month")
            fig = px.line(m, x="month", y="unique_customers", markers=True, color_discrete_sequence=PALETTE)
            fig.update_layout(height=390, margin=dict(t=10, b=10, l=10, r=10), yaxis_title=t("unique customers", "unique customer"), xaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(t("No monthly table found.", "Không có bảng theo tháng."))

    st.divider()
    st.header(t("Focus journeys — success and failure markers", "Journey trọng tâm — marker thành công và thất bại"))
    if focus.empty:
        st.info(t("No focus journey file found.", "Không có file focus journey."))
    else:
        f = focus.copy()
        if "unique_customers" not in f.columns:
            # Legacy focus_customer_summary is already customer × focus; make the chart
            # compatible without pretending its platform/outcome markers are known.
            f = (
                f.groupby("focus_object", as_index=False)
                .agg(record_count=("record_count", "sum"), unique_customers=("customer_id", "nunique"))
            )
            f["platform"] = "all"
            f["outcome"] = "unknown"
        f["platform"] = f.get("platform", "all").fillna("all")
        f["outcome"] = f.get("outcome", "unknown").fillna("unknown")
        if selected_platform != "all":
            f = f[f["platform"] == selected_platform]
        f["label"] = f["focus_object"] + " · " + f["outcome"]
        fig = px.bar(f.sort_values("unique_customers"), x="unique_customers", y="label", orientation="h", color="outcome", color_discrete_sequence=PALETTE)
        fig.update_layout(height=430, margin=dict(t=10, b=10, l=10, r=10), xaxis_title=t("unique customers", "unique customer"), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(f.sort_values("unique_customers", ascending=False), hide_index=True, width="stretch")

    st.divider()
    st.header(t("What customers do after signing with VNeID", "Customer làm gì sau khi ký bằng VNeID"))
    if post.empty:
        st.info(t("No post-VNeID analysis file found.", "Không có file phân tích sau VNeID."))
    else:
        p = post.copy()
        # Canonical output is one row per time bucket × named action. Older caches used
        # focus_object; accepting both keeps the page usable during cache refresh.
        action_col = "business_family" if "business_family" in p.columns else "focus_object"
        if action_col not in p.columns:
            st.info(t("Post-VNeID file has no action label column.", "File sau VNeID chưa có cột action label."))
            return
        fig = px.bar(
            p, x="time_bucket", y="unique_customers", color=action_col,
            barmode="group", color_discrete_sequence=PALETTE,
        )
        fig.update_layout(height=430, margin=dict(t=10, b=10, l=10, r=10), xaxis_title=None, yaxis_title=t("unique customers", "unique customer"))
        st.plotly_chart(fig, width="stretch")
        sort_cols = [c for c in ["time_bucket", "unique_customers"] if c in p.columns]
        st.dataframe(p.sort_values(sort_cols, ascending=[True, False][:len(sort_cols)]), hide_index=True, width="stretch")

        with st.expander(t("Show row-level post-signature actions", "Xem actions sau ký theo từng record")):
            if post_records.empty:
                st.info(t("No detail records found.", "Không có record chi tiết."))
            else:
                detail = post_records
                if selected_platform != "all":
                    detail = detail[detail["platform"] == selected_platform]
                st.caption(t("Rows are intentionally not concatenated by customer.", "Các dòng không được gom chuỗi theo customer."))
                st.dataframe(detail.head(5000), hide_index=True, width="stretch", height=520)

    st.divider()
    st.header(t("What customers do after using Loyalty", "Customer làm gì sau khi dùng Loyalty"))
    if loyalty_post.empty:
        st.info(t("No Loyalty analysis file found. Re-run the customer analysis script.", "Chưa có file phân tích Loyalty. Hãy chạy lại script customer analysis."))
    else:
        lp = loyalty_post.copy()
        loyalty_action_col = "business_family" if "business_family" in lp.columns else "focus_object"
        if loyalty_action_col in lp.columns and "unique_customers" in lp.columns:
            fig = px.bar(
                lp,
                x="time_bucket",
                y="unique_customers",
                color=loyalty_action_col,
                barmode="group",
                color_discrete_sequence=PALETTE,
            )
            fig.update_layout(
                height=430,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title=None,
                yaxis_title=t("unique customers", "unique customer"),
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                lp.sort_values(["time_bucket", "unique_customers"], ascending=[True, False]),
                hide_index=True,
                width="stretch",
            )
        if not loyalty_records.empty:
            with st.expander(t("Show row-level post-Loyalty actions", "Xem actions sau Loyalty theo từng record")):
                detail = loyalty_records
                if selected_platform != "all" and "platform" in detail.columns:
                    detail = detail[detail["platform"] == selected_platform]
                st.caption(t("Rows are intentionally not concatenated by customer.", "Các dòng không được gom chuỗi theo customer."))
                st.dataframe(detail.head(5000), hide_index=True, width="stretch", height=520)
