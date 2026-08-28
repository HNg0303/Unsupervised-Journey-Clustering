"""Page 4 — full-data production inference, read from compact summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.lib import (
    PALETTE,
    SUMMARY_DIR,
    fmt_int,
    fmt_pct,
    kpi_row,
    load_inference,
    load_inference_profile,
    load_summary_csv,
    load_summary_manifest,
    pretty_family,
    t,
)


PLATFORMS = ["android", "ios"]


def _selected(frame: pd.DataFrame, platforms: list[str]) -> pd.DataFrame:
    if frame.empty or "platform" not in frame.columns:
        return frame.copy()
    return frame[frame["platform"].isin(platforms)].copy()


def _kpi(platforms: list[str]) -> dict:
    table = load_summary_csv("kpi.csv")
    if table.empty:
        return {}
    scope = "all" if set(platforms) == set(PLATFORMS) else platforms[0]
    row = table[table["scope"].astype(str).eq(scope)]
    if not row.empty:
        return row.iloc[0].to_dict()
    rows = table[table["scope"].isin(platforms)]
    if rows.empty:
        return {}
    out = rows.iloc[0].to_dict()
    out["journeys"] = pd.to_numeric(rows["journeys"], errors="coerce").sum()
    out["sessions"] = pd.to_numeric(rows["sessions"], errors="coerce").sum()
    out["customers"] = pd.to_numeric(rows["customers"], errors="coerce").sum()
    weights = pd.to_numeric(rows["journeys"], errors="coerce").replace(0, np.nan)
    for col in ["mean_steps", "mean_span_seconds"]:
        out[col] = (pd.to_numeric(rows[col], errors="coerce") * weights).sum() / weights.sum()
    return out


def _cluster_metrics(platforms: list[str]) -> tuple[pd.DataFrame, dict]:
    """Return exact inference cluster totals and derived assignment coverage."""
    table = _selected(load_summary_csv("cluster_summary.csv"), platforms)
    if table.empty:
        return table, {}
    table["cluster"] = pd.to_numeric(table["cluster"], errors="coerce")
    table["journeys"] = pd.to_numeric(table["journeys"], errors="coerce").fillna(0)
    total = float(table["journeys"].sum())
    unresolved = float(table.loc[table["cluster"].eq(-1), "journeys"].sum())
    known = table[table["cluster"].ne(-1)].copy()
    n_clusters = int(known[["platform", "cluster"]].drop_duplicates().shape[0])
    n_types = int(known["journey_type_en"].nunique()) if "journey_type_en" in known else 0
    return table, {
        "total": total,
        "unresolved": unresolved,
        "known_rate": 1.0 - unresolved / total if total else 0.0,
        "n_clusters": n_clusters,
        "n_types": n_types,
    }


def _label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["type_label"] = frame["journey_type"] if "journey_type" in frame else "—"
    frame["family_label"] = frame["business_family"] if "business_family" in frame else "—"
    if st.session_state.get("lang", "EN") != "VI":
        if "journey_type_en" in frame:
            frame["type_label"] = frame["journey_type_en"].fillna(frame["type_label"])
        if "business_family_code" in frame:
            frame["family_label"] = frame["business_family_code"].fillna(frame["family_label"]).map(pretty_family)
    return frame


def _date_range(frame: pd.DataFrame):
    dates = pd.to_datetime(frame.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().date(), dates.max().date()


def render() -> None:
    st.title(t("4 · Production results — what users actually did", "4 · Kết quả production — user thực sự đã làm gì"))
    st.caption(
        t(
            "Full-data inference is shown from compact aggregates generated from the scored journey exports. "
            "The app does not load the multi-GB parquet lake into memory.",
            "Inference trên full data được đọc từ các bảng aggregate compact sinh ra từ output journey đã score. "
            "Dashboard không nạp toàn bộ parquet nhiều GB vào memory.",
        )
    )

    platforms = st.multiselect(
        t("Platform", "Platform"), PLATFORMS, default=PLATFORMS,
        format_func=str.title, key="inference_platforms",
    )
    if not platforms:
        st.warning(t("Select at least one platform.", "Hãy chọn ít nhất một platform."))
        return

    kpi = _kpi(platforms)
    clusters, cluster_meta = _cluster_metrics(platforms)
    if not kpi or clusters.empty:
        st.error(
            t(
                f"Inference summaries are missing under `{SUMMARY_DIR}`. Run the summary extractor first.",
                f"Chưa có inference summary trong `{SUMMARY_DIR}`. Hãy chạy script sinh summary trước.",
            )
        )
        return

    total = int(kpi.get("journeys", cluster_meta["total"]) or 0)
    assigned = cluster_meta["known_rate"]
    profiles = [load_inference_profile(p) for p in platforms]
    kpi_row(
        [
            (t("Scored journeys", "Journey đã score"), fmt_int(total), t("One row = one journey", "Một dòng = một journey")),
            (t("Sessions", "Session"), fmt_int(kpi.get("sessions")), t("Distinct sessions in the selected bundle", "Số session khác nhau trong bundle")),
            (t("Customers", "Customer"), fmt_int(kpi.get("customers")), t("Unique customer ids; anonymous ids excluded by the extractor", "Customer id duy nhất; loại id anonymous")),
            (t("Recognised by the scorer", "Được scorer nhận diện"), fmt_pct(assigned), t("Cluster assignment within the learned distance limit", "Được gán cluster trong distance limit đã học")),
            (t("Behavioural friction", "Behavioural friction"), fmt_pct(kpi.get("struggle_rate")), t("Back, revisit, loop or unusually slow", "Back, revisit, loop hoặc chậm bất thường")),
            (t("Any model flag", "Có model flag"), fmt_pct(kpi.get("any_flag_rate")), t("A journey may carry more than one flag", "Một journey có thể có nhiều flag")),
        ]
    )

    manifest = load_summary_manifest()
    if manifest:
        st.caption(
            t(
                f"Summary generated {manifest.get('generated_at', '—')}. Source grain: one row per journey; "
                f"valid start timestamps only. Source directory: `{SUMMARY_DIR}`.",
                f"Summary sinh lúc {manifest.get('generated_at', '—')}. Grain nguồn: một dòng mỗi journey; "
                f"chỉ tính dòng có start timestamp hợp lệ. Thư mục nguồn: `{SUMMARY_DIR}`.",
            )
        )

    st.divider()

    # ------------------------------------------------------------------ traffic mix
    st.header(t("What are people doing in the app?", "User đang làm gì trong app?"))
    family = _label_columns(_selected(load_summary_csv("family_summary.csv"), platforms))
    if not family.empty:
        # The summary keeps one row per platform/family-code. Collapse those rows into
        # one human-facing family with journey-weighted rates; a simple mean would give
        # tiny platform slices the same influence as the dominant traffic slice.
        family["_journeys_num"] = pd.to_numeric(family["journeys"], errors="coerce").fillna(0)
        family["_struggle_weight"] = family["struggle_rate"] * family["_journeys_num"]
        family["_flag_weight"] = family["any_flag_rate"] * family["_journeys_num"]
        family = family.groupby("family_label", as_index=False).agg(
            journeys=("journeys", "sum"),
            sessions=("sessions", "sum"),
            customers=("customers", "sum"),
            _journeys_num=("_journeys_num", "sum"),
            _struggle_weight=("_struggle_weight", "sum"),
            _flag_weight=("_flag_weight", "sum"),
            median_steps=("median_steps", "median"),
            median_span_seconds=("median_span_seconds", "median"),
            cluster_count=("cluster_count", "sum"),
        ).sort_values("journeys", ascending=False)
        family["struggle_rate"] = family["_struggle_weight"] / family["_journeys_num"].replace(0, np.nan)
        family["any_flag_rate"] = family["_flag_weight"] / family["_journeys_num"].replace(0, np.nan)
        family["platform_share"] = family["journeys"] / family["journeys"].sum()
        left, right = st.columns([3, 2])
        fig = px.bar(
            family.sort_values("journeys"), x="journeys", y="family_label", orientation="h",
            color="struggle_rate", color_continuous_scale=["#4C78A8", "#EECA3B", "#E45756"],
            hover_data={"sessions": ":,", "customers": ":,", "platform_share": ":.1%"},
        )
        fig.update_layout(height=430, margin=dict(t=10, b=10, l=10, r=10), yaxis_title=None,
                          coloraxis_colorbar_title=t("struggle", "chật vật"))
        left.plotly_chart(fig, width="stretch")
        show = family[["family_label", "journeys", "platform_share", "struggle_rate", "any_flag_rate",
                       "median_steps", "median_span_seconds", "cluster_count"]].rename(columns={
                           "family_label": t("business family", "business family"),
                           "journeys": t("journeys", "journey"),
                           "platform_share": t("share", "tỷ lệ"),
                           "struggle_rate": t("struggle", "chật vật"),
                           "any_flag_rate": t("any flag", "mọi flag"),
                           "median_steps": t("median steps", "bước trung vị"),
                           "median_span_seconds": t("median seconds", "giây trung vị"),
                           "cluster_count": t("clusters", "cluster"),
                       })
        right.dataframe(show, hide_index=True, width="stretch", height=430)
        lead = family.iloc[0]
        st.info(
            t(
                f"The largest family is **{lead['family_label']}** with **{int(lead['journeys']):,} journeys "
                f"({lead['platform_share']:.1%} of selected traffic). Behavioural friction is {lead['struggle_rate']:.1%}.",
                f"Family lớn nhất là **{lead['family_label']}**, có **{int(lead['journeys']):,} journey "
                f"({lead['platform_share']:.1%} traffic đang chọn). Behavioural friction là {lead['struggle_rate']:.1%}.",
            )
        )

    # ------------------------------------------------------------------ journey type leaderboard
    st.subheader(t("Top learned journey types", "Các journey type học được phổ biến nhất"))
    top = _label_columns(clusters[clusters["cluster"].ne(-1)].copy()).sort_values("journeys", ascending=False)
    if not top.empty:
        n = st.slider(t("Show top N", "Hiển thị top N"), 5, 30, 12, key="inference_top_n")
        show = top.head(n)[["platform", "cluster", "type_label", "family_label", "journeys", "platform_share",
                            "struggle_rate", "any_flag_rate", "median_steps", "median_span_seconds"]].rename(columns={
                                "type_label": t("journey type", "journey type"),
                                "family_label": t("family", "family"),
                                "platform": t("platform", "platform"),
                                "journeys": t("journeys", "journey"),
                                "platform_share": t("share", "tỷ lệ"),
                                "struggle_rate": t("struggle", "chật vật"),
                                "any_flag_rate": t("any flag", "mọi flag"),
                                "median_steps": t("steps", "bước"),
                                "median_span_seconds": t("seconds", "giây"),
                            })
        st.dataframe(show, hide_index=True, width="stretch", height=min(130 + n * 34, 620))

    st.divider()

    # ------------------------------------------------------------------ trend
    st.header(t("How the journey mix changes over time", "Cơ cấu journey thay đổi theo thời gian"))
    trend = _label_columns(_selected(load_summary_csv("daily_trend.csv"), platforms))
    lo, hi = _date_range(trend)
    if not trend.empty and lo and hi:
        selected_range = st.date_input(t("Date range", "Khoảng ngày"), value=(lo, hi), min_value=lo, max_value=hi, key="inference_dates")
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            trend = trend[(trend["date"] >= selected_range[0]) & (trend["date"] <= selected_range[1])]
        top_types = top.head(8)["type_label"].tolist() if not top.empty else []
        series = trend[trend["type_label"].isin(top_types)].groupby(["date", "type_label"], as_index=False)["journeys"].sum()
        if not series.empty:
            measure = st.segmented_control(
                t("Measure", "Chỉ số"), ["volume", "share"], default="volume",
                format_func=lambda x: t("journeys per day", "journey mỗi ngày") if x == "volume" else t("share of daily journeys", "tỷ lệ journey mỗi ngày"),
                key="inference_trend_measure",
            ) or "volume"
            if measure == "share":
                totals = trend.groupby("date", as_index=False)["journeys"].sum().rename(columns={"journeys": "total"})
                series = series.merge(totals, on="date", how="left")
                series["value"] = series["journeys"] / series["total"].replace(0, np.nan)
            else:
                series["value"] = series["journeys"]
            fig = px.line(series, x="date", y="value", color="type_label", markers=True, color_discrete_sequence=PALETTE)
            fig.update_layout(height=430, margin=dict(t=10, b=10), xaxis_title=None, yaxis_title=None, legend_title=None)
            if measure == "share":
                fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.caption(t("Daily rows are aggregates; they are not used as unique-customer totals.", "Các dòng theo ngày là aggregate; không dùng để tính unique customer."))

    # ------------------------------------------------------------------ friction / anomaly
    st.header(t("Where do customers get stuck?", "Khách hàng bị mắc kẹt ở đâu?"))
    friction = _selected(load_summary_csv("friction_summary.csv"), platforms)
    if not friction.empty:
        behavioral = friction[friction["flag_source"].eq("behavioral")].copy()
        model_flags = friction[friction["flag_source"].eq("model")].copy()
        a, b = st.columns(2)
        if not behavioral.empty:
            behavioral["label"] = behavioral["flag"].astype(str).str.replace("_", " ")
            fig = px.bar(behavioral.sort_values("journeys"), x="journeys", y="label", orientation="h", color_discrete_sequence=[PALETTE[3]])
            fig.update_layout(height=360, margin=dict(t=10, b=10), yaxis_title=None, xaxis_title=t("journeys", "journey"))
            a.plotly_chart(fig, width="stretch")
            a.caption(t("Behavioural flags are the clearest read of customer struggle.", "Behavioural flag là cách đọc rõ nhất về việc user chật vật."))
        if not model_flags.empty:
            model_flags["label"] = model_flags["flag"].astype(str).str.replace("_", " ")
            fig = px.bar(model_flags.sort_values("journeys"), x="journeys", y="label", orientation="h", color_discrete_sequence=[PALETTE[1]])
            fig.update_layout(height=360, margin=dict(t=10, b=10), yaxis_title=None, xaxis_title=t("journeys", "journey"))
            b.plotly_chart(fig, width="stretch")
            b.caption(t("Model flags include improbable transitions and novel archetypes; one journey may have multiple flags.", "Model flag gồm transition bất thường và archetype mới; một journey có thể có nhiều flag."))

    friction_type = _label_columns(_selected(load_summary_csv("friction_by_type.csv"), platforms))
    if not friction_type.empty:
        friction_type = friction_type.sort_values(["struggle_rate", "journeys"], ascending=[False, False]).head(15)
        friction_type["label"] = friction_type["type_label"] + " · " + friction_type["platform"].str.title()
        fig = px.bar(friction_type.sort_values("struggle_rate"), x="struggle_rate", y="label", orientation="h",
                     color="excess_seconds", color_continuous_scale=["#4C78A8", "#E45756"],
                     hover_data={"journeys": ":,", "median_steps": ":.0f", "median_span_seconds": ":.0f"})
        fig.update_layout(height=470, margin=dict(t=10, b=10), xaxis_tickformat=".0%", yaxis_title=None,
                          xaxis_title=t("behavioural struggle rate", "tỷ lệ behavioural struggle"))
        st.plotly_chart(fig, width="stretch")
        st.caption(t("Excess seconds estimates time above the clean median for the same journey type; it is directional, not causal.", "Excess seconds ước lượng thời gian vượt median của journey sạch cùng type; đây là chỉ báo định hướng, không phải quan hệ nhân quả."))

    anomaly = pd.DataFrame({
        "platform": platforms,
        "unresolved journeys": [int(load_inference_profile(p).get("geometric_anomalies", 0) or 0) for p in platforms],
        "severe anomalies": [int(load_inference_profile(p).get("severe_anomalies", 0) or 0) for p in platforms],
        "scored journeys": [int(load_inference_profile(p).get("rows", 0) or 0) for p in platforms],
    })
    anomaly["unresolved rate"] = anomaly["unresolved journeys"] / anomaly["scored journeys"].replace(0, np.nan)
    st.dataframe(anomaly, hide_index=True, width="stretch")

    # ------------------------------------------------------------------ next action + bounded explorer
    st.header(t("What is likely to happen next?", "Bước tiếp theo có khả năng xảy ra là gì?"))
    next_action = _selected(load_summary_csv("next_action_summary.csv"), platforms)
    if not next_action.empty:
        next_action = next_action.sort_values("journeys", ascending=False).head(15).copy()
        next_action["label"] = next_action["next_action"].fillna("<missing>").astype(str).str.slice(0, 100)
        fig = px.bar(next_action.sort_values("journeys"), x="journeys", y="label", orientation="h",
                     color="mean_confidence", color_continuous_scale=["#B279A2", "#4C78A8"],
                     hover_data={"action_share": ":.1%", "mean_confidence": ":.1%"})
        fig.update_layout(height=460, margin=dict(t=10, b=10), yaxis_title=None,
                          xaxis_title=t("journeys", "journey"), coloraxis_colorbar_title=t("confidence", "độ tin cậy"))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.header(t("Bounded journey examples", "Ví dụ journey có giới hạn"))
    st.caption(t("These rows are an audit sample only. All KPI and chart totals above come from aggregate tables.", "Các dòng này chỉ là sample để audit. Toàn bộ KPI và biểu đồ ở trên lấy từ bảng aggregate."))
    examples = pd.concat([load_inference(p) for p in platforms], ignore_index=True)
    if examples.empty:
        st.info(t("No bounded examples were generated.", "Chưa có bounded example."))
    else:
        type_options = ["all"] + sorted(examples.get("journey_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        selected_type = st.selectbox(t("Example type", "Type của example"), type_options, key="example_type")
        view = examples if selected_type == "all" else examples[examples["journey_type"].astype(str).eq(selected_type)]
        columns = [c for c in ["platform", "journey_id", "cluster", "journey_type", "business_family", "n_events_final", "span_seconds", "back_rate", "behavioral_friction_flags", "friction_flags", "entry_token", "exit_token"] if c in view.columns]
        st.dataframe(view[columns].head(200), hide_index=True, width="stretch", height=450)
