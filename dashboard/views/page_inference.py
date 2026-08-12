"""Page 4 — Production inference read as business results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import (
    PALETTE,
    explode_flags,
    fmt_int,
    fmt_pct,
    friction_meaning,
    friction_rule,
    friction_title,
    is_vi,
    kpi_row,
    load_inference,
    pretty_family,
    t,
)

SOURCE = {
    "android": "output/test/android_t5_1k_android_scored_named.csv",
    "ios": "output/test/ios_t5_1k_ios_scored_named.csv",
}


@st.cache_data(show_spinner=False)
def load_scope(platforms: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for p in platforms:
        d = load_inference(p).copy()
        d["platform"] = p
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # a handful of journeys carry an unparsable start timestamp; they cannot be placed on
    # any time axis, so they are excluded from the production read-out.
    df = df[df["start_ts"].notna()].copy()
    df["is_known"] = df["assignment_type"] != "unassigned_novel"
    df["has_struggle"] = df["behavioral_friction_flags"].fillna("").astype(str).str.len() > 0
    df["has_any_flag"] = df["friction_flags"].fillna("").astype(str).str.len() > 0
    df["date"] = df["start_ts"].dt.date
    df["hour"] = df["start_ts"].dt.hour
    return df


def add_display_names(df: pd.DataFrame) -> pd.DataFrame:
    """Journey type / family labels in the active language (kept out of the cache)."""
    df = df.copy()
    name_col = "cluster_name_vi" if is_vi() else "cluster_name"
    novel_label = t("Novel / unrecognised behaviour", "Hành vi mới / chưa nhận diện được")
    df["journey_type"] = np.where(df["is_known"], df[name_col].fillna(df["cluster_name"]), novel_label)
    df["family"] = np.where(
        df["is_known"],
        df["business_family"].map(pretty_family),
        t("novel", "hành vi mới"),
    )
    return df


def _excess_time(df: pd.DataFrame) -> float:
    """Extra seconds spent on struggling journeys vs the clean norm for their own type."""
    norm = (
        df[~df["has_struggle"]]
        .groupby("journey_type")["span_seconds"]
        .median()
        .rename("clean_median")
    )
    bad = df[df["has_struggle"]].join(norm, on="journey_type")
    excess = (bad["span_seconds"] - bad["clean_median"]).clip(lower=0)
    return float(excess.sum())


def render() -> None:
    st.title(t("4 · Production results — what users actually did", "4 · Kết quả trên production — user thực sự đã làm gì"))
    st.caption(
        t(
            "The fitted model scoring the held-out T5 production extract "
            "(`output/test/*_scored_named.csv`). Every number below is a business reading of "
            "journeys the model assigned without any labels.",
            "Model đã fit chấm điểm cho extract production T5 được giữ lại "
            "(`output/test/*_scored_named.csv`). Mọi con số bên dưới là cách đọc nghiệp vụ của các "
            "journey mà model tự gán, không dùng bất kỳ nhãn nào.",
        )
    )

    c1, c2 = st.columns([2, 3])
    platforms = c1.multiselect("Platform", ["android", "ios"], default=["android", "ios"])
    if not platforms:
        st.warning(t("Select at least one platform.", "Hãy chọn ít nhất một platform."))
        return
    df = add_display_names(load_scope(tuple(sorted(platforms))))

    dmin, dmax = df["date"].min(), df["date"].max()
    date_range = c2.date_input(t("Date range", "Khoảng thời gian"), value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]
    if df.empty:
        st.warning(t("No journeys in the selected range.", "Không có journey nào trong khoảng đã chọn."))
        return

    st.caption(t("Source files: ", "File nguồn: ") + " · ".join(f"`{SOURCE[p]}`" for p in platforms))

    # ------------------------------------------------------------------ headline
    n_j = len(df)
    n_sess = df["session_id"].nunique()
    n_cust = df["customer_id"].nunique()
    known = df["is_known"].mean()
    struggle = df["has_struggle"].mean()
    kpi_row(
        [
            (t("Sessions analysed", "Session đã phân tích"), fmt_int(n_sess), t("Distinct app sessions in the scored extract", "Số session khác nhau trong extract đã score")),
            (t("Journeys identified", "Journey nhận diện được"), fmt_int(n_j), t("Distinct user tasks carved out of those sessions", "Số task riêng biệt cắt ra từ các session đó")),
            (t("Customers", "Customer"), fmt_int(n_cust), None),
            (t("Journeys per session", "Journey mỗi session"), f"{n_j / max(n_sess, 1):.2f}", t("How many separate tasks a session contains on average", "Trung bình một session chứa bao nhiêu task riêng biệt")),
            (t("Recognised as a known journey", "Khớp journey type đã biết"), fmt_pct(known), t("Matched a learned journey type within its distance limit", "Khớp một journey type đã học, nằm trong distance limit")),
            (t("Journeys showing struggle", "Journey có dấu hiệu chật vật"), fmt_pct(struggle), t("Back-tracking, thrashing, looping or unusually slow", "Quay lui, nhảy qua lại, lặp vòng hoặc chậm bất thường")),
        ]
    )

    n_types = df.loc[df["is_known"], "journey_type"].nunique()
    n_clusters_seen = df.loc[df["is_known"], "cluster"].nunique()
    top3 = df.loc[df["is_known"], "journey_type"].value_counts(normalize=True).head(3)
    top3_str = ", ".join(f"**{name}** ({share:.0%})" for name, share in top3.items())
    st.success(
        t(
            f"**Headline.** {fmt_int(n_sess)} sessions from {fmt_int(n_cust)} customers break down into "
            f"**{fmt_int(n_j)} distinct user tasks** — about {n_j / max(n_sess, 1):.1f} per session. "
            f"**{known:.0%}** of them match a known journey type — {n_clusters_seen} learned clusters "
            f"that roll up into **{n_types} named journey types**; the rest is new behaviour worth "
            f"looking at. The three most common things people do are {top3_str}. "
            f"**{struggle:.0%}** of journeys show signs of the user struggling.",
            f"**Tóm tắt.** {fmt_int(n_sess)} session của {fmt_int(n_cust)} customer được tách thành "
            f"**{fmt_int(n_j)} task riêng biệt** — khoảng {n_j / max(n_sess, 1):.1f} task mỗi session. "
            f"**{known:.0%}** trong số đó khớp một journey type đã biết — {n_clusters_seen} cluster đã học, "
            f"gộp thành **{n_types} journey type có tên**; phần còn lại là hành vi mới đáng để xem xét. "
            f"Ba việc user làm nhiều nhất là {top3_str}. "
            f"**{struggle:.0%}** số journey có dấu hiệu user đang chật vật.",
        )
    )

    st.divider()

    # ------------------------------------------------------------------ what people do
    st.header(t("What are people doing in the app?", "User đang làm gì trong app?"))
    a, b = st.columns([3, 2])

    JC = t("journeys", "journey")
    SESS = t("sessions", "session")
    CUST = t("customers", "customer")
    STEPS = t("steps", "số bước")
    SECS = t("seconds", "giây")
    RATE = t("struggle rate", "tỷ lệ chật vật")
    SHARE = t("share of journeys", "tỷ lệ journey")
    FAM = t("family", "family")
    TYPE = t("what the user was doing", "user đang làm gì")

    by_type = (
        df.groupby(["family", "journey_type"], as_index=False)
        .agg(
            **{
                JC: ("journey_id", "count"),
                SESS: ("session_id", "nunique"),
                CUST: ("customer_id", "nunique"),
                STEPS: ("n_events_final", "median"),
                SECS: ("span_seconds", "median"),
                RATE: ("has_struggle", "mean"),
            }
        )
        .sort_values(JC, ascending=False)
    )
    by_type[SHARE] = by_type[JC] / by_type[JC].sum()

    fig = px.treemap(
        by_type,
        path=["family", "journey_type"],
        values=JC,
        color=RATE,
        color_continuous_scale=["#4C78A8", "#EECA3B", "#E45756"],
        hover_data={SESS: ":,", CUST: ":,", SECS: ":.0f"},
    )
    fig.update_layout(height=520, margin=dict(t=10, b=10, l=0, r=0), coloraxis_colorbar_title=RATE)
    a.plotly_chart(fig, width="stretch")
    a.caption(
        t(
            "Area = volume of journeys. Colour = share of those journeys where the user struggled.",
            "Diện tích = số lượng journey. Màu = tỷ lệ journey mà user gặp khó khăn.",
        )
    )

    fam = (
        df.groupby("family", as_index=False)
        .agg(**{JC: ("journey_id", "count"), CUST: ("customer_id", "nunique")})
        .sort_values(JC, ascending=False)
    )
    fam[SHARE] = fam[JC] / fam[JC].sum()
    fig = px.bar(fam.sort_values(JC), x=JC, y="family", orientation="h", color="family", color_discrete_sequence=PALETTE)
    fig.update_layout(height=380, margin=dict(t=10, b=10), showlegend=False, yaxis_title=None)
    b.plotly_chart(fig, width="stretch")
    lead = fam.iloc[0]
    b.markdown(
        t(
            f"**{str(lead['family']).title()}** dominates production traffic with "
            f"**{lead[SHARE]:.0%}** of all journeys, touching **{fmt_int(lead[CUST])}** customers. "
            "Read this as the app's real centre of gravity — not what the roadmap assumes it is.",
            f"**{str(lead['family']).capitalize()}** áp đảo traffic production với "
            f"**{lead[SHARE]:.0%}** tổng số journey, chạm tới **{fmt_int(lead[CUST])}** customer. "
            "Đây mới là trọng tâm thực sự của app — không hẳn là thứ roadmap đang giả định.",
        )
    )

    st.subheader(t("Top journey types", "Journey type phổ biến nhất"))
    top_n = st.slider(t("Show top N", "Hiển thị top N"), 5, 40, 15, key="topn")
    show = by_type.head(top_n)[["journey_type", "family", JC, SHARE, SESS, CUST, STEPS, SECS, RATE]].rename(
        columns={"journey_type": TYPE, "family": FAM}
    )
    st.dataframe(
        show, hide_index=True, width="stretch", height=min(80 + 35 * top_n, 640),
        column_config={
            SHARE: st.column_config.ProgressColumn(SHARE, format="%.2f%%", min_value=0.0, max_value=float(by_type[SHARE].max())),
            RATE: st.column_config.ProgressColumn(RATE, format="%.1f%%", min_value=0.0, max_value=1.0),
            SECS: st.column_config.NumberColumn(SECS, format="%.0f"),
            STEPS: st.column_config.NumberColumn(STEPS, format="%.0f"),
        },
    )

    st.divider()

    # ------------------------------------------------------------------ trend
    st.header(t("How the top journeys move over time", "Các journey top thay đổi thế nào theo thời gian"))
    st.caption(
        t(
            "Volume and mix of the leading journey types across the scored window. Volume answers "
            "\"how much\"; share answers \"what changed\" — a type can grow in share while total traffic falls.",
            "Số lượng và cơ cấu của các journey type dẫn đầu trong khoảng thời gian được score. Số lượng trả lời "
            "\"nhiều hay ít\"; tỷ lệ trả lời \"cái gì đã thay đổi\" — một type có thể tăng tỷ lệ ngay cả khi tổng traffic giảm.",
        )
    )

    tc1, tc2, tc3 = st.columns([2, 2, 3])
    trend_n = tc1.slider(t("Journey types to track", "Số journey type theo dõi"), 3, 12, 6, key="trend_n")
    freq_labels = {"D": t("daily", "theo ngày"), "W": t("weekly", "theo tuần")}
    freq = tc2.segmented_control(
        t("Granularity", "Độ mịn"), list(freq_labels), default="D",
        format_func=lambda k: freq_labels[k], key="trend_freq",
    ) or "D"
    measure_labels = {
        "volume": t("journeys per period", "số journey mỗi kỳ"),
        "share": t("share of journeys", "tỷ lệ trên tổng journey"),
        "struggle": t("struggle rate", "tỷ lệ chật vật"),
    }
    measure = tc3.segmented_control(
        t("Measure", "Chỉ số"), list(measure_labels), default="volume",
        format_func=lambda k: measure_labels[k], key="trend_measure",
    ) or "volume"

    # The extract carries a handful of stray timestamps months before the bulk of traffic,
    # so the trend is drawn over the dense window (central 98% of journeys) instead of the
    # full calendar range. Aggregates elsewhere on the page still use every journey.
    dense_lo, dense_hi = df["start_ts"].quantile([0.01, 0.99])
    dense = df[(df["start_ts"] >= dense_lo) & (df["start_ts"] <= dense_hi)]

    top_types = by_type.head(trend_n)["journey_type"].tolist()
    tdf = dense[dense["journey_type"].isin(top_types)].copy()
    period = pd.to_datetime(tdf["start_ts"]).dt.tz_convert(None).dt.to_period(freq).dt.start_time
    tdf = tdf.assign(period=period)
    all_period = pd.to_datetime(dense["start_ts"]).dt.tz_convert(None).dt.to_period(freq).dt.start_time

    PERIOD = t("period", "kỳ")
    series = (
        tdf.groupby(["period", "journey_type"], as_index=False)
        .agg(**{JC: ("journey_id", "count"), RATE: ("has_struggle", "mean")})
        .rename(columns={"period": PERIOD})
    )
    totals = dense.assign(period=all_period).groupby("period", as_index=False).size().rename(
        columns={"size": "total", "period": PERIOD}
    )
    series = series.merge(totals, on=PERIOD, how="left")
    series[SHARE] = series[JC] / series["total"]

    ycol = {"volume": JC, "share": SHARE, "struggle": RATE}[measure]
    fig = px.line(
        series.sort_values(PERIOD), x=PERIOD, y=ycol, color="journey_type",
        markers=True, color_discrete_sequence=PALETTE,
    )
    fig.update_layout(height=430, margin=dict(t=10, b=10), xaxis_title=None, legend_title=None)
    if measure in {"share", "struggle"}:
        fig.update_layout(yaxis_tickformat=".0%")
    if measure == "struggle":
        fig.add_hline(y=struggle, line_dash="dash", line_color="#888",
                      annotation_text=t("overall average", "trung bình chung"))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        t(
            f"Drawn over {dense_lo.date()} → {dense_hi.date()}, the window holding 98% of scored "
            f"journeys. A few stray timestamps months earlier are excluded from this chart only.",
            f"Vẽ trên khoảng {dense_lo.date()} → {dense_hi.date()}, tức khoảng chứa 98% số journey đã score. "
            f"Một vài timestamp lẻ tẻ từ nhiều tháng trước chỉ bị loại khỏi riêng biểu đồ này.",
        )
    )

    # Movers: split at the median timestamp so both halves hold the same number of journeys.
    # A calendar midpoint would be useless here — the extract is heavily back-loaded.
    mid_ts = dense["start_ts"].median()
    first = dense[dense["start_ts"] < mid_ts]
    second = dense[dense["start_ts"] >= mid_ts]
    if len(first) and len(second):
        f_share = first["journey_type"].value_counts(normalize=True)
        s_share = second["journey_type"].value_counts(normalize=True)
        vol = df["journey_type"].value_counts()
        movers = (
            pd.DataFrame({"first": f_share, "second": s_share})
            .fillna(0.0)
            .assign(journeys=vol)
            .query("journeys >= 200")
        )
        movers["delta"] = movers["second"] - movers["first"]
        movers = movers.sort_values("delta")
    else:
        movers = pd.DataFrame()
    if not movers.empty:
        pick = pd.concat([movers.head(5), movers.tail(5)]).drop_duplicates()
        m1, m2 = st.columns([3, 2])
        fig = go.Figure()
        fig.add_bar(
            x=pick["delta"], y=pick.index, orientation="h",
            marker_color=[PALETTE[3] if v < 0 else PALETTE[2] for v in pick["delta"]],
            text=[f"{v:+.1%}" for v in pick["delta"]],
        )
        fig.update_layout(
            height=380, margin=dict(t=30, b=10), xaxis_tickformat="+.1%",
            xaxis_title=t("change in share of journeys", "thay đổi tỷ lệ journey"),
            title=t("Risers and fallers, first half vs second half", "Tăng và giảm, nửa đầu so với nửa sau"),
        )
        m1.plotly_chart(fig, width="stretch")
        up = movers.iloc[-1]
        down = movers.iloc[0]
        m2.markdown(
            t(
                f"""
The window is split at its **median journey**, on {mid_ts.date()}, so both halves hold the same
number of journeys ({len(first):,} vs {len(second):,}) — {first['start_ts'].min().date()}–{first['start_ts'].max().date()}
against {second['start_ts'].min().date()}–{second['start_ts'].max().date()}.

- **Rising fastest:** *{movers.index[-1]}* — {up['first']:.1%} → {up['second']:.1%}
  of all journeys ({up['delta']:+.1%}).
- **Falling fastest:** *{movers.index[0]}* — {down['first']:.1%} → {down['second']:.1%}
  ({down['delta']:+.1%}).

Only journey types with at least 200 journeys are ranked, so a mover is a real shift in what
customers do rather than small-sample noise. A riser with a high struggle rate is the most
urgent thing on this page: more customers are walking into a flow that already hurts.
                """,
                f"""
Khoảng thời gian được chia tại **journey trung vị**, ngày {mid_ts.date()}, nên hai nửa có cùng số journey
({len(first):,} so với {len(second):,}) — {first['start_ts'].min().date()}–{first['start_ts'].max().date()}
so với {second['start_ts'].min().date()}–{second['start_ts'].max().date()}.

- **Tăng nhanh nhất:** *{movers.index[-1]}* — {up['first']:.1%} → {up['second']:.1%}
  trên tổng journey ({up['delta']:+.1%}).
- **Giảm nhanh nhất:** *{movers.index[0]}* — {down['first']:.1%} → {down['second']:.1%}
  ({down['delta']:+.1%}).

Chỉ xếp hạng các journey type có ít nhất 200 journey, nên một biến động ở đây là thay đổi thật trong
hành vi khách hàng chứ không phải nhiễu do mẫu nhỏ. Một journey type vừa tăng vừa có tỷ lệ chật vật cao
là thứ cấp bách nhất trên trang này: ngày càng nhiều khách hàng đi vào một flow vốn đã gây khó.
                """,
            )
        )

    st.divider()

    # ------------------------------------------------------------------ friction
    st.header(t("Where customers get stuck", "Khách hàng bị mắc kẹt ở đâu"))
    st.markdown(
        t(
            "Nobody tells us when an app frustrates them — they just struggle quietly and leave. "
            "These six signals are the fingerprints that struggle leaves in the data. Each one has a "
            "**plain meaning** and a **rule**: a journey is flagged only when it is more extreme than "
            "the vast majority of journeys the model saw while learning, so a flag always means "
            "*unusual for us*, not merely *long* or *complicated*.",
            "Không ai báo cho ta biết khi họ thấy app khó dùng — họ chỉ lặng lẽ vật lộn rồi rời đi. "
            "Sáu tín hiệu dưới đây là dấu vết mà sự vật lộn đó để lại trong dữ liệu. Mỗi tín hiệu có "
            "**ý nghĩa dễ hiểu** và một **quy tắc**: một journey chỉ bị gắn cờ khi nó cực đoan hơn đại đa số "
            "journey mà model đã thấy khi học. Nhờ vậy, gắn cờ luôn có nghĩa là *bất thường so với chính ta*, "
            "chứ không đơn thuần là *dài* hay *phức tạp*.",
        )
    )

    flags = explode_flags(df["behavioral_friction_flags"])
    model_flags = explode_flags(df["friction_flags"])
    all_flags = model_flags.add(flags, fill_value=0).sort_values(ascending=False)

    excess = _excess_time(df)
    n_struggle = int(df["has_struggle"].sum())

    kpi_row(
        [
            (
                t("Journeys with a struggle sign", "Journey có dấu hiệu vật lộn"),
                fmt_pct(struggle),
                t("At least one of the four behavioural signals", "Có ít nhất một trong bốn tín hiệu hành vi"),
            ),
            (
                t("Including model-based checks", "Tính cả kiểm tra dựa trên model"),
                fmt_pct(df["has_any_flag"].mean()),
                t("Adds unusual paths and unknown behaviour", "Cộng thêm đường đi bất thường và hành vi chưa từng thấy"),
            ),
            (
                t("Customer time lost", "Thời gian khách hàng mất thêm"),
                t(f"{excess / 3600:,.0f} hours", f"{excess / 3600:,.0f} giờ"),
                t(
                    "Time above the clean median of the same journey type — not just 'long journeys are long'",
                    "Phần vượt trên mức trung vị của journey sạch cùng loại — không phải chỉ là 'journey dài thì lâu'",
                ),
            ),
            (
                t("Per struggling journey", "Mỗi journey vật lộn"),
                t(f"+{excess / max(n_struggle, 1):.0f} seconds", f"+{excess / max(n_struggle, 1):.0f} giây"),
                t("Avoidable effort, on average", "Công sức lẽ ra tránh được, tính trung bình"),
            ),
        ]
    )

    SIG = t("signal", "tín hiệu")
    MEAN = t("what it means", "nghĩa là gì")
    RULE = t("when we flag it", "khi nào bị gắn cờ")
    PCT = t("% of journeys", "% trên tổng journey")

    fd = pd.DataFrame({"flag": all_flags.index, JC: all_flags.to_numpy()})
    fd[SIG] = fd["flag"].map(friction_title)
    fd[MEAN] = fd["flag"].map(friction_meaning)
    fd[RULE] = fd["flag"].map(lambda f: friction_rule(f, platforms))
    fd[PCT] = fd[JC] / n_j

    fig = px.bar(
        fd.sort_values(JC), x=JC, y=SIG, orientation="h",
        color_discrete_sequence=[PALETTE[3]], hover_data=[MEAN],
    )
    fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        fd[[SIG, MEAN, RULE, JC, PCT]],
        hide_index=True, width="stretch",
        column_config={
            SIG: st.column_config.TextColumn(SIG, width="small"),
            MEAN: st.column_config.TextColumn(MEAN, width="large"),
            RULE: st.column_config.TextColumn(RULE, width="large"),
            PCT: st.column_config.ProgressColumn(PCT, format="%.1f%%", min_value=0.0, max_value=float(fd[PCT].max())),
        },
    )
    st.caption(
        t(
            "The cut-offs are learned from the training data, not set by hand, and they are fitted "
            "separately for each platform — so \"unusual\" always means unusual for this app, on this "
            "platform. One journey can raise several signals at once; the counts above therefore add "
            "up to more than the number of flagged journeys.",
            "Các ngưỡng được học từ dữ liệu train chứ không do người đặt tay, và được fit riêng cho từng "
            "platform — nên \"bất thường\" luôn có nghĩa là bất thường với chính app này, trên chính platform đó. "
            "Một journey có thể bật nhiều tín hiệu cùng lúc, nên tổng các con số ở trên lớn hơn số journey "
            "thực sự bị gắn cờ.",
        )
    )

    st.subheader(t("Which tasks frustrate customers most", "Task nào khiến khách hàng bực nhất"))
    st.caption(
        t(
            "For each task, the share of attempts that carried a struggle sign. Only tasks with enough "
            "volume to be worth a sprint are ranked — this is the fix list, in priority order.",
            "Với mỗi task, đây là tỷ lệ lượt thực hiện có dấu hiệu vật lộn. Chỉ xếp hạng những task đủ nhiều "
            "để đáng bỏ một sprint ra sửa — đây chính là danh sách cần sửa, theo thứ tự ưu tiên.",
        )
    )
    min_vol = st.slider(t("Minimum journeys for a type to qualify", "Số journey tối thiểu để một type được xét"), 50, 2000, 300, step=50)
    worst = by_type[by_type[JC] >= min_vol].sort_values(RATE, ascending=False).head(12)
    if worst.empty:
        st.info(t("No journey type reaches that volume in the current selection.", "Không journey type nào đạt ngưỡng đó trong lựa chọn hiện tại."))
    else:
        fig = go.Figure()
        fig.add_bar(
            x=worst[RATE], y=worst["journey_type"], orientation="h",
            marker_color=PALETTE[3], name=RATE,
            text=[t(f"{v:.0%} of {int(n):,}", f"{v:.0%} trong {int(n):,}") for v, n in zip(worst[RATE], worst[JC])],
        )
        fig.add_vline(x=struggle, line_dash="dash", line_color="#888", annotation_text=t("overall average", "trung bình chung"))
        fig.update_layout(
            height=460, margin=dict(t=30, b=10), xaxis_tickformat=".0%",
            yaxis=dict(autorange="reversed"),
            xaxis_title=t("share of journeys with a struggle signal", "tỷ lệ journey có tín hiệu chật vật"),
        )
        st.plotly_chart(fig, width="stretch")

        w = worst.iloc[0]
        st.warning(
            t(
                f"**Priority fix:** *{w['journey_type']}* — "
                f"{w[RATE]:.0%} of its {int(w[JC]):,} journeys show struggle, "
                f"affecting {int(w[CUST]):,} customers. Typical run: "
                f"{w[STEPS]:.0f} steps in {w[SECS]:.0f} s.",
                f"**Ưu tiên sửa:** *{w['journey_type']}* — "
                f"{w[RATE]:.0%} trong {int(w[JC]):,} journey của nó có dấu hiệu chật vật, "
                f"ảnh hưởng {int(w[CUST]):,} customer. Một lượt điển hình: "
                f"{w[STEPS]:.0f} bước trong {w[SECS]:.0f} giây.",
            )
        )

        detail = st.selectbox(t("Look inside one task", "Xem sâu vào một task"), worst["journey_type"].tolist())
        sub = df[df["journey_type"] == detail]
        sf = explode_flags(sub["behavioral_friction_flags"]).add(explode_flags(sub["friction_flags"]), fill_value=0).sort_values(ascending=False)
        d1, d2 = st.columns([2, 3])
        d1.markdown(t("**How customers struggle here**", "**Khách hàng vật lộn kiểu gì ở đây**"))
        d1.dataframe(
            pd.DataFrame(
                {
                    SIG: [friction_title(f) for f in sf.index],
                    JC: sf.to_numpy().astype(int),
                    PCT: sf.to_numpy() / max(len(sub), 1),
                }
            ),
            hide_index=True, width="stretch",
            column_config={PCT: st.column_config.ProgressColumn(PCT, format="%.0f%%", min_value=0.0, max_value=1.0)},
        )
        d2.markdown(
            t(
                "**Where these journeys end** — the last screen the customer saw before finishing or giving up",
                "**Các journey này kết thúc ở đâu** — màn hình cuối khách hàng nhìn thấy trước khi xong việc hoặc bỏ cuộc",
            )
        )
        d2.dataframe(
            sub["exit_token"].value_counts().head(8).rename_axis(t("exit step", "bước kết thúc")).reset_index(name=JC),
            hide_index=True, width="stretch",
        )
        with st.expander(t("See real customers struggling, step by step", "Xem khách hàng thật vật lộn, từng bước một")):
            st.caption(
                t(
                    "The five slowest struggling attempts at this task. Read the step list as the screens "
                    "the customer actually moved through, in order.",
                    "Năm lượt vật lộn chậm nhất của task này. Hãy đọc danh sách bước như đúng những màn hình "
                    "khách hàng đã đi qua, theo thứ tự.",
                )
            )
            ex = sub[sub["has_struggle"]].nlargest(5, "span_seconds")
            for _, r in ex.iterrows():
                raw_flags = str(r["behavioral_friction_flags"] or r["friction_flags"] or "")
                nice = " · ".join(friction_title(f) for f in raw_flags.split("|") if f)
                st.markdown(
                    t(
                        f"**{nice}** — {r['n_events_final']} steps in {r['span_seconds']:.0f}s "
                        f"(`{r['journey_id']}`, flags `{raw_flags}`)",
                        f"**{nice}** — {r['n_events_final']} bước trong {r['span_seconds']:.0f} giây "
                        f"(`{r['journey_id']}`, flags `{raw_flags}`)",
                    )
                )
                st.code(str(r["sequence"]).replace(" -> ", "\n→ "), language=None)

    st.divider()

    # ------------------------------------------------------------------ novel
    st.header(t("New behaviour the model has never seen", "Hành vi mới mà model chưa từng thấy"))
    novel = df[~df["is_known"]]
    a, b = st.columns([2, 3])
    a.metric(t("Novel journeys", "Journey mới"), fmt_int(len(novel)), t(f"{len(novel) / n_j:.1%} of all journeys", f"{len(novel) / n_j:.1%} tổng số journey"))
    a.metric(t("Customers involved", "Customer liên quan"), fmt_int(novel["customer_id"].nunique()))
    a.metric(t("Median steps", "Số bước trung vị"), f"{novel['n_events_final'].median():.0f}" if len(novel) else "—")
    b.markdown(
        t(
            """
These journeys fell outside the distance limit of **every** learned journey type. In production
that is a signal, not an error — it is where a new feature, a broken release, an unusual
support pattern or an abuse pattern shows up first.

**Operational use:** watch this share over time. A stable few percent is healthy churn in user
behaviour; a sudden jump on a specific entry screen means something changed in the app.
            """,
            """
Những journey này nằm ngoài distance limit của **mọi** journey type đã học. Trên production đó là
một tín hiệu, không phải lỗi — đây là nơi một feature mới, một bản release lỗi, một dạng yêu cầu
hỗ trợ bất thường hay một hành vi lạm dụng sẽ lộ diện đầu tiên.

**Cách dùng vận hành:** theo dõi tỷ lệ này theo thời gian. Vài phần trăm ổn định là mức biến động
lành mạnh; một cú nhảy đột ngột gắn với một entry screen cụ thể nghĩa là có gì đó vừa thay đổi trong app.
            """,
        )
    )
    if len(novel):
        n1, n2 = st.columns(2)
        n1.markdown(t("**Where novel journeys start**", "**Journey mới bắt đầu ở đâu**"))
        n1.dataframe(
            novel["entry_token"].value_counts().head(10).rename_axis(t("entry step", "bước mở đầu")).reset_index(name=JC),
            hide_index=True, width="stretch",
        )
        n2.markdown(t("**Where they end**", "**Chúng kết thúc ở đâu**"))
        n2.dataframe(
            novel["exit_token"].value_counts().head(10).rename_axis(t("exit step", "bước kết thúc")).reset_index(name=JC),
            hide_index=True, width="stretch",
        )
        NSHARE = t("novel share", "tỷ lệ journey mới")
        DATE = t("date", "ngày")
        trend = novel.groupby("date", as_index=False).size().rename(columns={"size": "novel", "date": DATE})
        total_trend = df.groupby("date", as_index=False).size().rename(columns={"size": "all", "date": DATE})
        trend = trend.merge(total_trend, on=DATE)
        trend[NSHARE] = trend["novel"] / trend["all"]
        fig = px.line(trend, x=DATE, y=NSHARE, markers=True, color_discrete_sequence=[PALETTE[3]])
        fig.update_layout(height=280, margin=dict(t=10, b=10), yaxis_tickformat=".0%", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            t(
                "Daily share of journeys the model could not recognise — the drift monitor.",
                "Tỷ lệ journey model không nhận diện được theo ngày — chỉ báo theo dõi drift.",
            )
        )

    st.divider()

    # ------------------------------------------------------------------ next action
    st.header(t("What users do next", "User sẽ làm gì tiếp theo"))
    st.caption(
        t(
            "For each journey the model predicts the single most likely next step from the fitted "
            "transition model. Aggregated, this is a demand signal for shortcuts, deep links and prompts.",
            "Với mỗi journey, model dự đoán bước kế tiếp khả dĩ nhất dựa trên transition model đã fit. "
            "Khi tổng hợp lại, đây là tín hiệu nhu cầu để làm shortcut, deep link và gợi ý.",
        )
    )
    nxt = df[df["effective_next_action"].notna()]
    if len(nxt):
        NA = t("next action", "hành động kế tiếp")
        CONF = t("confidence", "độ tin cậy")
        agg = (
            nxt.groupby("effective_next_action", as_index=False)
            .agg(**{JC: ("journey_id", "count"), CONF: ("effective_next_action_share", "mean")})
            .sort_values(JC, ascending=False)
            .head(15)
            .rename(columns={"effective_next_action": NA})
        )
        fig = px.bar(agg.sort_values(JC), x=JC, y=NA, orientation="h", color=CONF, color_continuous_scale=["#B0C4DE", PALETTE[0]])
        fig.update_layout(height=520, margin=dict(t=10, b=10), yaxis_title=None, coloraxis_colorbar_title=CONF)
        fig.update_yaxes(tickfont_size=10)
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            t(
                f"The single most anticipated next step is `{agg.iloc[0][NA]}` "
                f"({fmt_int(agg.iloc[0][JC])} journeys, average confidence "
                f"{agg.iloc[0][CONF]:.0%}). Anywhere confidence is high and the step count "
                "to reach it is large, there is a shortcut worth building.",
                f"Bước kế tiếp được dự đoán nhiều nhất là `{agg.iloc[0][NA]}` "
                f"({fmt_int(agg.iloc[0][JC])} journey, độ tin cậy trung bình "
                f"{agg.iloc[0][CONF]:.0%}). Ở đâu độ tin cậy cao mà số bước để tới đó lại nhiều, "
                "ở đó đáng làm một shortcut.",
            )
        )

    st.divider()

    # ------------------------------------------------------------------ timing
    st.header(t("When it happens", "Diễn ra vào lúc nào"))
    DATE = t("date", "ngày")
    HOUR = t("hour (UTC)", "giờ (UTC)")
    a, b = st.columns([3, 2])
    daily = dense.groupby(["date", "family"], as_index=False).size().rename(columns={"size": JC, "date": DATE})
    fig = px.area(daily, x=DATE, y=JC, color="family", color_discrete_sequence=PALETTE)
    fig.update_layout(height=340, margin=dict(t=10, b=10), xaxis_title=None, legend_title=None)
    a.plotly_chart(fig, width="stretch")
    a.caption(
        t(
            f"Daily journey volume by business family, over the dense window {dense_lo.date()} → {dense_hi.date()}.",
            f"Lượng journey theo ngày chia theo business family, trong khoảng dày dữ liệu {dense_lo.date()} → {dense_hi.date()}.",
        )
    )

    heat = df.groupby(["family", "hour"], as_index=False).size().rename(columns={"size": JC})
    heat["share"] = heat[JC] / heat.groupby("family")[JC].transform("sum")
    pivot = heat.pivot(index="family", columns="hour", values="share").fillna(0)
    fig = px.imshow(pivot, color_continuous_scale=["#F5F5F5", PALETTE[0]], aspect="auto")
    fig.update_layout(height=340, margin=dict(t=10, b=10), xaxis_title=HOUR, yaxis_title=None)
    b.plotly_chart(fig, width="stretch")
    b.caption(
        t(
            "Hour-of-day profile per family, normalised within each family. Local time is UTC+7.",
            "Phân bố theo giờ của từng family, chuẩn hóa trong nội bộ family. Giờ Việt Nam là UTC+7.",
        )
    )

    st.divider()

    # ------------------------------------------------------------------ session view
    st.header(t("Session composition", "Cấu trúc của một session"))
    per_sess = df.groupby("session_id").agg(journeys=("journey_id", "count"), families=("family", "nunique"))
    a, b, c = st.columns(3)
    a.metric(t("Sessions with more than one task", "Session có nhiều hơn một task"), fmt_pct((per_sess["journeys"] > 1).mean()))
    b.metric(t("Sessions spanning several families", "Session trải nhiều family"), fmt_pct((per_sess["families"] > 1).mean()))
    c.metric(t("Busiest session", "Session dày đặc nhất"), t(f"{int(per_sess['journeys'].max())} tasks", f"{int(per_sess['journeys'].max())} task"))
    dist = per_sess["journeys"].clip(upper=10).value_counts().sort_index()
    fig = px.bar(
        x=dist.index.astype(str), y=dist.to_numpy(), color_discrete_sequence=[PALETTE[0]],
        labels={
            "x": t("tasks in the session (10 = 10+)", "số task trong session (10 = 10+)"),
            "y": t("sessions", "session"),
        },
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        t(
            "This is the practical argument for journeys over sessions: a large share of sessions "
            "contains **more than one unrelated task**, so any metric computed per session mixes them together.",
            "Đây là lập luận thực tế cho việc dùng journey thay vì session: một tỷ lệ lớn session chứa "
            "**nhiều hơn một task không liên quan**, nên mọi chỉ số tính theo session đều đang trộn chúng lại.",
        )
    )

    st.divider()

    # ------------------------------------------------------------------ explorer
    st.header(t("Journey explorer", "Khám phá journey"))
    e1, e2, e3 = st.columns([3, 2, 2])
    ALL = t("(all)", "(tất cả)")
    type_pick = e1.selectbox(t("Journey type", "Journey type"), [ALL] + by_type["journey_type"].tolist())
    only_struggle = e2.checkbox(t("Only journeys with struggle", "Chỉ journey có chật vật"), value=False)
    only_novel = e3.checkbox(t("Only novel journeys", "Chỉ journey mới"), value=False)
    view = df
    if type_pick != ALL:
        view = view[view["journey_type"] == type_pick]
    if only_struggle:
        view = view[view["has_struggle"]]
    if only_novel:
        view = view[~view["is_known"]]
    st.caption(t(f"{len(view):,} journeys match.", f"{len(view):,} journey khớp điều kiện."))
    st.dataframe(
        view[[
            "journey_id", "platform", "start_ts", "journey_type", "family", "n_events_final",
            "span_seconds", "back_rate", "behavioral_friction_flags", "effective_next_action", "sequence",
        ]].head(500),
        hide_index=True, width="stretch", height=420,
    )
    st.caption(
        t(
            "Showing at most 500 rows. `sequence` is the cleaned token path — the audit trail for "
            "every classification on this page.",
            "Hiển thị tối đa 500 dòng. `sequence` là path token đã làm sạch — bằng chứng kiểm chứng cho "
            "mọi phân loại trên trang này.",
        )
    )
