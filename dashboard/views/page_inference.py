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
    friction_explain,
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

    # ------------------------------------------------------------------ friction
    st.header(t("Where users struggle", "User gặp khó ở đâu"))
    flags = explode_flags(df["behavioral_friction_flags"])
    model_flags = explode_flags(df["friction_flags"])
    all_flags = model_flags.add(flags, fill_value=0).sort_values(ascending=False)

    excess = _excess_time(df)
    n_struggle = int(df["has_struggle"].sum())
    SIG = t("signal", "tín hiệu")
    MEAN = t("meaning", "nghĩa là gì")
    a, b = st.columns([3, 2])
    fd = pd.DataFrame({SIG: all_flags.index, JC: all_flags.to_numpy()})
    fd[MEAN] = fd[SIG].map(lambda s: friction_explain().get(s, ""))
    fig = px.bar(fd.sort_values(JC), x=JC, y=SIG, orientation="h", color_discrete_sequence=[PALETTE[3]], hover_data=[MEAN])
    fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    b.markdown(
        t(
            f"""
**{fmt_pct(struggle)}** of journeys carry at least one behavioural struggle signal, and
**{fmt_pct(df['has_any_flag'].mean())}** carry a signal once model-based checks are included.

Estimated **extra time users spent because of struggle: {excess / 3600:,.0f} hours**
across {fmt_int(n_struggle)} journeys — measured as time above the clean
median for the *same* journey type, so it is not just "long journeys are long".

Per struggling journey that is about **{excess / max(n_struggle, 1):.0f} extra seconds**
of avoidable effort.
            """,
            f"""
**{fmt_pct(struggle)}** số journey mang ít nhất một tín hiệu chật vật về hành vi, và
**{fmt_pct(df['has_any_flag'].mean())}** có tín hiệu nếu tính cả các kiểm tra dựa trên model.

Ước tính **thời gian phát sinh thêm do chật vật: {excess / 3600:,.0f} giờ**
trên {fmt_int(n_struggle)} journey — đo bằng phần vượt trên mức trung vị của journey "sạch"
thuộc *cùng* journey type, nên không phải chỉ là "journey dài thì lâu".

Trung bình mỗi journey chật vật tốn thêm khoảng **{excess / max(n_struggle, 1):.0f} giây**
công sức lẽ ra tránh được.
            """,
        )
    )
    st.dataframe(fd[[SIG, JC, MEAN]], hide_index=True, width="stretch")

    st.subheader(t("Which journeys hurt the most", "Journey nào gây khó chịu nhất"))
    st.caption(
        t(
            "Ranked by struggle rate among journey types with enough volume to act on. "
            "These are the concrete fix candidates.",
            "Xếp hạng theo tỷ lệ chật vật, chỉ xét các journey type đủ lớn để hành động. "
            "Đây là danh sách ứng viên cần sửa cụ thể.",
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

        detail = st.selectbox(t("Break down a journey type", "Phân tích sâu một journey type"), worst["journey_type"].tolist())
        sub = df[df["journey_type"] == detail]
        sf = explode_flags(sub["behavioral_friction_flags"]).add(explode_flags(sub["friction_flags"]), fill_value=0).sort_values(ascending=False)
        d1, d2 = st.columns([2, 3])
        d1.dataframe(pd.DataFrame({SIG: sf.index, JC: sf.to_numpy().astype(int)}), hide_index=True, width="stretch")
        d2.markdown(
            t(
                "**Where these journeys end** — the last screen users see before giving up or finishing",
                "**Các journey này kết thúc ở đâu** — màn hình cuối user thấy trước khi bỏ cuộc hoặc hoàn tất",
            )
        )
        d2.dataframe(
            sub["exit_token"].value_counts().head(8).rename_axis(t("exit step", "bước kết thúc")).reset_index(name=JC),
            hide_index=True, width="stretch",
        )
        with st.expander(t("Example struggling journeys (raw step sequences)", "Ví dụ journey chật vật (chuỗi bước thô)")):
            ex = sub[sub["has_struggle"]].nlargest(5, "span_seconds")
            for _, r in ex.iterrows():
                st.markdown(
                    t(
                        f"`{r['journey_id']}` · {r['n_events_final']} steps · {r['span_seconds']:.0f}s · "
                        f"flags: `{r['behavioral_friction_flags'] or r['friction_flags']}`",
                        f"`{r['journey_id']}` · {r['n_events_final']} bước · {r['span_seconds']:.0f} giây · "
                        f"flags: `{r['behavioral_friction_flags'] or r['friction_flags']}`",
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
    daily = df.groupby(["date", "family"], as_index=False).size().rename(columns={"size": JC, "date": DATE})
    fig = px.area(daily, x=DATE, y=JC, color="family", color_discrete_sequence=PALETTE)
    fig.update_layout(height=340, margin=dict(t=10, b=10), xaxis_title=None, legend_title=None)
    a.plotly_chart(fig, width="stretch")
    a.caption(t("Daily journey volume by business family.", "Lượng journey theo ngày, chia theo business family."))

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
