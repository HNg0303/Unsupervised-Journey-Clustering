"""Page 3 — The journey types the model learned."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import (
    PALETTE,
    fmt_int,
    fmt_pct,
    is_vi,
    kpi_row,
    load_cluster_names,
    load_run_csv,
    load_shareholder_catalog,
    load_shareholder_catalog_vi,
    load_train_run,
    pretty_family,
    t,
)


@st.cache_data(show_spinner=False)
def build_cluster_table(platform: str) -> pd.DataFrame:
    catalog = load_run_csv(f"{platform}_report_cluster_catalog.csv")
    names = load_cluster_names()
    names = names[names["platform"] == platform].rename(columns={"cluster_id": "cluster"})
    df = catalog.merge(
        names[
            ["cluster", "cluster_name", "cluster_name_vi", "business_family",
             "business_family_vi", "naming_confidence", "ngrams"]
        ],
        on="cluster",
        how="left",
    )
    df["business_family"] = df["business_family"].fillna("unknown")
    df["cluster_name"] = df["cluster_name"].fillna("Unnamed cluster")
    df["cluster_name_vi"] = df["cluster_name_vi"].fillna(df["cluster_name"])
    df["is_noise"] = df["cluster"] == -1
    return df


@st.cache_data(show_spinner=False)
def readable_steps(platform: str, vi: bool) -> dict:
    cat = load_shareholder_catalog_vi() if vi else load_shareholder_catalog()
    if not cat:
        cat = load_shareholder_catalog()
    out = {}
    for p in cat.get("platforms", []):
        if p.get("platform") != platform:
            continue
        for c in p.get("clusters", []):
            rep = c.get("representative_journey", {})
            # the EN catalog stores readable steps as a list, the VI one as an arrow-joined string
            raw_steps = rep.get("readable_steps", [])
            if isinstance(raw_steps, str):
                raw_steps = [s.strip() for s in raw_steps.replace(" -> ", " → ").split("→") if s.strip()]
            out[int(c["cluster_id"])] = {
                "steps": raw_steps,
                "raw_path": rep.get("raw_path", ""),
                "journey_id": rep.get("journey_id"),
            }
    return out


def render() -> None:
    st.title(t("3 · The journey types the model learned", "3 · Các journey type mà model học được"))
    st.caption(
        t(
            "Clusters are learned unsupervised from token sequences; the names are readable "
            "interpretations of the cluster's medoid path, entry/exit tokens and top n-grams — "
            "they are not labels that existed anywhere in the source system.",
            "Cluster được học không giám sát từ chuỗi token; tên gọi là cách diễn giải dễ đọc dựa trên "
            "medoid path, token entry/exit và các n-gram đặc trưng của cluster — chúng không phải nhãn "
            "có sẵn ở bất kỳ đâu trong hệ thống nguồn.",
        )
    )

    platform = st.radio(
        t("Platform model", "Model theo platform"), ["android", "ios"], horizontal=True, format_func=str.title
    )
    df = build_cluster_table(platform)
    if df.empty:
        st.warning(t("No cluster catalog found for this platform.", "Không tìm thấy cluster catalog cho platform này."))
        return

    vi = is_vi()
    name_col = "cluster_name_vi" if vi else "cluster_name"
    df["display_name"] = df[name_col]

    run = load_train_run()
    hdb = run["platforms"].get(platform, {}).get("run_config", {}).get("hdbscan", {})
    steps = readable_steps(platform, vi)

    named = df[~df["is_noise"]]
    noise = df[df["is_noise"]]
    noise_share = float(noise["share"].iloc[0]) if len(noise) else 0.0

    kpi_row(
        [
            (t("Clusters learned", "Số cluster học được"), fmt_int(len(named)), t("Excluding the unclassified bucket", "Không tính nhóm chưa phân loại")),
            (
                t("Distinct journey names", "Số tên journey khác nhau"),
                fmt_int(named["display_name"].nunique()),
                t(
                    "Several clusters can describe the same business journey at different lengths or entry points",
                    "Nhiều cluster có thể mô tả cùng một journey nghiệp vụ ở độ dài hoặc điểm vào khác nhau",
                ),
            ),
            (t("Business families", "Số business family"), fmt_int(named["business_family"].nunique()), None),
            (t("Journeys in catalog", "Journey trong catalog"), fmt_int(df["size"].sum()), None),
            (t("Largest type", "Type lớn nhất"), fmt_pct(named["share"].max()), t("Share of all training journeys", "Tỷ lệ trên tổng journey khi train")),
            (t("Unclassified", "Chưa phân loại"), fmt_pct(noise_share), t("Journeys HDBSCAN left as noise while fitting", "Journey bị HDBSCAN để là noise khi fit")),
        ]
    )

    st.info(
        t(
            f"**{len(named)} clusters** emerged from behaviour alone, rolling up into "
            f"**{named['display_name'].nunique()} named journey types**. They are long-tailed: the "
            f"largest cluster covers only {named['share'].max():.1%} of journeys, and the top 10 "
            f"together cover {named.nlargest(10, 'share')['share'].sum():.1%}. This is the expected "
            "shape for a super-app — no single dominant flow, many small specific tasks.",
            f"**{len(named)} cluster** xuất hiện chỉ từ hành vi, gộp lại thành "
            f"**{named['display_name'].nunique()} journey type có tên**. Phân phối đuôi dài: cluster lớn nhất "
            f"cũng chỉ chiếm {named['share'].max():.1%} tổng journey, và top 10 cộng lại mới được "
            f"{named.nlargest(10, 'share')['share'].sum():.1%}. Đây đúng là hình dạng của một super-app — "
            "không có flow nào áp đảo, mà rất nhiều task nhỏ và cụ thể.",
        )
    )

    st.divider()

    # ------------------------------------------------------------------ families
    st.subheader(t("What the app is used for, by business family", "App được dùng để làm gì, theo business family"))
    JC = t("journeys", "journey")
    TC = t("types", "số type")
    FC = t("family", "family")
    SC = t("share", "tỷ lệ")
    fam = (
        named.groupby("business_family", as_index=False)
        .agg(**{JC: ("size", "sum"), TC: ("cluster", "count")})
        .sort_values(JC, ascending=False)
    )
    fam[FC] = fam["business_family"].map(pretty_family)
    fam[SC] = fam[JC] / fam[JC].sum()
    a, b = st.columns([3, 2])
    fig = px.bar(fam.sort_values(JC), x=JC, y=FC, orientation="h", color=FC, color_discrete_sequence=PALETTE, text=TC)
    fig.update_traces(texttemplate="%{text} " + TC)
    fig.update_layout(height=420, margin=dict(t=10, b=10), showlegend=False, yaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    b.dataframe(
        fam[[FC, TC, JC, SC]],
        hide_index=True, width="stretch",
        column_config={SC: st.column_config.ProgressColumn(SC, format="%.1f%%", min_value=0.0, max_value=float(fam[SC].max()))},
    )

    # ------------------------------------------------------------------ landscape
    st.subheader(t("Cluster landscape", "Bản đồ cluster"))
    st.caption(
        t(
            "Each bubble is one learned journey type. Position tells you its *behavioural signature*: "
            "how long it runs, how tap-heavy it is; size is how common it is; a high back-rate "
            "(red end) means users struggle inside it.",
            "Mỗi bong bóng là một journey type đã học. Vị trí cho biết *chữ ký hành vi* của nó: dài bao nhiêu "
            "bước, nặng thao tác tap tới mức nào; kích thước thể hiện độ phổ biến; back rate cao (phía đỏ) "
            "nghĩa là user chật vật bên trong journey đó.",
        )
    )
    plot = named.copy()
    plot[FC] = plot["business_family"].map(pretty_family)
    fig = px.scatter(
        plot,
        x="median_length",
        y="mean_action_ratio",
        size="size",
        color="mean_back_rate",
        color_continuous_scale=["#4C78A8", "#EECA3B", "#E45756"],
        hover_name="display_name",
        hover_data={
            "cluster": True, FC: True, "size": ":,", "share": ":.2%",
            "median_span_s": ":.0f", "mean_back_rate": ":.3f", "median_length": False,
            "mean_action_ratio": ":.2f",
        },
        size_max=45,
    )
    fig.update_layout(
        height=520, margin=dict(t=10, b=10),
        xaxis_title=t("median journey length (events)", "độ dài journey trung vị (event)"),
        yaxis_title=t("mean action ratio (taps ÷ steps)", "action ratio trung bình (tap ÷ bước)"),
        coloraxis_colorbar_title=t("back rate", "back rate"),
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # ------------------------------------------------------------------ leaderboard
    st.subheader(t("Journey type catalogue", "Catalogue journey type"))
    c1, c2, c3 = st.columns([2, 2, 2])
    fam_pick = c1.multiselect(t("Business family", "Business family"), sorted(named["business_family"].unique()), format_func=pretty_family)
    conf_pick = c2.multiselect(t("Naming confidence", "Độ tin cậy khi đặt tên"), sorted(named["naming_confidence"].dropna().unique()))
    sort_labels = {
        "size": t("Volume", "Số lượng"),
        "share": t("Share", "Tỷ lệ"),
        "median_span_s": t("Duration", "Thời lượng"),
        "mean_back_rate": t("Back rate (struggle)", "Back rate (mức chật vật)"),
        "mean_revisit_ratio": t("Revisit ratio", "Revisit ratio"),
        "loop_journey_share": t("Loop share", "Tỷ lệ có vòng lặp"),
    }
    sort_by = c3.selectbox(t("Sort by", "Sắp xếp theo"), list(sort_labels), format_func=lambda v: sort_labels[v])
    view = named.copy()
    if fam_pick:
        view = view[view["business_family"].isin(fam_pick)]
    if conf_pick:
        view = view[view["naming_confidence"].isin(conf_pick)]
    view = view.sort_values(sort_by, ascending=False)

    cols = {
        "cluster": "id",
        "display_name": t("journey type", "journey type"),
        "business_family": FC,
        "naming_confidence": t("confidence", "độ tin cậy"),
        "size": JC,
        "share": SC,
        "n_sessions": t("sessions", "session"),
        "n_devices": t("devices", "device"),
        "median_length": t("steps", "số bước"),
        "median_span_s": t("duration (s)", "thời lượng (giây)"),
        "mean_action_ratio": "action ratio",
        "mean_back_rate": "back rate",
        "mean_revisit_ratio": t("revisit", "revisit"),
        "loop_journey_share": t("loops", "vòng lặp"),
    }
    show = view[list(cols)].rename(columns=cols)
    show[FC] = show[FC].map(pretty_family)
    st.dataframe(
        show, hide_index=True, width="stretch", height=430,
        column_config={
            SC: st.column_config.ProgressColumn(SC, format="%.2f%%", min_value=0.0, max_value=float(named["share"].max())),
            "back rate": st.column_config.ProgressColumn("back rate", format="%.3f", min_value=0.0, max_value=float(named["mean_back_rate"].max() or 1)),
        },
    )

    st.divider()

    # ------------------------------------------------------------------ detail
    st.subheader(t("Inspect one journey type", "Xem chi tiết một journey type"))
    order = view.sort_values("size", ascending=False)
    pick = st.selectbox(
        t("Journey type", "Journey type"),
        order["cluster"].tolist(),
        format_func=lambda cid: f"#{cid} · {order.loc[order['cluster'] == cid, 'display_name'].iloc[0]} "
        f"({int(order.loc[order['cluster'] == cid, 'size'].iloc[0]):,} {JC})",
    )
    row = named[named["cluster"] == pick].iloc[0]

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### {row['display_name']}")
        alt_name = row["cluster_name"] if vi else row["cluster_name_vi"]
        st.markdown(
            f"*{alt_name}* · {t('family', 'family')} **{pretty_family(row['business_family'])}** "
            f"· {t('naming confidence', 'độ tin cậy khi đặt tên')} **{row['naming_confidence']}**"
        )
        st.markdown(
            t(
                "**Representative journey** (the cluster medoid — the most typical real journey in it):",
                "**Journey đại diện** (medoid của cluster — journey thật điển hình nhất trong đó):",
            )
        )
        rs = steps.get(int(pick), {})
        if rs.get("steps"):
            st.markdown("  \n".join(f"{i + 1}. {s}" for i, s in enumerate(rs["steps"])))
        st.markdown(t("**Underlying technical path**", "**Path kỹ thuật bên dưới**"))
        st.code(row["medoid_path"].replace(" -> ", "\n→ "), language=None)
        st.caption(
            t(
                f"Medoid journey id `{row['medoid_journey_id']}`, {int(row['medoid_length'])} events.",
                f"Journey medoid `{row['medoid_journey_id']}`, {int(row['medoid_length'])} event.",
            )
        )

    with right:
        st.markdown(t("**Behavioural signature vs all clusters**", "**Chữ ký hành vi so với toàn bộ cluster**"))
        metrics = ["median_length", "median_span_s", "mean_action_ratio", "mean_back_rate", "mean_revisit_ratio"]
        labels = [
            t("steps", "số bước"), t("duration", "thời lượng"), "action ratio", "back rate", t("revisit", "revisit"),
        ]
        cohort = named[metrics].median()
        rel = [float(row[m]) / float(cohort[m]) if cohort[m] else 0.0 for m in metrics]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[1] * len(metrics) + [1], theta=labels + [labels[0]], name=t("typical cluster", "cluster điển hình"), line_color="#888"))
        fig.add_trace(go.Scatterpolar(r=rel + [rel[0]], theta=labels + [labels[0]], fill="toself", name=t("this cluster", "cluster này"), line_color=PALETTE[1]))
        fig.update_layout(height=330, margin=dict(t=30, b=10), showlegend=True, polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig, width="stretch")
        st.caption(t("1.0 = the median cluster on that metric.", "1.0 = mức trung vị của các cluster ở chỉ số đó."))

        mcol = t("metric", "chỉ số")
        vcol = t("value", "giá trị")
        st.dataframe(
            pd.DataFrame(
                [
                    {mcol: t("journeys", "journey"), vcol: fmt_int(row["size"])},
                    {mcol: t("share of all journeys", "tỷ lệ trên tổng journey"), vcol: fmt_pct(row["share"], 2)},
                    {mcol: t("sessions", "session"), vcol: fmt_int(row["n_sessions"])},
                    {mcol: t("devices", "device"), vcol: fmt_int(row["n_devices"])},
                    {mcol: t("median steps", "số bước trung vị"), vcol: f"{row['median_length']:.0f}"},
                    {mcol: t("median duration (s)", "thời lượng trung vị (giây)"), vcol: f"{row['median_span_s']:.1f}"},
                    {mcol: "action ratio", vcol: f"{row['mean_action_ratio']:.3f}"},
                    {mcol: "back rate", vcol: f"{row['mean_back_rate']:.3f}"},
                    {mcol: t("revisit ratio", "revisit ratio"), vcol: f"{row['mean_revisit_ratio']:.3f}"},
                    {mcol: t("journeys containing a loop", "journey có vòng lặp"), vcol: fmt_pct(row["loop_journey_share"])},
                ]
            ),
            hide_index=True, width="stretch",
        )

    st.markdown(t("**Entry and exit**", "**Điểm vào và điểm ra**"))
    e1, e2 = st.columns(2)
    e1.code(row["top_entry_token"], language=None)
    e1.caption(t("Most common first step", "Bước đầu tiên phổ biến nhất"))
    e2.code(row["top_exit_token"], language=None)
    e2.caption(t("Most common last step", "Bước cuối cùng phổ biến nhất"))

    ng = load_run_csv(f"{platform}_cluster_ngrams.csv")
    ng = ng[ng["cluster"] == pick].sort_values("rank").head(10)
    if not ng.empty:
        st.markdown(
            t(
                "**Signature step patterns** — n-grams that are far more likely inside this cluster than outside it",
                "**Các mẫu bước đặc trưng** — n-gram xuất hiện trong cluster này nhiều hơn hẳn so với bên ngoài",
            )
        )
        fig = px.bar(ng.sort_values("lift"), x="lift", y="ngram", orientation="h", color_discrete_sequence=[PALETTE[2]], hover_data=["cluster_mass"])
        fig.update_layout(height=420, margin=dict(t=10, b=10), yaxis_title=None, xaxis_title=t("lift vs corpus", "lift so với toàn corpus"))
        fig.update_yaxes(tickfont_size=10)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            t(
                "`lift` = how much more frequent the pattern is inside this cluster than in the corpus; "
                "`cluster_mass` = how much of the cluster's own traffic the pattern accounts for. "
                "These two columns are the audit trail behind the cluster's name.",
                "`lift` = mẫu này xuất hiện trong cluster nhiều hơn bao nhiêu lần so với toàn corpus; "
                "`cluster_mass` = mẫu này chiếm bao nhiêu phần traffic của chính cluster đó. "
                "Hai cột này là bằng chứng kiểm chứng cho tên gọi của cluster.",
            )
        )

    st.divider()

    # ------------------------------------------------------------------ noise
    if len(noise):
        with st.expander(
            t(
                f"About the unclassified bucket ({fmt_pct(noise_share)} of training journeys)",
                f"Về nhóm chưa phân loại ({fmt_pct(noise_share)} số journey khi train)",
            )
        ):
            n = noise.iloc[0]
            st.markdown(
                t(
                    f"""
During fitting, HDBSCAN leaves **{fmt_int(n['size'])} journeys** unassigned. This is
deliberate — they are not forced into a business journey type.

Its profile is close to the global average (median {n['median_length']:.0f} steps,
{n['median_span_s']:.0f} s, back rate {n['mean_back_rate']:.3f}), which is exactly what a
*mixture* looks like: short navigation fragments, half-finished tasks, and rare flows
that never reach the 100-journey minimum cluster size.

Two things matter for the business reading:
1. At **scoring** time most of these journeys do get assigned, because a journey only has to
   fall inside a centroid's learned distance limit — see the coverage numbers on the
   inference page.
2. What stays unassigned at scoring time is the genuinely **novel behaviour watchlist**,
   not a modelling failure.
                    """,
                    f"""
Khi fit, HDBSCAN để lại **{fmt_int(n['size'])} journey** chưa gán. Đây là chủ ý — chúng không bị
ép vào một journey type nghiệp vụ nào.

Hồ sơ của nhóm này gần với mức trung bình toàn cục (trung vị {n['median_length']:.0f} bước,
{n['median_span_s']:.0f} giây, back rate {n['mean_back_rate']:.3f}) — đúng đặc trưng của một
*hỗn hợp*: các mảnh điều hướng ngắn, task làm dở, và những flow hiếm không bao giờ đạt ngưỡng
tối thiểu 100 journey mỗi cluster.

Hai điểm quan trọng khi đọc dưới góc nhìn nghiệp vụ:
1. Khi **scoring**, phần lớn các journey này vẫn được gán, vì chỉ cần rơi vào distance limit đã học
   của một centroid — xem số liệu độ phủ ở trang inference.
2. Phần vẫn không gán được khi scoring mới thực sự là **watchlist hành vi mới**, chứ không phải
   lỗi của mô hình.
                    """,
                )
            )
            st.code(n["medoid_path"].replace(" -> ", "\n→ "), language=None)
