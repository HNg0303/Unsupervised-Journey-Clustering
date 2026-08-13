"""Page 1 — Raw production data: what the clickstream actually looks like."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import PALETTE, cards, fmt_int, fmt_pct, kpi_row, load_eda, load_train_run, t

# `lib` puts src/ on the path, so it must be imported first for these to resolve.
from Rule_based.config import SegmentConfig  # noqa: E402
from Rule_based.segment import assign_journeys  # noqa: E402

# Colour per boundary rule, reused by the timeline and the cut-reason chart so the two
# read as one picture.
REASON_COLOR = {
    "session_start": "#8E8E8E",
    "idle_gap": "#F58518",
    "root_return": "#4C78A8",
    "auth_change": "#B279A2",
    "length_cap": "#E45756",
}


def _reason_label(reason: str) -> str:
    return {
        "session_start": t("session start", "bắt đầu session"),
        "idle_gap": t("idle gap", "nghỉ quá lâu"),
        "root_return": t("root return", "quay về hub"),
        "auth_change": t("auth change", "đổi trạng thái auth"),
        "length_cap": t("length cap", "chạm trần độ dài"),
    }.get(reason, reason)


def _selected_files(eda: dict, platforms: list[str], weeks: list[str]) -> dict:
    return {
        k: v
        for k, v in eda["per_file"].items()
        if v["platform"] in platforms and v["week"] in weeks
    }


def _segment_demo(demo_rows: list[dict], platform: str, cfg: SegmentConfig) -> pd.DataFrame:
    """Run the *production* segmenter over one raw demo session.

    The frame is shaped like the pipeline's tokenised stream (`src/Rule_based/segment.py`
    is imported directly), so what the chart shows is what the pipeline does — the rules
    are never restated here.
    """
    df = pd.DataFrame(demo_rows)
    df["platform"] = platform
    df["session_id"] = "demo"
    df["gap_prev_seconds"] = pd.to_numeric(df["gap_s"], errors="coerce").fillna(0.0)
    df["segment_name"] = df["segmentation_name"].astype(str)
    df["event_type"] = df["key"].astype(str)
    return assign_journeys(df, cfg)


def render() -> None:
    st.title(t("1 · Raw production data", "1 · Dữ liệu production thô"))
    st.caption(
        t(
            "Source: `data/train_data/raw_data_production/data_raw_sample` — Android and iOS "
            "clickstream extracts T3, T4, T5. Everything below is computed on the raw CSVs "
            "before any tokenization or cleaning.",
            "Nguồn: `data/train_data/raw_data_production/data_raw_sample` — clickstream Android và "
            "iOS, các extract T3, T4, T5. Toàn bộ số liệu bên dưới tính trên CSV thô, trước khi "
            "tokenize hay làm sạch.",
        )
    )

    eda = load_eda()

    c1, c2 = st.columns([2, 3])
    platforms = c1.multiselect("Platform", ["android", "ios"], default=["android", "ios"])
    weeks = c2.multiselect("Extract", ["T3", "T4", "T5"], default=["T3", "T4", "T5"])
    files = _selected_files(eda, platforms, weeks)
    if not files:
        st.warning(t("Select at least one platform and one extract.", "Hãy chọn ít nhất một platform và một extract."))
        return

    rows = sum(f["rows"] for f in files.values())
    sessions = sum(f["n_sessions"] for f in files.values())
    devices = max(f["n_devices"] for f in files.values())
    customers = max(f["n_customers"] for f in files.values())
    actions = sum(f["key_counts"].get("action", 0) for f in files.values())

    kpi_row(
        [
            (
                t("Raw events", "Event thô"),
                fmt_int(rows),
                t("Rows after dropping unparsable timestamps", "Số dòng sau khi bỏ các timestamp không parse được"),
            ),
            (
                t("Sessions", "Session"),
                fmt_int(sessions),
                t("Distinct session_id across selected extracts", "Số session_id khác nhau trong các extract đã chọn"),
            ),
            (
                t("Devices (max/extract)", "Device (max/extract)"),
                fmt_int(devices),
                t("device_id — the sample is ~1k devices per extract", "device_id — mẫu lấy ~1k device mỗi extract"),
            ),
            (
                t("Customers (max/extract)", "Customer (max/extract)"),
                fmt_int(customers),
                t("customer_id — sampled at 1k users per extract", "customer_id — lấy mẫu 1k user mỗi extract"),
            ),
            (
                t("Action events", "Event action"),
                fmt_pct(actions / rows),
                t("Share of `key = action`; the rest are screen views", "Tỷ lệ `key = action`; phần còn lại là screen view"),
            ),
        ]
    )

    st.divider()

    # ------------------------------------------------------------------ file inventory
    st.subheader(t("What is in each extract", "Mỗi extract có gì"))
    L = {
        "file": t("file", "file"),
        "platform": "platform",
        "extract": "extract",
        "events": t("events", "event"),
        "sessions": t("sessions", "session"),
        "customers": t("customers", "customer"),
        "action_share": t("action share", "tỷ lệ action"),
        "first": t("first event", "event đầu"),
        "last": t("last event", "event cuối"),
        "vocab": t("distinct segment_name", "số segment_name khác nhau"),
    }
    inv = pd.DataFrame(
        [
            {
                L["file"]: f["file"],
                L["platform"]: f["platform"],
                L["extract"]: f["week"],
                L["events"]: f["rows"],
                L["sessions"]: f["n_sessions"],
                L["customers"]: f["n_customers"],
                L["action_share"]: f["action_share"],
                L["first"]: f["ts_min"][:10],
                L["last"]: f["ts_max"][:10],
                L["vocab"]: f["taxonomy_signals"]["vocab_size"],
            }
            for f in files.values()
        ]
    ).sort_values([L["platform"], L["extract"]])
    st.dataframe(
        inv,
        hide_index=True,
        width="stretch",
        column_config={
            L["action_share"]: st.column_config.ProgressColumn(
                L["action_share"], format="%.2f", min_value=0.0, max_value=1.0
            ),
            L["events"]: st.column_config.NumberColumn(format="%d"),
            L["sessions"]: st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption(
        t(
            "The extracts overlap in time and are sampled per user, not per day — so they are "
            "treated as one pooled production sample rather than as strict consecutive weeks.",
            "Các extract có khoảng thời gian chồng lấn và được lấy mẫu theo user chứ không theo ngày "
            "— nên được xem như một mẫu production gộp, không phải các tuần liên tiếp.",
        )
    )

    # ------------------------------------------------------------------ raw stream
    st.subheader(t("A raw event stream, untouched", "Một luồng event thô, chưa xử lý gì"))
    demo_key = st.selectbox(
        t("Show one real session from", "Xem một session thật từ"), list(files.keys())
    )
    demo = files[demo_key]["demo_session"]
    if demo:
        d = pd.DataFrame(demo)
        d[t("event token", "event token")] = d["key"] + "@" + d["segmentation_name"]
        st.dataframe(
            d[["client_time", "gap_s", "key", "segmentation_name", t("event token", "event token")]],
            hide_index=True,
            width="stretch",
            height=320,
        )
        st.markdown(
            t(
                "**Read it like a product person:** every row is one screen view or one tap. "
                "There is no notion of *task* anywhere in the schema — no funnel id, no flow "
                "name, no completion flag. The only structure the app emits is `key` "
                "(`view` / `action`) plus a free-text `segmentation_name`. Any concept of "
                '"the user was trying to pay a bill" has to be **reconstructed** from this stream.',
                "**Đọc dưới góc nhìn product:** mỗi dòng là một screen view hoặc một cú tap. "
                "Trong schema không hề có khái niệm *task* — không funnel id, không tên flow, "
                "không cờ hoàn thành. App chỉ phát ra `key` (`view` / `action`) kèm một chuỗi "
                "tự do `segmentation_name`. Mọi khái niệm kiểu \"user đang định thanh toán hóa đơn\" "
                "đều phải được **tái dựng lại** từ luồng này.",
            )
        )

    st.divider()

    # ------------------------------------------------------------------ why taxonomy
    st.header(t("Why this data needs a taxonomy", "Vì sao dữ liệu này cần taxonomy"))
    st.markdown(
        t(
            """
`segmentation_name` is a developer string, not a product concept. That breaks the data in
**two different ways**, and each one needs its own fix — together they are what the word
*taxonomy* means in this pipeline:

1. **The same action is written down differently every time.** URLs, UUIDs and record ids
   sit inside the name, so two users doing one thing produce two strings.
   → **Canonization** is string hygiene: strip non-semantic query params, mask id-like path
   segments, truncate over-deep action paths, fold tokens too rare to learn from.
   *Without it the vocabulary explodes and identical behaviour never aligns.*
2. **Even after cleaning, the names describe the code, not the product.** `HomeVC` and
   `android/Home` are the same surface; nothing in the string says either is "home", and
   nothing groups payment screens together.
   → **The family taxonomy** is the semantic layer: every canonical screen is mapped to
   `business_family / business_module / object / operation`.
   *Without it Android and iOS can never be compared, clusters cannot be named in business
   language, and rare tokens have no coarser level to back off to.*

The three tabs below are the evidence for both claims, measured on the raw vocabulary.
            """,
            """
`segmentation_name` là một chuỗi do lập trình viên đặt, không phải khái niệm sản phẩm. Điều đó
làm dữ liệu hỏng theo **hai kiểu khác nhau**, mỗi kiểu cần một cách xử lý riêng — gộp lại chính
là ý nghĩa của chữ *taxonomy* trong pipeline này:

1. **Cùng một hành động nhưng mỗi lần lại được ghi khác nhau.** URL, UUID và id bản ghi nằm
   ngay trong tên, nên hai user làm cùng một việc lại sinh ra hai chuỗi khác nhau.
   → **Canonize** là bước làm sạch chuỗi: bỏ query param không mang ngữ nghĩa, mask các đoạn
   path dạng id, cắt bớt path action quá sâu, gộp các token quá hiếm để học được.
   *Không có bước này, vocabulary bùng nổ và các hành vi giống hệt nhau không bao giờ khớp nhau.*
2. **Sạch rồi thì tên vẫn mô tả code chứ không mô tả sản phẩm.** `HomeVC` và `android/Home` là
   cùng một màn hình; trong chuỗi không có gì nói rằng đó là "home", cũng không có gì gom các
   màn hình thanh toán về một nhóm.
   → **Family taxonomy** là lớp ngữ nghĩa: mọi screen đã canonize được ánh xạ về
   `business_family / business_module / object / operation`.
   *Không có lớp này, Android và iOS không thể so sánh với nhau, cluster không thể đặt tên theo
   ngôn ngữ nghiệp vụ, và token hiếm không có mức thô hơn nào để backoff về.*

Ba tab bên dưới là bằng chứng cho cả hai luận điểm, đo trên vocabulary thô.
            """,
        )
    )

    sig = {k: v["taxonomy_signals"] for k, v in files.items()}
    C = {
        "file": t("file", "file"),
        "vocab": t("vocabulary", "vocabulary"),
        "singletons": t("singletons", "singleton"),
        "singleton_share": t("singleton share", "tỷ lệ singleton"),
        "url": t("contains URL", "chứa URL"),
        "uuid": t("contains UUID", "chứa UUID"),
        "longid": t("contains long id", "chứa id dài"),
        "query": t("has query string", "có query string"),
        "top100": t("top-100 event coverage", "top-100 phủ % event"),
    }
    tax = pd.DataFrame(
        [
            {
                C["file"]: k,
                C["vocab"]: s["vocab_size"],
                C["singletons"]: s["singletons"],
                C["singleton_share"]: s["singleton_share"],
                C["url"]: s["with_url"] / max(s["vocab_size"], 1),
                C["uuid"]: s["with_uuid"] / max(s["vocab_size"], 1),
                C["longid"]: s["with_long_number"] / max(s["vocab_size"], 1),
                C["query"]: s["with_query_string"] / max(s["vocab_size"], 1),
                C["top100"]: s["top100_event_coverage"],
            }
            for k, s in sig.items()
        ]
    )

    t1, t2, t3 = st.tabs(
        [
            t("① Long tail of one-off names", "① Đuôi dài các tên chỉ xuất hiện 1 lần"),
            t("② Names carry raw identifiers", "② Tên screen chứa identifier thô"),
            t("③ Android ≠ iOS naming", "③ Android ≠ iOS về cách đặt tên"),
        ]
    )

    with t1:
        a, b = st.columns([3, 2])
        fig = px.bar(
            tax.sort_values(C["file"]),
            x=C["file"],
            y=[C["singleton_share"], C["top100"]],
            barmode="group",
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(
            yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None,
            legend_title=None, height=340, margin=dict(t=10, b=10),
        )
        a.plotly_chart(fig, width="stretch")
        b.markdown(
            t(
                f"""
Across the selected extracts the app emits **{tax[C['vocab']].max():,}** distinct
`segmentation_name` values at most — but roughly **{tax[C['singleton_share']].mean():.0%}**
of them are seen exactly **once**, while the top 100 names already cover
**~{tax[C['top100']].mean():.0%}** of all events.

**Consequence:** a raw-string vocabulary is mostly noise. Clustering on it would spend
its capacity on names that never repeat, and every rare-but-real flow would be
indistinguishable from a typo. This is why the taxonomy needs *levels*: rare tokens fold
into their coarser form (`min_journey_df = 3`, backoff to L2 = family/module) instead of
being thrown away.
                """,
                f"""
Trên các extract đã chọn, app phát ra nhiều nhất **{tax[C['vocab']].max():,}** giá trị
`segmentation_name` khác nhau — nhưng khoảng **{tax[C['singleton_share']].mean():.0%}**
trong số đó chỉ xuất hiện **đúng một lần**, trong khi 100 tên phổ biến nhất đã phủ
**~{tax[C['top100']].mean():.0%}** tổng số event.

**Hệ quả:** một vocabulary bằng chuỗi thô phần lớn là noise. Cluster trên đó sẽ tiêu tốn
năng lực mô hình vào những tên không bao giờ lặp lại, và mọi flow hiếm-nhưng-có-thật đều
không phân biệt được với một lỗi gõ. Đây chính là lý do taxonomy phải có *nhiều mức*: token
hiếm được gộp về dạng thô hơn (`min_journey_df = 3`, backoff sang L2 = family/module)
thay vì bị vứt đi.
                """,
            )
        )

    with t2:
        a, b = st.columns([3, 2])
        melted = tax.melt(
            id_vars=C["file"],
            value_vars=[C["url"], C["query"], C["uuid"], C["longid"]],
            var_name=t("pattern", "dạng"),
            value_name=t("share of vocabulary", "tỷ lệ trong vocabulary"),
        )
        fig = px.bar(
            melted, x=C["file"], y=t("share of vocabulary", "tỷ lệ trong vocabulary"),
            color=t("pattern", "dạng"), barmode="group", color_discrete_sequence=PALETTE,
        )
        fig.update_layout(
            yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None,
            legend_title=None, height=340, margin=dict(t=10, b=10),
        )
        a.plotly_chart(fig, width="stretch")
        b.markdown(
            t(
                """
A large share of screen names are **URLs with query strings, UUIDs and numeric ids**,
e.g. `Home/166/1318/open_url_in_app_with_access_token/https://hi.fpt.vn/dkol/product-detail?merchantId=BH_CAM&packageId=combo-camera-3-v2`.

**Consequence:** two users doing *the same thing* produce *different* strings, because
the ids differ. This is exactly the job of canonization (`canonize_urls`,
`mask_id_segments`): without it the vocabulary explodes and identical behaviour never
aligns. It is also the single biggest driver of the singleton long tail in the first tab.
                """,
                """
Một phần lớn tên screen là **URL kèm query string, UUID và id dạng số**,
ví dụ `Home/166/1318/open_url_in_app_with_access_token/https://hi.fpt.vn/dkol/product-detail?merchantId=BH_CAM&packageId=combo-camera-3-v2`.

**Hệ quả:** hai user làm *cùng một việc* lại sinh ra *hai chuỗi khác nhau*, chỉ vì id khác nhau.
Đây đúng là nhiệm vụ của bước canonize (`canonize_urls`, `mask_id_segments`): thiếu nó,
vocabulary bùng nổ và các hành vi giống hệt nhau không bao giờ khớp được. Đây cũng là nguyên
nhân lớn nhất tạo ra cái đuôi singleton ở tab đầu tiên.
                """,
            )
        )
        with st.expander(t("Where the ids hide: path depth", "Id nằm ở đâu: độ sâu path")):
            depth_rows = []
            depth_col = t("path depth (`/` count)", "độ sâu path (số `/`)")
            names_col = t("names", "số tên")
            for k, s in sig.items():
                for d, n in s["depth_hist"].items():
                    depth_rows.append({C["file"]: k, depth_col: d, names_col: n})
            dd = pd.DataFrame(depth_rows)
            fig2 = px.bar(
                dd, x=depth_col, y=names_col, color=C["file"],
                barmode="group", color_discrete_sequence=PALETTE,
            )
            fig2.update_layout(height=280, margin=dict(t=10, b=10), legend_title=None)
            st.plotly_chart(fig2, width="stretch")
            st.caption(
                t(
                    "Path depth of distinct screen names (6 = six or more `/`). The deep tail is where "
                    "ids and URLs live; canonization truncates action paths to depth 3 (`action_path_depth_mid`).",
                    "Độ sâu path của các tên screen khác nhau (6 = từ sáu `/` trở lên). Phần đuôi sâu chính là "
                    "nơi chứa id và URL; bước canonize cắt path của action về độ sâu 3 (`action_path_depth_mid`).",
                )
            )

    with t3:
        ov = eda.get("platform_vocab_overlap", {})
        if ov:
            a, b = st.columns([2, 3])
            fig = go.Figure(
                go.Bar(
                    x=[ov["android_only"], ov["shared"], ov["ios_only"]],
                    y=[
                        t("Android only", "Chỉ Android"),
                        t("Shared", "Dùng chung"),
                        t("iOS only", "Chỉ iOS"),
                    ],
                    orientation="h",
                    marker_color=[PALETTE[0], PALETTE[2], PALETTE[1]],
                    text=[ov["android_only"], ov["shared"], ov["ios_only"]],
                )
            )
            fig.update_layout(
                height=260, margin=dict(t=10, b=10),
                xaxis_title=t("distinct screen names", "số tên screen khác nhau"),
            )
            a.plotly_chart(fig, width="stretch")
            b.markdown(
                t(
                    f"""
Only **{ov['shared']:,}** screen names are shared between Android and iOS —
a Jaccard overlap of **{ov['jaccard']:.1%}**.

The same product surface is called `MainAppActivity` / `android/Home` on Android and
`MainTabBarController` / `HomeVC` on iOS. The names come from the *implementation*
(Activities, ViewControllers), not from the *product* — and no amount of string cleaning
can fix that, because the two strings genuinely share no characters.

**Consequence:** this is the part canonization cannot solve, and the reason the second
layer exists. The **family taxonomy** maps both platforms into one semantic space
(`business_family` / `business_module` / operation), which is what makes a single
cross-platform journey catalogue — and business-readable cluster names — possible.
Models are still fit **per platform** (`namespace_by_os = true`), but their outputs land
in one shared vocabulary.
                    """,
                    f"""
Chỉ **{ov['shared']:,}** tên screen được dùng chung giữa Android và iOS —
độ trùng Jaccard chỉ **{ov['jaccard']:.1%}**.

Cùng một màn hình sản phẩm nhưng Android gọi là `MainAppActivity` / `android/Home`, còn iOS gọi là
`MainTabBarController` / `HomeVC`. Tên sinh ra từ *cách implement* (Activity, ViewController),
không phải từ *sản phẩm* — và không cách làm sạch chuỗi nào chữa được, vì hai chuỗi đó thực sự
không có ký tự nào chung.

**Hệ quả:** đây là phần mà canonize không giải quyết được, và là lý do phải có lớp thứ hai.
**Family taxonomy** ánh xạ cả hai platform về một không gian ngữ nghĩa chung
(`business_family` / `business_module` / operation) — nhờ vậy mới có một catalogue journey
duy nhất cho cả hai platform và mới đặt được tên cluster theo ngôn ngữ nghiệp vụ.
Model vẫn được fit **riêng theo từng platform** (`namespace_by_os = true`), nhưng output đổ về
chung một vocabulary.
                    """,
                )
            )
            st.caption(
                t("Examples of the few genuinely shared names: ", "Vài ví dụ hiếm hoi thực sự dùng chung: ")
                + ", ".join(f"`{x}`" for x in ov["shared_examples"][:8])
            )

    st.divider()

    # ------------------------------------------------------------------ sessions
    st.header(t("Sessions are too coarse to be a unit of analysis", "Session quá thô để làm đơn vị phân tích"))
    st.caption(
        t(
            "This is the reason sessions get re-segmented into journeys before modelling.",
            "Đây là lý do session phải được cắt lại thành journey trước khi mô hình hóa.",
        )
    )

    qcol = t("quantile", "quantile")
    lcol = t("events per session", "event mỗi session")
    scol = t("session span (s)", "độ dài session (giây)")
    qrows = []
    for k, f in files.items():
        for qk, qv in f["session_len_q"].items():
            qrows.append({C["file"]: k, qcol: float(qk), lcol: qv})
    qd = pd.DataFrame(qrows)
    a, b = st.columns(2)
    fig = px.line(qd, x=qcol, y=lcol, color=C["file"], markers=True, color_discrete_sequence=PALETTE, log_y=True)
    fig.update_layout(height=330, margin=dict(t=10, b=10), legend_title=None)
    a.plotly_chart(fig, width="stretch")

    srows = []
    for k, f in files.items():
        for qk, qv in f["session_span_q"].items():
            srows.append({C["file"]: k, qcol: float(qk), scol: max(qv or 0, 0.1)})
    sd = pd.DataFrame(srows)
    fig = px.line(sd, x=qcol, y=scol, color=C["file"], markers=True, color_discrete_sequence=PALETTE, log_y=True)
    fig.update_layout(height=330, margin=dict(t=10, b=10), legend_title=None)
    b.plotly_chart(fig, width="stretch")

    med_len = qd[qd[qcol] == 0.5][lcol].median()
    p99_len = qd[qd[qcol] == 0.99][lcol].median()
    med_span = sd[sd[qcol] == 0.5][scol].median()
    p99_span = sd[sd[qcol] == 0.99][scol].median()
    st.markdown(
        t(
            f"""
The distribution is brutally skewed. A median session holds **~{med_len:.0f} events over
~{med_span:.0f} s**, but the top 1% of sessions run to **~{p99_len:.0f} events spread over
~{p99_span/3600:.0f} hours**. Those long sessions are not one long task — a single
`session_id` bundles *several unrelated tasks* separated by long idle periods: paying a bill
in the morning and checking Wi-Fi in the evening land in one row.
            """,
            f"""
Phân phối lệch cực mạnh. Session trung vị chỉ có **~{med_len:.0f} event trong ~{med_span:.0f} giây**,
nhưng 1% session dài nhất lên tới **~{p99_len:.0f} event kéo dài ~{p99_span/3600:.0f} giờ**.
Những session dài đó không phải là một task dài — một `session_id` gói *nhiều task không liên quan*
cách nhau bởi những khoảng nghỉ dài: thanh toán hóa đơn buổi sáng và kiểm tra Wi-Fi buổi tối
nằm chung một dòng.
            """,
        )
    )

    st.divider()

    # ------------------------------------------------------------------ segmentation
    st.header(t("From session to journey: the rule-based cut", "Từ session sang journey: cách cắt rule-based"))
    st.markdown(
        t(
            "A **journey** is one goal-directed stretch of behaviour. Since the schema never "
            "marks where a goal starts or ends, the pipeline places a boundary whenever the "
            "stream shows one of five interpretable signals (`src/Rule_based/segment.py`, "
            "stage L0 — no fitting, fully auditable). A journey never spans two sessions.",
            "**Journey** là một đoạn hành vi hướng tới một mục tiêu. Vì schema không hề đánh dấu "
            "mục tiêu bắt đầu hay kết thúc ở đâu, pipeline đặt ranh giới mỗi khi luồng event xuất "
            "hiện một trong năm tín hiệu có thể giải thích được (`src/Rule_based/segment.py`, "
            "tầng L0 — không cần fit, kiểm tra được từng bước). Một journey không bao giờ nằm vắt "
            "qua hai session.",
        )
    )

    defaults = SegmentConfig()
    gap_shares = [f["gap_over_90s_share"] for f in files.values()]
    cards(
        [
            (
                t("① Session change", "① Đổi session"),
                t(
                    "A new `session_id` always opens a new journey — journeys are cut inside "
                    "sessions, never across them.",
                    "Một `session_id` mới luôn mở một journey mới — journey chỉ được cắt bên trong "
                    "session, không bao giờ vắt qua hai session.",
                ),
                "session_start",
                REASON_COLOR["session_start"],
            ),
            (
                t("② Idle gap > τ", "② Nghỉ quá τ"),
                t(
                    f"The user stopped for more than **{defaults.idle_gap_seconds:.0f} s**. Only "
                    f"~{sum(gap_shares)/len(gap_shares):.1%} of steps qualify — inside a task the "
                    "next step comes in seconds.",
                    f"User dừng lâu hơn **{defaults.idle_gap_seconds:.0f} giây**. Chỉ khoảng "
                    f"~{sum(gap_shares)/len(gap_shares):.1%} số bước rơi vào ngưỡng này — trong một "
                    "task, bước kế tiếp đến sau vài giây.",
                ),
                "idle_gap",
                REASON_COLOR["idle_gap"],
            ),
            (
                t("③ Return to a hub", "③ Quay về hub"),
                t(
                    f"Landing back on a root screen (`HOME`, `HomeVC`, …) after ≥ "
                    f"{defaults.root_return_min_events} events usually means the previous goal is done.",
                    f"Quay lại màn hình gốc (`HOME`, `HomeVC`, …) sau ≥ {defaults.root_return_min_events} "
                    "event thường có nghĩa mục tiêu trước đó đã xong.",
                ),
                "root_return",
                REASON_COLOR["root_return"],
            ),
            (
                t("④ Auth change", "④ Đổi trạng thái auth"),
                t(
                    "A login / logout **action** (never a mere view of a login screen, which "
                    "fires all through guest browsing).",
                    "Một **action** login / logout (không tính việc chỉ xem màn hình login — điều đó "
                    "xảy ra suốt lúc duyệt ở chế độ khách).",
                ),
                "auth_change",
                REASON_COLOR["auth_change"],
            ),
            (
                t("⑤ Length cap", "⑤ Trần độ dài"),
                t(
                    f"A hard stop at **{defaults.max_journey_length} events**, so a pathological "
                    "1,500-event session cannot become one journey.",
                    f"Chốt cứng ở **{defaults.max_journey_length} event**, để một session bất thường "
                    "1.500 event không thể thành một journey duy nhất.",
                ),
                "length_cap",
                REASON_COLOR["length_cap"],
            ),
        ]
    )

    # ---- the evidence behind tau
    with st.expander(t("Evidence for τ = 90 s: the gap distribution", "Căn cứ chọn τ = 90 giây: phân phối gap")):
        grows = []
        for k, f in files.items():
            grows.append(
                {
                    C["file"]: k,
                    t("median gap (s)", "gap trung vị (giây)"): f["gap_q"]["0.5"],
                    t("p90 gap (s)", "gap p90 (giây)"): f["gap_q"]["0.9"],
                    t("p99 gap (s)", "gap p99 (giây)"): f["gap_q"]["0.99"],
                    t("share of gaps > 90 s", "tỷ lệ gap > 90 giây"): f["gap_over_90s_share"],
                    t("share of gaps > 30 min", "tỷ lệ gap > 30 phút"): f["gap_over_1800s_share"],
                }
            )
        st.dataframe(pd.DataFrame(grows), hide_index=True, width="stretch")
        st.caption(
            t(
                "Consecutive-event gaps within a session. Nearly all steps happen within seconds; "
                "τ sits just past the p97.5 of that distribution, so it only fires on the tail "
                "where one task ends and another begins.",
                "Khoảng cách giữa hai event liên tiếp trong cùng session. Gần như mọi bước diễn ra "
                "trong vài giây; τ được đặt ngay sau p97.5 của phân phối này, nên chỉ kích hoạt ở "
                "phần đuôi — nơi một task kết thúc và task khác bắt đầu.",
            )
        )

    # ---- live demo on the real session shown above
    st.subheader(t("Watch the rules cut a real session", "Xem các luật cắt một session thật"))
    st.caption(
        t(
            "The controls below feed the production segmenter (`assign_journeys`) directly, on "
            "the same raw session shown earlier. Defaults are the production settings — move "
            "them to see how each rule changes the cut.",
            "Các thanh điều khiển bên dưới chạy thẳng vào hàm segment production (`assign_journeys`), "
            "trên đúng session thô ở phần trên. Giá trị mặc định là cấu hình production — hãy kéo thử "
            "để thấy từng luật ảnh hưởng ra sao.",
        )
    )

    if not demo:
        st.info(t("No demo session available for this extract.", "Extract này không có session mẫu."))
    else:
        k1, k2, k3, k4 = st.columns([2, 2, 1.4, 1.4])
        tau = k1.slider(
            t("② idle gap τ (s)", "② ngưỡng nghỉ τ (giây)"),
            min_value=1.0, max_value=300.0, value=float(defaults.idle_gap_seconds), step=1.0,
        )
        cap = k2.slider(
            t("⑤ length cap (events)", "⑤ trần độ dài (event)"),
            min_value=4, max_value=int(defaults.max_journey_length), value=int(defaults.max_journey_length), step=1,
        )
        use_root = k3.toggle(t("③ hub return", "③ về hub"), value=defaults.cut_on_root_return)
        use_auth = k4.toggle(t("④ auth change", "④ đổi auth"), value=defaults.cut_on_auth_change)

        cfg = SegmentConfig(
            idle_gap_seconds=tau,
            cut_on_root_return=use_root,
            cut_on_auth_change=use_auth,
            max_journey_length=cap,
            root_return_min_events=defaults.root_return_min_events,
        )
        seg = _segment_demo(demo, files[demo_key]["platform"], cfg)
        seg["step"] = range(1, len(seg) + 1)
        journey_order = {j: i for i, j in enumerate(seg["journey_id"].unique(), start=1)}
        seg["lane"] = seg["journey_id"].map(journey_order)
        seg["token"] = seg["event_type"] + "@" + seg["segment_name"]

        sizes = seg.groupby("journey_id", sort=False).size()
        m1, m2, m3 = st.columns(3)
        m1.metric(t("Events in this session", "Event trong session này"), len(seg))
        m2.metric(t("Journeys produced", "Journey tạo ra"), int(sizes.size))
        m3.metric(
            t("Below min length (dropped)", "Ngắn hơn min length (bị loại)"),
            int((sizes < defaults.min_journey_length).sum()),
            help=t(
                f"Journeys shorter than {defaults.min_journey_length} events carry no order "
                "information and are kept on disk but excluded from clustering.",
                f"Journey ngắn hơn {defaults.min_journey_length} event không còn thông tin về thứ tự; "
                "vẫn được lưu ra file nhưng bị loại khỏi bước clustering.",
            ),
        )

        # --- timeline: gaps on top, the cut sequence below
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7], vertical_spacing=0.06
        )
        fig.add_trace(
            go.Bar(
                x=seg["step"], y=seg["gap_prev_seconds"],
                marker_color=[
                    REASON_COLOR["idle_gap"] if g > tau else "rgba(128,128,128,.45)"
                    for g in seg["gap_prev_seconds"]
                ],
                name=t("gap before event (s)", "gap trước event (giây)"),
                hovertemplate="step %{x}<br>gap %{y:.0f}s<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_hline(
            y=tau, line_dash="dot", line_color=REASON_COLOR["idle_gap"], row=1, col=1,
            annotation_text=f"τ = {tau:.0f}s", annotation_position="top left",
        )

        for lane, (jid, grp) in enumerate(seg.groupby("journey_id", sort=False), start=1):
            colour = PALETTE[(lane - 1) % len(PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=grp["step"], y=grp["lane"],
                    mode="lines+markers",
                    line=dict(color=colour, width=3),
                    marker=dict(
                        color=colour, size=12,
                        symbol=["diamond" if e == "action" else "circle" for e in grp["event_type"]],
                        line=dict(color="rgba(255,255,255,.35)", width=1),
                    ),
                    name=f"{jid} ({len(grp)})",
                    customdata=grp[["token"]],
                    hovertemplate="step %{x}<br>%{customdata[0]}<extra>" + jid + "</extra>",
                ),
                row=2, col=1,
            )

        for _, r in seg[seg["boundary_reason"].ne("")].iterrows():
            colour = REASON_COLOR.get(r["boundary_reason"], "#999")
            fig.add_vline(
                x=r["step"] - 0.5, line_dash="dash", line_color=colour, line_width=2,
                row=2, col=1,
                annotation_text=_reason_label(r["boundary_reason"]),
                annotation_position="top",
                annotation_font=dict(color=colour, size=11),
            )

        fig.update_yaxes(title_text=t("gap (s)", "gap (giây)"), row=1, col=1)
        fig.update_yaxes(
            title_text=t("journey", "journey"), row=2, col=1,
            tickmode="array", tickvals=list(journey_order.values()),
            ticktext=[f"#{i}" for i in journey_order.values()],
            autorange="reversed",
        )
        fig.update_xaxes(title_text=t("event # in session", "event thứ mấy trong session"), row=2, col=1)
        fig.update_layout(
            height=520, margin=dict(t=40, b=10), showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            bargap=0.35,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            t(
                "Each staircase step is one journey. ● = screen view, ◆ = tap. Dashed lines are "
                "boundaries, coloured by the rule that fired; orange bars above are gaps that "
                "exceed τ.",
                "Mỗi bậc thang là một journey. ● = screen view, ◆ = thao tác tap. Đường đứt nét là "
                "ranh giới, màu theo luật đã kích hoạt; cột màu cam phía trên là các gap vượt τ.",
            )
        )

        with st.expander(t("The same cut, event by event", "Cũng bản cắt đó, xem theo từng event")):
            show = seg[["step", "journey_id", "boundary_reason", "gap_prev_seconds", "event_type", "segment_name"]].copy()
            show["boundary_reason"] = show["boundary_reason"].map(lambda r: _reason_label(r) if r else "")
            show.columns = [
                t("step", "bước"), "journey_id",
                t("cut reason", "lý do cắt"),
                t("gap (s)", "gap (giây)"),
                t("event type", "loại event"),
                "segment_name",
            ]
            st.dataframe(show, hide_index=True, width="stretch", height=340)

    # ---- what the rules do at production scale
    train = load_train_run()
    reason_rows = []
    for platform in ["android", "ios"]:
        counts = train["platforms"].get(platform, {}).get("journeys", {}).get("boundary_reason", {})
        total = sum(counts.values()) or 1
        for reason, n in counts.items():
            reason_rows.append(
                {
                    "platform": platform,
                    t("rule", "luật"): _reason_label(reason),
                    t("share of journeys", "tỷ lệ journey"): n / total,
                    t("journeys", "số journey"): n,
                }
            )
    if reason_rows:
        st.subheader(t("Which rule does the work in production", "Luật nào thực sự làm việc trên production"))
        rdf = pd.DataFrame(reason_rows)
        fig = px.bar(
            rdf, x=t("share of journeys", "tỷ lệ journey"), y=t("rule", "luật"),
            color="platform", barmode="group", orientation="h",
            color_discrete_sequence=[PALETTE[0], PALETTE[1]],
            hover_data=[t("journeys", "số journey")],
        )
        fig.update_layout(
            height=320, margin=dict(t=10, b=10), xaxis_tickformat=".0%",
            yaxis_title=None, legend_title=None,
        )
        st.plotly_chart(fig, width="stretch")
        tot_j = sum(train["platforms"][p]["journeys"]["n_journeys"] for p in ["android", "ios"] if train["platforms"].get(p, {}).get("journeys"))
        tot_s = sum(train["platforms"][p]["journeys"]["n_sessions"] for p in ["android", "ios"] if train["platforms"].get(p, {}).get("journeys"))
        st.markdown(
            t(
                f"""
On the full training sample the rules turn **{tot_s:,} sessions into {tot_j:,} journeys**
(≈ {tot_j/max(tot_s,1):.1f} goals per session, median length 9–10 events). Hub return and
idle gap do almost all of the work; the length cap fires on a few hundred pathological
sessions, which is exactly what a safety net should look like.
                """,
                f"""
Trên toàn bộ mẫu train, các luật biến **{tot_s:,} session thành {tot_j:,} journey**
(≈ {tot_j/max(tot_s,1):.1f} mục tiêu mỗi session, độ dài trung vị 9–10 event). Quay về hub và
nghỉ quá lâu gánh gần hết công việc; trần độ dài chỉ kích hoạt ở vài trăm session bất thường —
đúng vai trò của một lưới an toàn.
                """,
            )
        )

    st.divider()

    # ------------------------------------------------------------------ tokens
    st.header(t("Event tokens and traffic shape", "Event token và hình dạng traffic"))
    left, right = st.columns(2)
    pick2 = left.selectbox("Extract", list(files.keys()), key="token_pick")
    f = files[pick2]
    kind = right.radio(t("Token type", "Loại token"), ["view", "action"], horizontal=True)
    top = f["top_view_tokens"] if kind == "view" else f["top_action_tokens"]
    ev_col = t("events", "event")
    td = pd.DataFrame(top, columns=["segmentation_name", ev_col]).head(20)
    td["token"] = kind + "@" + td["segmentation_name"]
    fig = px.bar(
        td.sort_values(ev_col), x=ev_col, y="token", orientation="h",
        color_discrete_sequence=[PALETTE[0] if kind == "view" else PALETTE[1]],
    )
    fig.update_layout(height=560, margin=dict(t=10, b=10), yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        t(
            "`event_type@segment_name` is the exact token the model consumes. Note how the "
            "top of the distribution is dominated by container screens (splash, home, tab bar) "
            "that carry no product intent — the pipeline drops chrome/boot noise before clustering.",
            "`event_type@segment_name` chính là token mà model đọc vào. Chú ý phần đỉnh phân phối "
            "bị chiếm bởi các screen container (splash, home, tab bar) — không mang ý định sản phẩm nào. "
            "Pipeline loại bỏ noise chrome/boot này trước khi cluster.",
        )
    )

    st.divider()
    st.subheader(t("What the raw data tells us — summary", "Dữ liệu thô nói lên điều gì — tóm tắt"))
    st.markdown(
        t(
            """
| Finding | Evidence | What it forces the pipeline to do |
| --- | --- | --- |
| No task/flow concept exists in the schema | Only `key` + free-text `segmentation_name` | Reconstruct journeys unsupervised from sequences |
| Names embed URLs, UUIDs and record ids | Majority of distinct names contain a URL or query string | **Canonize**: URL cleaning + id masking before tokenizing |
| Half the vocabulary is seen once | Singleton share ≈ 30–51%, top-100 names cover ~90% of events | Rare-token folding with backoff to a coarser taxonomy level |
| Screen names are an implementation artefact | ~9% vocabulary overlap between Android and iOS | **Family taxonomy**: one semantic space; fit per platform |
| Sessions mix unrelated tasks | Median session ≈ minutes, p99 ≈ hours | Re-segment sessions into journeys (5 rules, τ = 90 s) |
| Traffic is dominated by chrome | Splash / home / tab-bar top the token ranking | Drop chrome + collapse repeats and loops |
            """,
            """
| Phát hiện | Bằng chứng | Buộc pipeline phải làm gì |
| --- | --- | --- |
| Schema không có khái niệm task/flow | Chỉ có `key` + `segmentation_name` dạng text tự do | Tái dựng journey theo hướng unsupervised từ chuỗi event |
| Tên chứa URL, UUID và id bản ghi | Đa số tên khác nhau có chứa URL hoặc query string | **Canonize**: làm sạch URL + mask id trước khi tokenize |
| Một nửa vocabulary chỉ xuất hiện 1 lần | Tỷ lệ singleton ≈ 30–51%, top-100 tên phủ ~90% event | Gộp token hiếm, backoff về mức taxonomy thô hơn |
| Tên screen là sản phẩm của cách implement | Vocabulary Android và iOS chỉ trùng ~9% | **Family taxonomy**: một không gian ngữ nghĩa chung; fit riêng từng platform |
| Session trộn nhiều task không liên quan | Session trung vị tính bằng phút, p99 tính bằng giờ | Cắt session thành journey (5 luật, τ = 90 giây) |
| Traffic bị chrome lấn át | Splash / home / tab bar đứng đầu bảng token | Loại chrome + gộp lặp và vòng lặp |
            """,
        )
    )
