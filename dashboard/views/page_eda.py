"""Page 1 — Raw production data: what the clickstream actually looks like."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import PALETTE, fmt_int, fmt_pct, hist_to_df, kpi_row, load_eda, t


def _selected_files(eda: dict, platforms: list[str], weeks: list[str]) -> dict:
    return {
        k: v
        for k, v in eda["per_file"].items()
        if v["platform"] in platforms and v["week"] in weeks
    }


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
        "size": t("size (MB)", "dung lượng (MB)"),
        "events": t("events", "event"),
        "sessions": t("sessions", "session"),
        "devices": t("devices", "device"),
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
                L["size"]: f["size_mb"],
                L["events"]: f["rows"],
                L["sessions"]: f["n_sessions"],
                L["devices"]: f["n_devices"],
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
    st.caption(
        t(
            "The three findings below are the reason the pipeline canonizes screens into a "
            "governed token vocabulary instead of clustering the raw `segmentation_name` strings.",
            "Ba phát hiện dưới đây là lý do pipeline phải canonize screen thành một bộ token có "
            "kiểm soát, thay vì cluster trực tiếp trên chuỗi `segmentation_name` thô.",
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
indistinguishable from a typo. The pipeline folds rare tokens into a coarser level
(`min_journey_df = 3`, backoff to L2) precisely because of this shape.
                """,
                f"""
Trên các extract đã chọn, app phát ra nhiều nhất **{tax[C['vocab']].max():,}** giá trị
`segmentation_name` khác nhau — nhưng khoảng **{tax[C['singleton_share']].mean():.0%}**
trong số đó chỉ xuất hiện **đúng một lần**, trong khi 100 tên phổ biến nhất đã phủ
**~{tax[C['top100']].mean():.0%}** tổng số event.

**Hệ quả:** một vocabulary bằng chuỗi thô phần lớn là noise. Cluster trên đó sẽ tiêu tốn
năng lực mô hình vào những tên không bao giờ lặp lại, và mọi flow hiếm-nhưng-có-thật đều
không phân biệt được với một lỗi gõ. Pipeline gộp các token hiếm về mức thô hơn
(`min_journey_df = 3`, backoff sang L2) chính vì hình dạng phân phối này.
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
the ids differ. Without URL canonization and id masking
(`canonize_urls`, `mask_id_segments`) the vocabulary explodes and identical behaviour
never aligns. This is the single biggest driver of the singleton long tail above.
                """,
                """
Một phần lớn tên screen là **URL kèm query string, UUID và id dạng số**,
ví dụ `Home/166/1318/open_url_in_app_with_access_token/https://hi.fpt.vn/dkol/product-detail?merchantId=BH_CAM&packageId=combo-camera-3-v2`.

**Hệ quả:** hai user làm *cùng một việc* lại sinh ra *hai chuỗi khác nhau*, chỉ vì id khác nhau.
Nếu không canonize URL và mask id (`canonize_urls`, `mask_id_segments`), vocabulary sẽ bùng nổ
và các hành vi giống hệt nhau không bao giờ khớp được với nhau. Đây là nguyên nhân lớn nhất
tạo ra cái đuôi singleton ở tab trước.
                """,
            )
        )
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
                "ids and URLs live; the taxonomy truncates action paths to depth 3 (`action_path_depth_mid`).",
                "Độ sâu path của các tên screen khác nhau (6 = từ sáu `/` trở lên). Phần đuôi sâu chính là "
                "nơi chứa id và URL; taxonomy cắt path của action về độ sâu 3 (`action_path_depth_mid`).",
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
(Activities, ViewControllers), not from the *product*.

**Consequence:** platform-specific strings cannot be compared, aggregated, or reported on
together. The taxonomy layer maps both into a shared semantic space
(`business_family` / `business_module` / operation), which is what makes a single
cross-platform journey catalogue possible. Models are still fit **per platform**
(`namespace_by_os = true`), but their outputs land in one vocabulary.
                    """,
                    f"""
Chỉ **{ov['shared']:,}** tên screen được dùng chung giữa Android và iOS —
độ trùng Jaccard chỉ **{ov['jaccard']:.1%}**.

Cùng một màn hình sản phẩm nhưng Android gọi là `MainAppActivity` / `android/Home`, còn iOS gọi là
`MainTabBarController` / `HomeVC`. Tên sinh ra từ *cách implement* (Activity, ViewController),
không phải từ *sản phẩm*.

**Hệ quả:** chuỗi của từng platform không thể so sánh, cộng gộp hay report chung. Lớp taxonomy
ánh xạ cả hai về một không gian ngữ nghĩa chung (`business_family` / `business_module` / operation),
nhờ vậy mới có một catalogue journey duy nhất cho cả hai platform. Model vẫn được fit **riêng theo
từng platform** (`namespace_by_os = true`), nhưng output đổ về chung một vocabulary.
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

**Consequence:** sessions are segmented into journeys on idle gap (90 s), return-to-root,
auth change and a length cap. The gap distribution below is what sets that threshold.
            """,
            f"""
Phân phối lệch cực mạnh. Session trung vị chỉ có **~{med_len:.0f} event trong ~{med_span:.0f} giây**,
nhưng 1% session dài nhất lên tới **~{p99_len:.0f} event kéo dài ~{p99_span/3600:.0f} giờ**.
Những session dài đó không phải là một task dài — một `session_id` gói *nhiều task không liên quan*
cách nhau bởi những khoảng nghỉ dài: thanh toán hóa đơn buổi sáng và kiểm tra Wi-Fi buổi tối
nằm chung một dòng.

**Hệ quả:** session được cắt thành journey theo idle gap (90 giây), quay về root, đổi trạng thái
auth và giới hạn độ dài. Phân phối gap bên dưới chính là căn cứ chọn ngưỡng đó.
            """,
        )
    )

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
            "the small tail beyond 90 s is where one task ends and another begins.",
            "Khoảng cách giữa hai event liên tiếp trong cùng session. Gần như mọi bước diễn ra trong "
            "vài giây; phần đuôi nhỏ vượt 90 giây chính là chỗ một task kết thúc và task khác bắt đầu.",
        )
    )

    st.subheader(t("Session size distribution", "Phân phối kích thước session"))
    pick = st.selectbox("Extract", list(files.keys()), key="len_hist_pick")
    hdf = hist_to_df(files[pick]["session_len_hist"], t("sessions", "session"))
    fig = px.bar(hdf, x="bin", y=t("sessions", "session"), color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(
        height=300, margin=dict(t=10, b=10),
        xaxis_title=t("events per session (clipped at 400)", "event mỗi session (cắt ở 400)"),
        yaxis_title=t("sessions", "session"),
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # ------------------------------------------------------------------ tokens & volume
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

    a, b = st.columns(2)
    day_col = t("day", "ngày")
    vol_rows = []
    for k, fl in files.items():
        for day, n in fl["events_per_day"].items():
            vol_rows.append({C["file"]: k, day_col: day, ev_col: n})
    vd = pd.DataFrame(vol_rows)
    vd[day_col] = pd.to_datetime(vd[day_col])
    fig = px.area(vd.sort_values(day_col), x=day_col, y=ev_col, color=C["file"], color_discrete_sequence=PALETTE)
    fig.update_layout(height=320, margin=dict(t=10, b=10), legend_title=None, xaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    a.caption(t("Daily event volume per extract.", "Lượng event theo ngày của từng extract."))

    hour_col = t("hour (UTC)", "giờ (UTC)")
    share_col = t("share of events", "tỷ lệ event")
    hr_rows = []
    for k, fl in files.items():
        total = sum(fl["hour_hist"].values()) or 1
        for h, n in fl["hour_hist"].items():
            hr_rows.append({C["file"]: k, hour_col: int(h), share_col: n / total})
    hd = pd.DataFrame(hr_rows)
    fig = px.line(hd.sort_values(hour_col), x=hour_col, y=share_col, color=C["file"], markers=True, color_discrete_sequence=PALETTE)
    fig.update_layout(height=320, margin=dict(t=10, b=10), legend_title=None, yaxis_tickformat=".1%")
    b.plotly_chart(fig, width="stretch")
    b.caption(t("Hour-of-day profile (UTC; local time is UTC+7).", "Phân bố theo giờ trong ngày (UTC; giờ Việt Nam là UTC+7)."))

    st.divider()
    st.subheader(t("What the raw data tells us — summary", "Dữ liệu thô nói lên điều gì — tóm tắt"))
    st.markdown(
        t(
            """
| Finding | Evidence | What it forces the pipeline to do |
| --- | --- | --- |
| No task/flow concept exists in the schema | Only `key` + free-text `segmentation_name` | Reconstruct journeys unsupervised from sequences |
| Screen names are an implementation artefact | ~9% vocabulary overlap between Android and iOS | Canonize into a governed taxonomy; fit per platform |
| Names embed URLs, UUIDs and record ids | Majority of distinct names contain a URL or query string | URL canonization + id masking before tokenizing |
| Half the vocabulary is seen once | Singleton share ≈ 30–51%, top-100 names cover ~90% of events | Rare-token folding with backoff to a coarser level |
| Sessions mix unrelated tasks | Median session ≈ minutes, p99 ≈ hours | Re-segment sessions into journeys (90 s idle gap, root return, auth change) |
| Traffic is dominated by chrome | Splash / home / tab-bar top the token ranking | Drop chrome + collapse repeats and loops |
            """,
            """
| Phát hiện | Bằng chứng | Buộc pipeline phải làm gì |
| --- | --- | --- |
| Schema không có khái niệm task/flow | Chỉ có `key` + `segmentation_name` dạng text tự do | Tái dựng journey theo hướng unsupervised từ chuỗi event |
| Tên screen là sản phẩm của cách implement | Vocabulary Android và iOS chỉ trùng ~9% | Canonize về taxonomy có kiểm soát; fit riêng từng platform |
| Tên chứa URL, UUID và id bản ghi | Đa số tên khác nhau có chứa URL hoặc query string | Canonize URL + mask id trước khi tokenize |
| Một nửa vocabulary chỉ xuất hiện 1 lần | Tỷ lệ singleton ≈ 30–51%, top-100 tên phủ ~90% event | Gộp token hiếm, backoff về mức thô hơn |
| Session trộn nhiều task không liên quan | Session trung vị tính bằng phút, p99 tính bằng giờ | Cắt session thành journey (idle gap 90 giây, về root, đổi auth) |
| Traffic bị chrome lấn át | Splash / home / tab bar đứng đầu bảng token | Loại chrome + gộp lặp và vòng lặp |
            """,
        )
    )
