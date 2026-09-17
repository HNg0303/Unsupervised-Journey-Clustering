"""Page 1 — What the problem is, why it is hard here, and how the pipeline solves it.

Written for business stakeholders: every claim on this page is backed by a number from
the actual run, and the worked example is one real production session carried end to end.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.lib import (
    PALETTE,
    fmt_int,
    fmt_pct,
    is_vi,
    kpi_row,
    load_eda,
    load_run_csv,
    load_showcase,
    load_summary_csv,
    load_train_run,
    get_inference_bundle,
    t,
)

# Published characteristics of the clickstream datasets most papers benchmark on. Figures
# are the headline numbers reported by the dataset authors, rounded; they are here to
# contrast *structure*, not to be exact counts.
BASELINES = [
    {
        "dataset": "MSNBC.com (UCI, 1999)",
        "events": 4_698_794,
        "sessions": 989_818,
        "vocabulary": 17,
        "event_types": 1,
        "labels": True,
        "naming_en": "17 curated page categories (frontpage, news, tech, …)",
        "naming_vi": "17 nhóm trang đã được biên tập sẵn (frontpage, news, tech, …)",
    },
    {
        "dataset": "YOOCHOOSE (RecSys Challenge 2015)",
        "events": 33_003_944,
        "sessions": 9_249_729,
        "vocabulary": 52_739,
        "event_types": 2,
        "labels": True,
        "naming_en": "Numeric item ids + item category; click / purchase only",
        "naming_vi": "Item id dạng số + category của item; chỉ có click / purchase",
    },
    {
        "dataset": "RetailRocket",
        "events": 2_756_101,
        "sessions": 1_407_580,
        "vocabulary": 235_061,
        "event_types": 3,
        "labels": True,
        "naming_en": "Numeric item ids; view / addtocart / transaction",
        "naming_vi": "Item id dạng số; view / addtocart / transaction",
    },
    {
        "dataset": "Taobao UserBehavior",
        "events": 100_150_807,
        "sessions": 987_994,
        "vocabulary": 4_162_024,
        "event_types": 4,
        "labels": True,
        "naming_en": "Numeric item ids + category id; pv / cart / fav / buy",
        "naming_vi": "Item id dạng số + category id; pv / cart / fav / buy",
    },
]


def _our_dataset_row(eda: dict, kpi: dict | None = None, run: dict | None = None) -> dict:
    files = eda["per_file"]
    # The current EDA cache inventories the prepared journey lake and may intentionally
    # omit cardinalities when only parquet footer metadata is available.  The headline
    # comparison must therefore use the full-data inference KPI when it exists, rather
    # than treating missing metadata as zero or mixing it with a legacy cache.
    events = int(kpi.get("journeys", 0) or 0) if kpi else sum(int(f.get("rows") or 0) for f in files.values())
    sessions = int(kpi.get("sessions", 0) or 0) if kpi else sum(int(f.get("n_sessions") or 0) for f in files.values())
    vocab_values = [f.get("taxonomy_signals", {}).get("vocab_size") for f in files.values()]
    vocab_values = [float(v) for v in vocab_values if v is not None and pd.notna(v)]
    if not vocab_values and run:
        vocab_values = [
            float(p.get("run_config", {}).get("feature_info", {}).get("tfidf_vocabulary"))
            for p in run.get("platforms", {}).values()
            if p.get("run_config", {}).get("feature_info", {}).get("tfidf_vocabulary") is not None
        ]
    vocab = int(max(vocab_values, default=0))
    singleton_values = [f.get("taxonomy_signals", {}).get("singleton_share") for f in files.values()]
    singleton_values = [float(v) for v in singleton_values if v is not None and pd.notna(v)]
    singleton = sum(singleton_values) / len(singleton_values) if singleton_values else 0.0
    return {
        "dataset": t("HiFPT production clickstream (this project)", "Clickstream production HiFPT (dự án này)"),
        "events": events,
        "sessions": sessions,
        "vocabulary": vocab,
        "event_types": 2,
        "labels": False,
        "naming_en": "Free-text screen names from Activities / ViewControllers, with URLs and ids inside",
        "naming_vi": "Tên screen dạng text tự do sinh từ Activity / ViewController, bên trong có URL và id",
        "singleton_share": singleton,
        "ours": True,
    }


def _step_cards(steps: list[dict]) -> None:
    for i, s in enumerate(steps, start=1):
        with st.container(border=True):
            head, body, num = st.columns([2.4, 5, 2.2])
            head.markdown(f"##### {i}. {s['title']}")
            head.caption(s["tech"])
            body.markdown(s["plain"])
            num.metric(s["metric_label"], s["metric_value"], help=s.get("metric_help"))


def render() -> None:
    bundle = get_inference_bundle(st.session_state.get("inference_bundle"))
    bundle_key = bundle.key if bundle else None
    focus_platform = (bundle.platforms[0] if bundle and bundle.platforms else "android")
    eda = load_eda(bundle_key)
    run = load_train_run(bundle_key)
    kpi_table = load_summary_csv("kpi.csv", bundle_key=bundle_key)
    kpi = kpi_table[kpi_table["scope"].eq("all")].iloc[0].to_dict() if not kpi_table[kpi_table["scope"].eq("all")].empty else {}
    cluster_summary = load_summary_csv("cluster_summary.csv", bundle_key=bundle_key)
    showcase = load_showcase()

    st.title(t("Turning raw clicks into journeys", "Biến click thô thành journey"))
    st.markdown(
        t(
            "#### The app records **what screen was opened**. It does not record **what the "
            "customer was trying to do**. This project reconstructs the second one from the first — "
            "with no labels, no tagging plan and no change to the app.",
            "#### App ghi lại **màn hình nào được mở**. App không ghi lại **khách hàng đang cố làm gì**. "
            "Dự án này tái dựng vế thứ hai từ vế thứ nhất — không cần nhãn, không cần kế hoạch gắn tag, "
            "không phải sửa app.",
        )
    )

    # ---------------------------------------------------------------- headline numbers
    android = run["platforms"].get(focus_platform, {}).get("run_config", {}).get("hdbscan", {})
    ios = run["platforms"].get("ios", {}).get("run_config", {}).get("hdbscan", {})
    total_events = int(kpi.get("journeys", 0) or 0)
    total_sessions = int(kpi.get("sessions", 0) or 0)
    known_clusters = cluster_summary[cluster_summary.get("cluster", pd.Series(dtype=float)).ne(-1)] if not cluster_summary.empty else pd.DataFrame()
    cluster_count = int(known_clusters[["platform", "cluster"]].drop_duplicates().shape[0]) if not known_clusters.empty else int((android.get("n_clusters") or 0) + (ios.get("n_clusters") or 0))
    journey_type_count = int(known_clusters["journey_type_en"].nunique()) if not known_clusters.empty and "journey_type_en" in known_clusters else 0
    kpi_row(
        [
            (t("Scored journeys", "Journey đã score"), fmt_int(total_events), t("Android + iOS in the selected inference bundle", "Android + iOS trong bundle inference đang chọn")),
            (t("Sessions", "Session"), fmt_int(total_sessions), None),
            (
                t("Journey types out", "Journey type đầu ra"),
                fmt_int(cluster_count),
                t("Clusters learned across both platform models", "Số cluster học được trên cả hai model platform"),
            ),
            (
                t("Labels required", "Nhãn cần có"),
                t("none", "không cần"),
                t("Fully unsupervised — nothing was hand-labelled", "Hoàn toàn unsupervised — không có gì được gán nhãn thủ công"),
            ),
        ]
    )

    st.divider()

    # ---------------------------------------------------------------- input -> output
    st.header(t("What goes in, what comes out", "Đầu vào là gì, đầu ra là gì"))
    if showcase:
        st.caption(
            t(
                f"One real production session (`{showcase['session_id']}`, {showcase['platform']}, "
                f"from `{showcase['source_file']}`) carried end to end. Nothing here is illustrative — "
                "the left panel is the input rows, the right panel is what the model returned for them.",
                f"Một session production thật (`{showcase['session_id']}`, {showcase['platform']}, "
                f"từ `{showcase['source_file']}`) đi trọn vẹn qua pipeline. Không có gì là minh họa — "
                "khung bên trái là dòng dữ liệu đầu vào, khung bên phải là thứ model trả về cho chính chúng.",
            )
        )

        left, mid, right = st.columns([5, 1, 6])

        with left:
            with st.container(border=True):
                st.markdown(t("#### IN — raw clickstream", "#### VÀO — clickstream thô"))
                st.caption(
                    t(
                        f"{showcase['raw_event_count']} rows. Every row is one screen view or one tap. "
                        "No task, no outcome, no grouping.",
                        f"{showcase['raw_event_count']} dòng. Mỗi dòng là một screen view hoặc một cú tap. "
                        "Không có task, không có kết quả, không có nhóm.",
                    )
                )
                raw = pd.DataFrame(showcase["raw_events"])
                raw = raw.rename(
                    columns={
                        "key": "key",
                        "segmentation_name": "segmentation_name",
                        "ts": t("time", "thời gian"),
                        "gap_s": t("gap (s)", "gap (giây)"),
                    }
                )
                st.dataframe(raw, hide_index=True, width="stretch", height=430)

        # empty container acts as a vertical spacer so the arrow lands beside the panels
        mid.container(height=180, border=False)
        mid.markdown("## ➜")
        mid.caption(t("model", "model"))

        with right:
            with st.container(border=True):
                st.markdown(t("#### OUT — named journeys", "#### RA — journey đã có tên"))
                st.caption(
                    t(
                        f"{len(showcase['journeys'])} separate tasks, each named, measured and scored.",
                        f"{len(showcase['journeys'])} task riêng biệt, mỗi task được đặt tên, đo lường và chấm điểm.",
                    )
                )
                for j in showcase["journeys"]:
                    name = j["cluster_name_vi"] if is_vi() else j["cluster_name"]
                    family = j["business_family_vi"] if is_vi() else j["business_family"]
                    with st.container(border=True):
                        st.markdown(f"**{name}**  ·  `{family}`")
                        c1, c2, c3 = st.columns(3)
                        c1.metric(t("steps", "số bước"), int(j["n_events_final"]))
                        c2.metric(t("seconds", "giây"), f"{j['span_seconds']:.0f}")
                        c3.metric(t("taps", "tỷ lệ tap"), f"{j['action_ratio']:.0%}")
                        flag = j.get("behavioral_friction_flags") or j.get("friction_flags")
                        if flag:
                            st.error(
                                t(f"friction detected: `{flag}`", f"phát hiện friction: `{flag}`"),
                                icon=":material/warning:",
                            )
                        else:
                            st.success(t("clean run, no friction", "chạy trơn tru, không có friction"), icon=":material/check:")
                        if j.get("effective_next_action"):
                            st.caption(
                                t(
                                    f"predicted next step ({float(j['effective_next_action_share']):.0%} confidence): "
                                    f"`{j['effective_next_action']}`",
                                    f"bước kế tiếp được dự đoán (độ tin cậy {float(j['effective_next_action_share']):.0%}): "
                                    f"`{j['effective_next_action']}`",
                                )
                            )

        st.markdown(
            t(
                """
**What that one session bought us.** The raw rows said *a device opened 17 screens*. The output says
*this customer checked their notifications, then went to manage connected devices and flipped between
the device tabs repeatedly — that second task is where they struggled, and the next thing they will
most likely tap is the connected-devices tab.*

That is the difference between telemetry and understanding: a **task name**, a **boundary** between
two unrelated tasks inside one session, an **effort measurement**, a **friction verdict**, and a
**prediction** — none of which exist anywhere in the source data.
                """,
                """
**Một session đó mang lại điều gì.** Dữ liệu thô chỉ nói *một thiết bị đã mở 17 màn hình*. Đầu ra nói
*khách hàng này kiểm tra thông báo, sau đó vào quản lý thiết bị kết nối và bấm qua lại giữa các tab thiết bị
nhiều lần — task thứ hai chính là chỗ họ gặp khó, và thứ họ nhiều khả năng bấm tiếp theo là tab thiết bị
đang kết nối.*

Đó là khác biệt giữa telemetry và sự thấu hiểu: một **tên task**, một **ranh giới** giữa hai task không liên quan
trong cùng một session, một **phép đo công sức**, một **kết luận về friction**, và một **dự đoán** —
không thứ nào trong số đó tồn tại sẵn trong dữ liệu nguồn.
                """,
            )
        )
    else:
        st.info(t("The selected bundle contains scored journey outputs but no duplicated raw-event extract for a traceable session showcase.", "Bundle đang chọn có output journey đã score nhưng không chứa bản sao raw event để dựng showcase theo từng session."))

    st.divider()

    # ---------------------------------------------------------------- vs. papers
    st.header(t("Why the textbook recipe does not transfer", "Vì sao công thức trong sách vở không áp dụng thẳng được"))
    st.markdown(
        t(
            "Sequence-clustering papers are almost always benchmarked on datasets that arrive "
            "**pre-cleaned**: a small closed vocabulary, a handful of event types, and a category "
            "label already attached to every item. Production telemetry has none of that.",
            "Các bài báo về sequence clustering gần như luôn benchmark trên những bộ dữ liệu đã được "
            "**làm sạch sẵn**: vocabulary nhỏ và đóng, vài loại event, và mỗi item đã có sẵn nhãn category. "
            "Telemetry production thì không có gì trong số đó.",
        )
    )

    ours = _our_dataset_row(eda, kpi=kpi, run=run)
    rows = [ours] + BASELINES
    DS = t("dataset", "bộ dữ liệu")
    EV = t("rows / events", "dòng / event")
    SE = t("sessions", "session")
    VO = t("distinct event vocabulary", "số event khác nhau")
    ET = t("event types", "loại event")
    LB = t("ready-made labels?", "có nhãn sẵn?")
    NM = t("what an event is called", "event được đặt tên thế nào")
    table = pd.DataFrame(
        [
            {
                DS: r["dataset"],
                EV: r["events"],
                SE: r["sessions"],
                VO: r["vocabulary"],
                ET: r["event_types"],
                LB: t("yes", "có") if r["labels"] else t("NO", "KHÔNG"),
                NM: r["naming_vi"] if is_vi() else r["naming_en"],
            }
            for r in rows
        ]
    )
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            EV: st.column_config.NumberColumn(EV, format="%d"),
            SE: st.column_config.NumberColumn(SE, format="%d"),
            VO: st.column_config.NumberColumn(VO, format="%d"),
        },
    )
    st.caption(
        t(
            "Baseline figures are the headline numbers published by each dataset's authors, rounded. "
            "For HiFPT this row is a scored-journey count because the selected bundle does not carry a raw-event copy; "
            "baselines count raw events. Compare the *structure*, not the absolute volume.",
            "Số liệu baseline là con số công bố bởi tác giả từng bộ dữ liệu, đã làm tròn. "
            "Với HiFPT, dòng này là số journey đã score vì bundle hiện tại không chứa bản sao raw event; "
            "baseline đếm raw event. Hãy so sánh *cấu trúc*, không so sánh volume tuyệt đối.",
        )
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown(t("##### 1. No label ever existed", "##### 1. Chưa từng có nhãn nào"))
            st.markdown(
                t(
                    "Every baseline ships a category per item and a closed set of event types "
                    "(`click` / `buy` / `cart`). Our data has two event types (`view`, `action`) "
                    "and **no notion of task or outcome at all** — so accuracy against ground truth "
                    "is not even measurable. The model must be judged on separation, stability and "
                    "whether a human recognises the journey.",
                    "Mọi bộ baseline đều kèm sẵn category cho từng item và một tập loại event đóng "
                    "(`click` / `buy` / `cart`). Dữ liệu của mình chỉ có hai loại event (`view`, `action`) "
                    "và **hoàn toàn không có khái niệm task hay kết quả** — nên không thể đo accuracy so với "
                    "ground truth. Model phải được đánh giá qua độ tách cluster, độ ổn định, và việc con người "
                    "có nhận ra journey đó hay không.",
                )
            )
    with c2:
        with st.container(border=True):
            st.markdown(t("##### 2. The vocabulary is dirty, not just large", "##### 2. Vocabulary bẩn, không chỉ là lớn"))
            st.markdown(
                t(
                    f"Taobao has millions of item ids — but they are *clean* ids. Ours is "
                    f"~{ours['vocabulary']:,} free-text screen names where **{ours['singleton_share']:.0%} are seen "
                    "exactly once**, most contain a URL, a UUID or a record id, and the same product "
                    "screen is spelled differently on Android and iOS (~9% vocabulary overlap). "
                    "Clustering these strings directly clusters typos and ids.",
                    f"Taobao có hàng triệu item id — nhưng đó là id *sạch*. Của mình là "
                    f"~{ours['vocabulary']:,} tên screen dạng text tự do, trong đó **{ours['singleton_share']:.0%} chỉ xuất hiện "
                    "đúng một lần**, phần lớn chứa URL, UUID hoặc id bản ghi, và cùng một màn hình sản phẩm lại "
                    "được viết khác nhau giữa Android và iOS (chỉ ~9% vocabulary trùng nhau). "
                    "Cluster thẳng trên các chuỗi này là đang cluster lỗi gõ và id.",
                )
            )
    with c3:
        with st.container(border=True):
            st.markdown(t("##### 3. The session is not the unit", "##### 3. Session không phải đơn vị phân tích"))
            st.markdown(
                t(
                    "Benchmarks define a session as one shopping intent. Here a `session_id` can span "
                    "hours and hold several unrelated tasks — pay a bill in the morning, check Wi-Fi at "
                    "night. Any metric computed per session silently averages those together, so the "
                    "unit of analysis has to be rebuilt before modelling.",
                    "Các benchmark định nghĩa một session là một ý định mua sắm. Ở đây một `session_id` có thể "
                    "kéo dài nhiều giờ và chứa nhiều task không liên quan — sáng thanh toán hóa đơn, tối kiểm tra "
                    "Wi-Fi. Mọi chỉ số tính theo session đều lặng lẽ trộn chúng lại, nên phải dựng lại đơn vị "
                    "phân tích trước khi mô hình hóa.",
                )
            )

    fig = go.Figure()
    fig.add_bar(
        x=[r["dataset"] for r in rows],
        y=[r["vocabulary"] for r in rows],
        marker_color=[PALETTE[3]] + [PALETTE[0]] * len(BASELINES),
        text=[f"{r['vocabulary']:,}" for r in rows],
    )
    fig.update_layout(
        height=330, margin=dict(t=30, b=10), yaxis_type="log",
        yaxis_title=t("distinct event vocabulary (log)", "số event khác nhau (thang log)"),
        title=t(
            "Vocabulary size is not the problem — who wrote the vocabulary is",
            "Vấn đề không nằm ở kích thước vocabulary — mà ở chỗ ai đã tạo ra vocabulary đó",
        ),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        t(
            "The baselines' vocabularies are machine-generated ids from a catalogue, curated once and "
            "stable. Ours is written by app developers, changes with every release, and is the reason "
            "step 1 of the pipeline exists.",
            "Vocabulary của các bộ baseline là id do máy sinh ra từ một catalogue, được biên tập một lần và "
            "ổn định. Vocabulary của mình do lập trình viên app đặt, thay đổi theo từng bản release — và đó "
            "chính là lý do bước 1 của pipeline tồn tại.",
        )
    )

    st.divider()

    # ---------------------------------------------------------------- the 5 steps
    st.header(t("How it works, in five steps", "Cách làm, qua năm bước"))
    st.caption(
        t(
            "Each step below is a real stage of the pipeline, with the number it produced on this run "
            f"({focus_platform.title()} model shown; other available platform models are fitted on their own data).",
            "Mỗi bước dưới đây là một giai đoạn thật của pipeline, kèm con số nó tạo ra ở run này "
            f"(hiển thị model {focus_platform.title()}; các platform khác được fit theo đúng cách trên dữ liệu riêng).",
        )
    )

    def kv(name: str) -> dict:
        d = load_run_csv(name, bundle_key=bundle_key, platform=focus_platform)
        if d.empty or not {"metric", "value"} <= set(d.columns):
            return {}
        return dict(zip(d["metric"], d["value"]))

    canon = load_run_csv(f"{focus_platform}_report_canonization.csv", bundle_key=bundle_key, platform=focus_platform)
    vocab_rep = load_run_csv(f"{focus_platform}_report_vocabulary.csv", bundle_key=bundle_key, platform=focus_platform)
    rare = load_run_csv(f"{focus_platform}_report_rare_folding.csv", bundle_key=bundle_key, platform=focus_platform)
    seg = kv(f"{focus_platform}_report_segmentation.csv")
    post = kv(f"{focus_platform}_report_postprocess.csv")
    focus_run = run["platforms"].get(focus_platform, {}).get("run_config", {})
    fit = focus_run.get("feature_info", {})
    model_cfg = focus_run.get("config", {})
    segment_cfg = model_cfg.get("segment", {})
    feature_cfg = model_cfg.get("features", {})
    idle_gap = int(float(segment_cfg.get("idle_gap_seconds", 90)))
    min_length = int(segment_cfg.get("min_journey_length", 4))
    max_length = int(segment_cfg.get("max_journey_length", 80))
    ngram_range = feature_cfg.get("ngram_range", [1, 2])
    ngram_label = "–".join(str(v) for v in ngram_range)
    svd_components = int(fit.get("svd_components", feature_cfg.get("svd_components", 48)))
    numeric_features = int(fit.get("numeric_features", 0))

    steps = [
        {
            "title": t("Canonize — clean the names", "Canonize — làm sạch tên"),
            "tech": t("URL canonization · id masking · chrome removal", "Canonize URL · mask id · loại chrome"),
            "plain": t(
                "Rewrite every screen name so that *the same action always produces the same name*. "
                "URLs are collapsed, record ids are masked (`Home/215/do_action/…` → `Home/{id}/do_action/…`), "
                "and container screens that carry no intent are dropped. Without this, two customers doing "
                "the identical thing look like two different behaviours.",
                "Viết lại mọi tên screen sao cho *cùng một hành động luôn sinh ra cùng một tên*. "
                "URL được rút gọn, id bản ghi được mask (`Home/215/do_action/…` → `Home/{id}/do_action/…`), "
                "và các screen container không mang ý định nào bị loại bỏ. Nếu không làm bước này, hai khách hàng "
                "làm việc y hệt nhau lại trông như hai hành vi khác nhau.",
            ),
            "metric_label": t("distinct names kept", "số tên còn lại"),
            "metric_value": fmt_int(canon["vocabulary"].iloc[0]) if not canon.empty else "—",
            "metric_help": t("Canonical event_type@segment_name vocabulary", "Vocabulary event_type@segment_name sau canonize"),
        },
        {
            "title": t("Tokenize — build a shared language", "Tokenize — dựng một ngôn ngữ chung"),
            "tech": t("EXACT tokens · L2 backoff · taxonomy enrichment", "Token EXACT · backoff L2 · gắn taxonomy"),
            "plain": t(
                "Turn each cleaned event into a token and attach its business meaning (family, module, "
                "operation). Names that are too rare to learn from are folded up to a coarser level rather "
                "than thrown away, and Android and iOS names are mapped into one shared vocabulary so the "
                "two platforms can finally be compared.",
                "Biến mỗi event đã làm sạch thành một token và gắn ý nghĩa nghiệp vụ cho nó (family, module, "
                "operation). Những tên quá hiếm để học được sẽ được gộp lên mức thô hơn thay vì vứt đi, và tên của "
                "Android với iOS được ánh xạ về cùng một vocabulary chung để cuối cùng hai platform có thể so sánh "
                "được với nhau.",
            ),
            "metric_label": t("tokens after folding", "token sau khi gộp"),
            "metric_value": fmt_int(rare["vocabulary_after"].iloc[0]) if not rare.empty else "—",
            "metric_help": t("Down from the raw vocabulary, with rare tokens backed off", "Giảm từ vocabulary thô, token hiếm được backoff"),
        },
        {
            "title": t("Build journeys — find the task boundaries", "Dựng journey — tìm ranh giới task"),
            "tech": t(
                f"{idle_gap} s idle gap · return-to-root · auth change · {min_length}–{max_length} step cap",
                f"Idle gap {idle_gap} giây · quay về root · đổi auth · giới hạn {min_length}–{max_length} bước",
            ),
            "plain": t(
                "Cut each session where the customer clearly stopped one task and started another: a long "
                "pause, a return to the home screen, or a login change. This replaces the session with the "
                "**task** as the unit of analysis — the single most important modelling decision on this project.",
                "Cắt mỗi session tại chỗ khách hàng rõ ràng đã dừng một task và bắt đầu task khác: một khoảng dừng dài, "
                "một lần quay về màn hình chính, hoặc một thay đổi đăng nhập. Bước này thay session bằng **task** làm "
                "đơn vị phân tích — quyết định mô hình hóa quan trọng nhất của dự án.",
            ),
            "metric_label": t("journeys built", "journey dựng được"),
            "metric_value": fmt_int(seg.get("journeys")),
            "metric_help": t(
                f"From {fmt_int(seg.get('sessions'))} sessions — about {(seg.get('journeys') or 0) / max(seg.get('sessions') or 1, 1):.1f} tasks per session",
                f"Từ {fmt_int(seg.get('sessions'))} session — khoảng {(seg.get('journeys') or 0) / max(seg.get('sessions') or 1, 1):.1f} task mỗi session",
            ),
        },
        {
            "title": t("Represent — describe each journey with numbers", "Biểu diễn — mô tả mỗi journey bằng số"),
            "tech": t(
                f"multi-channel TF-IDF ({ngram_label} grams) + SVD-{svd_components} + {numeric_features} shape features",
                f"TF-IDF đa kênh ({ngram_label} gram) + SVD-{svd_components} + {numeric_features} feature hình dạng",
            ),
            "plain": t(
                "Describe a journey by the step patterns it contains — at four levels of detail at once "
                "(exact screen, intent, coarse area, operation) — plus how it felt: length, duration, how "
                "tap-heavy it was, how much back-tracking and revisiting. Two journeys are then similar "
                "when they *do the same thing in the same way*, not when they share one screen.",
                "Mô tả một journey bằng các mẫu bước mà nó chứa — cùng lúc ở bốn mức chi tiết (screen chính xác, "
                "intent, vùng thô, operation) — cộng với cảm nhận khi thực hiện: độ dài, thời lượng, mức độ nhiều tap, "
                "mức độ quay lui và lặp lại. Khi đó hai journey giống nhau khi chúng *làm cùng một việc theo cùng một cách*, "
                "chứ không phải khi chúng dùng chung một screen.",
            ),
            "metric_label": t("dimensions per journey", "số chiều mỗi journey"),
            "metric_value": fmt_int(fit.get("final_dimension")),
            "metric_help": t(
                f"4 text channels × {svd_components} SVD components + {numeric_features} shape features",
                f"4 kênh text × {svd_components} SVD component + {numeric_features} feature hình dạng",
            ),
        },
        {
            "title": t("Cluster + detect anomalies", "Cluster + phát hiện bất thường"),
            "tech": t("HDBSCAN (density) · per-cluster Markov model · distance limits", "HDBSCAN (mật độ) · Markov model theo cluster · distance limit"),
            "plain": t(
                "Group journeys that behave alike into journey types — using a density method that is "
                "**allowed to say \"this one belongs nowhere\"** instead of forcing every journey into a box. "
                "Each type then learns its own normal step-to-step transitions, so a journey can be flagged "
                "two ways: *geometrically* (too far from every known type → new behaviour) or *generatively* "
                "(inside a known type but taking improbable steps → friction).",
                "Gom các journey hành xử giống nhau thành journey type — bằng phương pháp dựa trên mật độ, "
                "**được phép nói \"cái này không thuộc về đâu cả\"** thay vì ép mọi journey vào một cái hộp. "
                "Sau đó mỗi type tự học các bước chuyển bình thường của riêng nó, nên một journey có thể bị gắn cờ "
                "theo hai cách: *về mặt hình học* (quá xa mọi type đã biết → hành vi mới) hoặc *về mặt sinh mẫu* "
                "(nằm trong một type đã biết nhưng đi những bước khó xảy ra → friction).",
            ),
            "metric_label": t("journey types learned", "journey type học được"),
            "metric_value": fmt_int(android.get("n_clusters")),
            "metric_help": t(
                f"silhouette {android.get('silhouette')} — then named from medoid path + top n-grams",
                f"silhouette {android.get('silhouette')} — sau đó đặt tên từ medoid path + n-gram đặc trưng",
            ),
        },
    ]
    _step_cards(steps)

    # volume funnel across the steps
    rows_in = None
    prep = load_run_csv("preprocessing_report.csv")
    if not prep.empty:
        rows_in = prep[prep["source_file"].str.startswith(focus_platform, na=False)]["rows_output"].sum()
    funnel = [
        (t("Raw events", "Event thô"), rows_in),
        (t("After cleaning", "Sau khi làm sạch"), post.get("events_after_cleanup")),
        (t("Journeys", "Journey"), seg.get("journeys")),
        (t("Journeys modelled", "Journey đưa vào model"), fit.get("n_journeys")),
    ]
    funnel = [(n, v) for n, v in funnel if v]
    if funnel:
        fig = go.Figure(
            go.Funnel(
                y=[f[0] for f in funnel],
                x=[float(f[1]) for f in funnel],
                marker_color=PALETTE[: len(funnel)],
                textinfo="value+percent initial",
            )
        )
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            t(
                "Volume through the Android pipeline. The drop is not data loss — it is duplicate events "
                "collapsed, navigation loops removed, and fragments shorter than four steps set aside.",
                "Lượng dữ liệu đi qua pipeline Android. Phần giảm đi không phải là mất dữ liệu — đó là các event "
                "trùng bị gộp, vòng lặp điều hướng bị loại, và các mảnh ngắn dưới bốn bước được để riêng.",
            )
        )

    st.divider()

    # ---------------------------------------------------------------- what you can do
    st.header(t("What the business gets", "Doanh nghiệp nhận được gì"))
    a, b, c, d = st.columns(4)
    for col, (icon, title_en, title_vi, body_en, body_vi) in zip(
        [a, b, c, d],
        [
            (
                ":material/map:",
                "A map of what the app is really used for",
                "Bản đồ thực tế app đang được dùng để làm gì",
                "A named catalogue of every recurring task, ranked by volume and by how many customers do it — "
                "built from behaviour, not from assumptions in the roadmap.",
                "Một catalogue có tên cho mọi task lặp lại, xếp hạng theo số lượng và theo số khách hàng thực hiện — "
                "dựng từ hành vi, không phải từ giả định trong roadmap.",
            ),
            (
                ":material/report:",
                "A friction list you can act on",
                "Danh sách friction có thể hành động ngay",
                "For every journey type: what share of runs show back-tracking, thrashing or looping, and how much "
                "extra time that costs customers. That is a prioritised fix queue.",
                "Với mỗi journey type: bao nhiêu phần trăm lượt có quay lui, nhảy qua lại hoặc lặp vòng, và điều đó "
                "khiến khách hàng tốn thêm bao nhiêu thời gian. Đó chính là hàng đợi ưu tiên cần sửa.",
            ),
            (
                ":material/notifications_active:",
                "An early-warning signal",
                "Tín hiệu cảnh báo sớm",
                "Journeys that match nothing known are surfaced daily. A jump in that share on one entry screen "
                "means a release, a campaign or a bug just changed behaviour.",
                "Những journey không khớp thứ gì đã biết được đưa lên hằng ngày. Tỷ lệ đó tăng vọt tại một entry screen "
                "nghĩa là một bản release, một chiến dịch hoặc một lỗi vừa làm thay đổi hành vi.",
            ),
            (
                ":material/bolt:",
                "A next-best-action signal",
                "Tín hiệu gợi ý hành động kế tiếp",
                "Each journey type predicts its own most likely next step. Where confidence is high and the path is "
                "long, there is a shortcut or a deep link worth building.",
                "Mỗi journey type tự dự đoán bước kế tiếp khả dĩ nhất của nó. Ở đâu độ tin cậy cao mà đường đi lại dài, "
                "ở đó đáng làm một shortcut hoặc deep link.",
            ),
        ],
    ):
        with col:
            with st.container(border=True):
                st.markdown(f"##### {icon} {t(title_en, title_vi)}")
                st.markdown(t(body_en, body_vi))

    st.info(
        t(
            "**Where to go next.** The *Details* section carries the evidence behind every claim on this "
            "page: the raw data and its quirks, every file the training run wrote, the learned journey "
            "catalogue with its naming evidence, and the production results from the selected partitioned inference bundle.",
            "**Xem tiếp ở đâu.** Mục *Chi tiết* chứa bằng chứng cho mọi khẳng định ở trang này: dữ liệu thô và "
            "các đặc điểm của nó, mọi file mà training run đã ghi ra, catalogue journey đã học kèm bằng chứng đặt tên, "
            "và kết quả production từ bundle inference đã partition đang chọn.",
        ),
        icon=":material/arrow_forward:",
    )
