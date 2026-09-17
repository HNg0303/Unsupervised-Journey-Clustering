"""Page 2 — Every artefact the training run wrote, and what it says."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.lib import (
    PALETTE,
    RUN_DIR,
    bundle_model_dir,
    bundle_inspection_dir,
    bundle_platform_root,
    bundle_training_root,
    fmt_int,
    fmt_pct,
    hist_to_df,
    kpi_row,
    load_run_csv,
    load_train_run,
    get_inference_bundle,
    model_dir,
    inspection_dir,
    t,
)


def artifacts() -> dict:
    """What each artefact in the run directory is for, in plain language."""
    return {
        "{p}_events_canonical.csv": (
            t("Canonical events", "Event đã canonize"),
            t(
                "Every raw event after URL canonization, id masking and taxonomy enrichment. "
                "One row per event, with the exact token and its semantic family/module.",
                "Toàn bộ event thô sau khi canonize URL, mask id và gắn taxonomy. Mỗi dòng là một "
                "event, kèm exact token và family/module ngữ nghĩa của nó.",
            ),
        ),
        "{p}_token_dictionary.csv": (
            t("Token dictionary", "Từ điển token"),
            t(
                "The governed vocabulary: each token with event/session frequency, business family, "
                "module, semantic confidence and first/last seen date.",
                "Bộ vocabulary có kiểm soát: mỗi token kèm tần suất theo event/session, business family, "
                "module, độ tin cậy ngữ nghĩa và ngày xuất hiện đầu/cuối.",
            ),
        ),
        "{p}_sequences.jsonl": (
            t("Journey sequences", "Chuỗi journey"),
            t(
                "One JSON line per journey holding the ordered token sequence fed to the vectorizer.",
                "Mỗi dòng JSON là một journey, chứa chuỗi token theo thứ tự đưa vào vectorizer.",
            ),
        ),
        "{p}_journeys.csv": (
            t("Journeys (modelling set)", "Journey (tập để model)"),
            t(
                "One row per journey ≥ 4 events, with shape features, boundary reason, assigned "
                "cluster and Markov log-probability. This is the training table.",
                "Mỗi dòng là một journey ≥ 4 event, kèm feature hình dạng, lý do cắt, cluster được gán "
                "và Markov log-probability. Đây là bảng dùng để train.",
            ),
        ),
        "{p}_journeys_all_including_short.csv": (
            t("Journeys incl. short", "Journey kể cả journey ngắn"),
            t(
                "Same as above but keeping journeys below the 4-event minimum, for auditing what was dropped.",
                "Giống bảng trên nhưng giữ cả journey dưới ngưỡng 4 event, để kiểm tra lại phần bị loại.",
            ),
        ),
        "{p}_journey_scorer.pkl": (
            t("Fitted scorer", "Scorer đã fit"),
            t(
                "Serialized vectorizer + SVD + cluster centroids + Markov model — the object used at inference time.",
                "Vectorizer + SVD + centroid cluster + Markov model đã serialize — chính object dùng khi inference.",
            ),
        ),
        "{p}_cluster_catalog.json": (
            t("Cluster catalog", "Catalog cluster"),
            t(
                "Per-cluster profile: size, sessions, devices, median length/span, entry/exit tokens and medoid path.",
                "Hồ sơ từng cluster: size, số session, số device, độ dài/thời lượng trung vị, token entry/exit và medoid path.",
            ),
        ),
        "{p}_cluster_ngrams.csv": (
            t("Cluster n-grams", "N-gram của cluster"),
            t(
                "Top discriminative n-grams per cluster with lift and cluster mass — the evidence used for naming.",
                "Các n-gram đặc trưng nhất của từng cluster kèm lift và cluster mass — bằng chứng để đặt tên.",
            ),
        ),
        "{p}_cluster_business_mapping.csv": (
            t("Business cluster mapping", "Mapping cluster nghiệp vụ"),
            t(
                "Cluster-to-function mapping with evidence share, dominant intents, and medoid support.",
                "Mapping cluster tới function kèm tỷ lệ bằng chứng, intent trội và medoid support.",
            ),
        ),
        "{p}_cluster_name_mapping.csv": (
            t("Cluster names", "Tên cluster"),
            t(
                "cluster_id → business name (EN + VI), business family and naming confidence.",
                "cluster_id → tên nghiệp vụ (EN + VI), business family và độ tin cậy khi đặt tên.",
            ),
        ),
        "{p}_scored_holdout.csv": (
            t("Scored holdout", "Holdout đã score"),
            t(
                "The chronological 20% holdout scored with the fitted model — the honest generalisation check.",
                "Phần holdout 20% theo thời gian được score bằng model đã fit — phép kiểm tra khả năng tổng quát hóa.",
            ),
        ),
        "{p}_run_config.json": (
            t("Run config + fit metrics", "Config run + metric fit"),
            t(
                "Full hyper-parameters plus feature/HDBSCAN quality metrics for this run.",
                "Toàn bộ hyper-parameter cùng metric chất lượng feature/HDBSCAN của run này.",
            ),
        ),
        "{p}_RUN_REPORT.md": (
            t("Run report", "Báo cáo run"),
            t("Human-readable narrative report of the whole run.", "Báo cáo dạng văn bản mô tả toàn bộ run."),
        ),
        "inference_manifest.json": (
            t("Inference manifest", "Manifest inference"),
            t("Partition-level counts and source/output lineage for the scored production data.", "Số liệu từng partition và lineage nguồn/đầu ra của dữ liệu production đã score."),
        ),
    }


def report_csvs() -> dict:
    return {
        "{p}_report_canonization.csv": t(
            "Canonization — vocabulary after normalising event_type@segment_name.",
            "Canonization — vocabulary sau khi chuẩn hóa event_type@segment_name.",
        ),
        "{p}_report_vocabulary.csv": t(
            "Vocabulary per feature channel (exact / intent / coarse / family / operation).",
            "Vocabulary theo từng feature channel (exact / intent / coarse / family / operation).",
        ),
        "{p}_report_rare_folding.csv": t(
            "How many rare tokens were backed off to the coarser level.",
            "Bao nhiêu token hiếm đã được backoff về mức thô hơn.",
        ),
        "{p}_report_postprocess.csv": t(
            "Event cleanup: consecutive duplicates and navigation loops removed.",
            "Làm sạch event: loại bỏ trùng liên tiếp và navigation loop.",
        ),
        "{p}_report_segmentation.csv": t(
            "Session → journey segmentation, including why each cut was made.",
            "Cắt session → journey, kèm lý do của từng điểm cắt.",
        ),
        "{p}_report_idle_gap_sweep.csv": t(
            "Sensitivity of journey shape to the idle-gap threshold.",
            "Độ nhạy của hình dạng journey theo ngưỡng idle gap.",
        ),
        "{p}_report_semantics.csv": t(
            "Taxonomy coverage: family/module/object/operation resolution rates.",
            "Độ phủ taxonomy: tỷ lệ giải được family/module/object/operation.",
        ),
        "{p}_report_semantic_unknowns.csv": t(
            "Events the taxonomy could not resolve — the backlog to curate.",
            "Các event taxonomy chưa giải được — backlog cần bổ sung.",
        ),
        "{p}_report_kmeans_sweep.csv": t(
            "K-means reference sweep used to sanity-check the density model.",
            "Sweep k-means dùng làm mốc tham chiếu để đối chiếu với model mật độ.",
        ),
        "{p}_report_cluster_catalog.csv": t(
            "Flat CSV mirror of the cluster catalog.",
            "Bản CSV phẳng của cluster catalog.",
        ),
    }


def _kv_report(df: pd.DataFrame) -> dict:
    if df.empty or not {"metric", "value"} <= set(df.columns):
        return {}
    return dict(zip(df["metric"], df["value"]))


def render() -> None:
    st.title(t("2 · What the training run produced", "2 · Training run đã tạo ra những gì"))
    bundle_key = st.session_state.get("inference_bundle")
    active_train_root = bundle_training_root(bundle_key)
    active_bundle = get_inference_bundle(bundle_key)
    active_inference_root = active_bundle.root if active_bundle else RUN_DIR
    st.caption(
        t("Active fitted run: ", "Run fit đang dùng: ")
        + f"`{active_train_root.relative_to(active_train_root.parents[2])}`"
        + t(" · names/inference bundle: ", " · bundle tên/inference: ")
        + f"`{active_inference_root.relative_to(active_inference_root.parents[2])}`"
    )

    run = load_train_run(bundle_key)
    available_platforms = [p for p in ["android", "ios"] if run.get("platforms", {}).get(p, {}).get("run_config")]
    available_platforms = available_platforms or ["android", "ios"]
    platform = st.radio(
        t("Platform model", "Model theo platform"), available_platforms, horizontal=True, format_func=str.title
    )
    def run_csv(name: str) -> pd.DataFrame:
        return load_run_csv(name, bundle_key=bundle_key, platform=platform)

    pdata = run["platforms"].get(platform, {})
    cfg = pdata.get("run_config", {})
    fit = cfg.get("feature_info", {})
    hdb = cfg.get("hdbscan", {})

    seg = _kv_report(run_csv(f"{platform}_report_segmentation.csv"))
    post = _kv_report(run_csv(f"{platform}_report_postprocess.csv"))
    sem = _kv_report(run_csv(f"{platform}_report_semantics.csv"))
    canon = run_csv(f"{platform}_report_canonization.csv")
    vocab = run_csv(f"{platform}_report_vocabulary.csv")
    rare = run_csv(f"{platform}_report_rare_folding.csv")
    split = run_csv("split_report.csv")
    prep = run_csv("preprocessing_report.csv")

    kpi_row(
        [
            (t("Events in", "Event đầu vào"), fmt_int(seg.get("events")), t("Events entering segmentation", "Số event đi vào bước cắt journey")),
            (t("Sessions", "Session"), fmt_int(seg.get("sessions")), None),
            (t("Journeys", "Journey"), fmt_int(seg.get("journeys")), t("After segmentation, before the ≥4-event filter", "Sau khi cắt, trước bộ lọc ≥ 4 event")),
            (t("Journeys modelled", "Journey đưa vào model"), fmt_int(fit.get("n_journeys")), t("Journeys that reached the clustering step", "Số journey đi tới bước clustering")),
            (t("Clusters found", "Số cluster tìm được"), fmt_int(hdb.get("n_clusters")), t("HDBSCAN clusters, excluding noise", "Cluster của HDBSCAN, không tính noise")),
            (t("Noise share", "Tỷ lệ noise"), fmt_pct(hdb.get("noise_share")), t("Journeys HDBSCAN left unassigned during training", "Journey mà HDBSCAN để chưa gán khi train")),
        ]
    )

    st.divider()

    # -------------------------------------------------------------- pipeline funnel
    st.subheader(t("Pipeline funnel — raw rows to modelled journeys", "Phễu pipeline — từ dòng thô tới journey được model"))
    rows_in = prep[prep["source_file"].str.startswith(platform, na=False)]["rows_output"].sum() if not prep.empty else None
    stages = [
        (t("Rows read from raw CSVs", "Dòng đọc từ CSV thô"), rows_in),
        (t("Events after cleanup", "Event sau khi làm sạch"), post.get("events_after_cleanup")),
        (t("Journeys segmented", "Journey sau khi cắt"), seg.get("journeys")),
        (t("Journeys ≥ 4 events", "Journey ≥ 4 event"), (seg.get("journeys") or 0) - (seg.get("journeys_below_min_len_4") or 0)),
        (t("Journeys clustered", "Journey được cluster"), fit.get("n_journeys")),
    ]
    stages = [(n, v) for n, v in stages if v]
    fig = go.Figure(
        go.Funnel(
            y=[s[0] for s in stages],
            x=[float(s[1]) for s in stages],
            marker_color=PALETTE[: len(stages)],
            textinfo="value+percent initial",
        )
    )
    fig.update_layout(height=360, margin=dict(t=10, b=10, l=10))
    st.plotly_chart(fig, width="stretch")
    if post:
        st.caption(
            t(
                f"Cleanup removed {fmt_int(post.get('consecutive_dupes_removed'))} consecutive duplicate events and "
                f"{fmt_int(post.get('loop_events_removed'))} loop events "
                f"(compression ratio {post.get('compression_ratio')}). "
                f"{fmt_int(post.get('journeys_with_loops'))} journeys contained a navigation loop.",
                f"Bước làm sạch đã loại {fmt_int(post.get('consecutive_dupes_removed'))} event trùng liên tiếp và "
                f"{fmt_int(post.get('loop_events_removed'))} event thuộc vòng lặp "
                f"(compression ratio {post.get('compression_ratio')}). "
                f"{fmt_int(post.get('journeys_with_loops'))} journey có chứa navigation loop.",
            )
        )

    st.divider()

    # -------------------------------------------------------------- artefact browser
    st.subheader(t("Artefacts written by this run", "Các file run này đã ghi ra"))
    A = t("artefact", "file kết quả")
    F = t("file", "tên file")
    S = t("size (MB)", "dung lượng (MB)")
    W = t("what it holds", "chứa gì")
    inv = []
    bundle = active_bundle
    artifact_roots = [active_train_root / platform]
    if bundle:
        artifact_roots.extend([bundle_model_dir(bundle, platform), bundle_inspection_dir(bundle, platform)])
    artifact_roots.extend([model_dir(platform), inspection_dir(platform)])
    for pattern, (title, desc) in artifacts().items():
        name = pattern.format(p=platform)
        path = next((root / name for root in artifact_roots if (root / name).exists()), None)
        if path is None:
            continue
        inv.append({A: title, F: name, S: round(path.stat().st_size / 1e6, 2), W: desc})
    for pattern, desc in report_csvs().items():
        name = pattern.format(p=platform)
        path = next((root / name for root in artifact_roots if (root / name).exists()), None)
        if path is None:
            continue
        inv.append(
            {
                A: t("Report — ", "Report — ") + name.replace(f"{platform}_report_", "").replace(".csv", ""),
                F: name,
                S: round(path.stat().st_size / 1e6, 3),
                W: desc,
            }
        )
    naming_candidates = []
    if bundle:
        naming_candidates.append(bundle_platform_root(bundle, platform) / "model_version=latest" / f"{platform}_cluster_mapping.csv")
    naming_candidates.extend([RUN_DIR / "Cluster_naming.csv", RUN_DIR / "cluster_mapping.csv"])
    naming_path = next((path for path in naming_candidates if path.exists()), None)
    if naming_path is not None:
        inv.append(
            {
                A: t("Authoritative cluster naming", "Tên cluster authoritative"),
                F: "Cluster_naming.csv",
                S: round(naming_path.stat().st_size / 1e6, 2),
                W: t("Shared Android/iOS business names and naming confidence.", "Tên nghiệp vụ dùng chung Android/iOS và độ tin cậy khi đặt tên."),
            }
        )
    st.dataframe(pd.DataFrame(inv), hide_index=True, width="stretch", height=430)

    with st.expander(t("Open a report CSV", "Mở một report CSV")):
        options = [p.format(p=platform) for p in report_csvs()] + [
            "preprocessing_report.csv",
            "split_report.csv",
            f"{platform}_cluster_name_mapping.csv",
        ]
        options = [o for o in options if any((root / o).exists() for root in artifact_roots)]
        pick = st.selectbox(F, options)
        st.dataframe(run_csv(pick).head(300), hide_index=True, width="stretch")

    st.divider()

    # -------------------------------------------------------------- tokenization
    st.header(t("Tokenization and taxonomy quality", "Chất lượng tokenization và taxonomy"))
    a, b = st.columns([3, 2])
    if not vocab.empty:
        fig = px.bar(vocab, x="channel", y="vocabulary", color="channel", color_discrete_sequence=PALETTE, text="vocabulary")
        fig.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False, xaxis_title=None)
        a.plotly_chart(fig, width="stretch")
        a.caption(
            t(
                "Vocabulary size per feature channel. The model does not use one representation — "
                "it blends an exact channel with coarse / intent / operation channels "
                f"(weights {cfg.get('config', {}).get('features', {}).get('channel_weights', {})}).",
                "Kích thước vocabulary theo từng feature channel. Model không dùng một biểu diễn duy nhất — "
                "nó trộn channel exact với các channel coarse / intent / operation "
                f"(trọng số {cfg.get('config', {}).get('features', {}).get('channel_weights', {})}).",
            )
        )
        b.dataframe(vocab[["channel", "vocabulary", "singleton_share", "top100_coverage"]], hide_index=True, width="stretch")
    if not canon.empty:
        b.caption(
            t(
                f"Canonization reduced {fmt_int(canon['rows'].iloc[0])} events to "
                f"{fmt_int(canon['vocabulary'].iloc[0])} distinct tokens "
                f"({fmt_int(canon['singletons'].iloc[0])} of them singletons).",
                f"Canonization rút {fmt_int(canon['rows'].iloc[0])} event xuống còn "
                f"{fmt_int(canon['vocabulary'].iloc[0])} token khác nhau "
                f"(trong đó {fmt_int(canon['singletons'].iloc[0])} là singleton).",
            )
        )
    if not rare.empty:
        r = rare.iloc[0]
        b.caption(
            t(
                f"Rare folding: vocabulary {fmt_int(r['vocabulary_before'])} → {fmt_int(r['vocabulary_after'])}; "
                f"{fmt_pct(r['backoff_share'], 2)} of token occurrences were backed off to the coarser level.",
                f"Gộp token hiếm: vocabulary {fmt_int(r['vocabulary_before'])} → {fmt_int(r['vocabulary_after'])}; "
                f"{fmt_pct(r['backoff_share'], 2)} số lần xuất hiện token đã được backoff về mức thô hơn.",
            )
        )

    if sem:
        st.subheader(t("Semantic coverage", "Độ phủ ngữ nghĩa"))
        mcol = t("metric", "chỉ số")
        vcol = t("value", "giá trị")
        cov = pd.DataFrame(
            [
                {mcol: t("family resolved", "giải được family"), vcol: sem.get("family_coverage", 0)},
                {mcol: t("module specific", "module cụ thể"), vcol: sem.get("module_specific_rate", 0)},
                {mcol: t("operation resolved", "giải được operation"), vcol: 1 - sem.get("operation_unknown_rate", 0)},
                {mcol: t("object resolved", "giải được object"), vcol: 1 - sem.get("object_unknown_rate", 0)},
                {mcol: t("high confidence", "độ tin cậy cao"), vcol: 1 - sem.get("low_confidence_rate", 0)},
            ]
        )
        c1, c2 = st.columns([3, 2])
        fig = px.bar(cov, x=vcol, y=mcol, orientation="h", color_discrete_sequence=[PALETTE[2]])
        fig.update_layout(height=280, margin=dict(t=10, b=10), xaxis_tickformat=".0%", xaxis_range=[0, 1], yaxis_title=None)
        c1.plotly_chart(fig, width="stretch")
        c2.markdown(
            t(
                f"""
The taxonomy resolves a **business family for {sem.get('family_coverage', 0):.1%}** of events
across **{int(sem.get('distinct_families', 0))} families**, but only
**{sem.get('module_specific_rate', 0):.1%}** of events land on a specific module and
**{1 - sem.get('object_unknown_rate', 0):.1%}** on a concrete object.

That gap is the curation backlog: mean semantic confidence is
**{sem.get('mean_confidence', 0):.2f}** and **{sem.get('low_confidence_rate', 0):.1%}**
of events are low-confidence. The unresolved events are listed in
`{platform}_report_semantic_unknowns.csv`.
                """,
                f"""
Taxonomy giải được **business family cho {sem.get('family_coverage', 0):.1%}** số event
trên **{int(sem.get('distinct_families', 0))} family**, nhưng chỉ
**{sem.get('module_specific_rate', 0):.1%}** event rơi vào một module cụ thể và
**{1 - sem.get('object_unknown_rate', 0):.1%}** xác định được object cụ thể.

Khoảng trống đó chính là backlog cần bổ sung: độ tin cậy ngữ nghĩa trung bình là
**{sem.get('mean_confidence', 0):.2f}** và **{sem.get('low_confidence_rate', 0):.1%}**
số event có độ tin cậy thấp. Danh sách event chưa giải được nằm ở
`{platform}_report_semantic_unknowns.csv`.
                """,
            )
        )
        unknowns = run_csv(f"{platform}_report_semantic_unknowns.csv")
        if not unknowns.empty:
            with st.expander(t("Top unresolved events (taxonomy backlog)", "Các event chưa giải được (backlog taxonomy)")):
                st.dataframe(unknowns.head(30), hide_index=True, width="stretch")

    st.divider()

    # -------------------------------------------------------------- segmentation
    st.header(t("Segmentation — how sessions became journeys", "Segmentation — session trở thành journey như thế nào"))
    a, b = st.columns([2, 3])
    cuts = {k.replace("cut_reason::", ""): v for k, v in seg.items() if k.startswith("cut_reason::")}
    if cuts:
        rcol = t("cut reason", "lý do cắt")
        jcol = t("journeys", "journey")
        cd = pd.DataFrame({rcol: list(cuts.keys()), jcol: list(cuts.values())})
        fig = px.pie(cd, names=rcol, values=jcol, hole=0.5, color_discrete_sequence=PALETTE)
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        a.plotly_chart(fig, width="stretch")
        a.caption(t("Why each journey boundary was created.", "Lý do tạo ra mỗi điểm cắt journey."))
    sweep = run_csv(f"{platform}_report_idle_gap_sweep.csv")
    if not sweep.empty:
        fig = go.Figure()
        fig.add_bar(x=sweep["idle_gap_seconds"], y=sweep["journeys"], name=t("journeys", "journey"), marker_color=PALETTE[0])
        fig.add_scatter(
            x=sweep["idle_gap_seconds"], y=sweep["share_len_lt_3"],
            name=t("share of journeys < 3 events", "tỷ lệ journey < 3 event"),
            yaxis="y2", mode="lines+markers", line_color=PALETTE[3],
        )
        fig.update_layout(
            height=340, margin=dict(t=10, b=10),
            xaxis_title=t("idle gap threshold (s)", "ngưỡng idle gap (giây)"), xaxis_type="log",
            yaxis_title=t("journeys", "journey"),
            yaxis2=dict(overlaying="y", side="right", tickformat=".0%", title=t("fragment share", "tỷ lệ mảnh vụn")),
            legend=dict(orientation="h", y=1.15),
        )
        b.plotly_chart(fig, width="stretch")
        b.caption(
            t(
                "Idle-gap sweep. Below ~90 s the cut starts shredding real tasks into fragments; "
                "above it the curve flattens and journeys just absorb unrelated activity. "
                "**90 s** is the chosen operating point.",
                "Sweep idle gap. Dưới ~90 giây, việc cắt bắt đầu băm nhỏ task thật thành mảnh vụn; "
                "trên ngưỡng đó đường cong đi ngang và journey chỉ hút thêm hoạt động không liên quan. "
                "**90 giây** là điểm vận hành được chọn.",
            )
        )

    j = pdata.get("journeys", {})
    if j:
        st.subheader(t("Shape of the modelled journeys", "Hình dạng của các journey được model"))
        kpi_row(
            [
                (t("Journeys", "Journey"), fmt_int(j["n_journeys"]), None),
                (t("Sessions covered", "Session được phủ"), fmt_int(j["n_sessions"]), None),
                (t("Devices", "Device"), fmt_int(j["n_devices"]), None),
                (t("Median length", "Độ dài trung vị"), t(f"{j['quantiles']['n_events_final']['0.5']:.0f} events", f"{j['quantiles']['n_events_final']['0.5']:.0f} event"), None),
                (t("Median duration", "Thời lượng trung vị"), f"{j['quantiles']['span_seconds']['0.5']:.0f} s", None),
                (
                    t("Median action ratio", "Action ratio trung vị"),
                    f"{j['quantiles']['action_ratio']['0.5']:.2f}",
                    t("Share of steps that are taps rather than views", "Tỷ lệ bước là thao tác tap thay vì view"),
                ),
            ]
        )
        jl = t("journeys", "journey")
        c1, c2 = st.columns(2)
        fig = px.bar(hist_to_df(j["length_hist"], jl), x="bin", y=jl, color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title=t("events per journey", "event mỗi journey"), yaxis_title=None)
        c1.plotly_chart(fig, width="stretch")
        fig = px.bar(hist_to_df(j["span_hist"], jl), x="bin", y=jl, color_discrete_sequence=[PALETTE[1]])
        fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title=t("journey duration (s, clipped at 600)", "thời lượng journey (giây, cắt ở 600)"), yaxis_title=None)
        c2.plotly_chart(fig, width="stretch")

        c3, c4 = st.columns(2)
        fig = px.bar(hist_to_df(j["action_ratio_hist"], jl), x="bin", y=jl, color_discrete_sequence=[PALETTE[2]])
        fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="action ratio", yaxis_title=None)
        c3.plotly_chart(fig, width="stretch")
        fig = px.bar(hist_to_df(j["markov_hist"], jl), x="bin", y=jl, color_discrete_sequence=[PALETTE[3]])
        fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title=t("Markov log-probability per step", "Markov log-probability mỗi bước"), yaxis_title=None)
        c4.plotly_chart(fig, width="stretch")
        c4.caption(
            t(
                "How predictable each journey is under the fitted transition model — the left tail is where friction lives.",
                "Mức độ dễ đoán của mỗi journey theo transition model đã fit — phần đuôi bên trái là nơi có friction.",
            )
        )

        e1, e2 = st.columns(2)
        ed = pd.DataFrame(j["top_entry_tokens"], columns=["token", jl]).head(12)
        fig = px.bar(ed.sort_values(jl), x=jl, y="token", orientation="h", color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(height=380, margin=dict(t=10, b=10), yaxis_title=None, title=t("Most common entry tokens", "Token mở đầu phổ biến nhất"))
        e1.plotly_chart(fig, width="stretch")
        xd = pd.DataFrame(j["top_exit_tokens"], columns=["token", jl]).head(12)
        fig = px.bar(xd.sort_values(jl), x=jl, y="token", orientation="h", color_discrete_sequence=[PALETTE[1]])
        fig.update_layout(height=380, margin=dict(t=10, b=10), yaxis_title=None, title=t("Most common exit tokens", "Token kết thúc phổ biến nhất"))
        e2.plotly_chart(fig, width="stretch")

    st.divider()

    # -------------------------------------------------------------- model quality
    st.header(t("Model fit and generalisation", "Chất lượng fit và khả năng tổng quát hóa"))
    icol = t("item", "hạng mục")
    vcol = t("value", "giá trị")
    a, b, c = st.columns([2, 2, 3])
    a.markdown(t("**Feature space**", "**Không gian feature**"))
    a.dataframe(
        pd.DataFrame(
            [
                {icol: t("journeys", "journey"), vcol: fmt_int(fit.get("n_journeys"))},
                {icol: t("TF-IDF vocabulary", "vocabulary TF-IDF"), vcol: fmt_int(fit.get("tfidf_vocabulary"))},
                {icol: t("SVD components / channel", "số SVD component / channel"), vcol: str(fit.get("svd_components"))},
                {icol: t("explained variance (primary)", "explained variance (channel chính)"), vcol: str(fit.get("svd_explained_variance"))},
                {icol: t("numeric features", "feature dạng số"), vcol: str(fit.get("numeric_features"))},
                {icol: t("final dimension", "số chiều cuối cùng"), vcol: str(fit.get("final_dimension"))},
            ]
        ),
        hide_index=True, width="stretch",
    )
    b.markdown(t("**HDBSCAN quality**", "**Chất lượng HDBSCAN**"))
    b.dataframe(
        pd.DataFrame(
            [
                {icol: t("clusters", "số cluster"), vcol: str(hdb.get("n_clusters"))},
                {icol: t("noise share", "tỷ lệ noise"), vcol: fmt_pct(hdb.get("noise_share"))},
                {icol: "silhouette", vcol: str(hdb.get("silhouette"))},
                {icol: t("Davies–Bouldin (lower=better)", "Davies–Bouldin (càng thấp càng tốt)"), vcol: str(hdb.get("davies_bouldin"))},
                {icol: t("Calinski–Harabasz (higher=better)", "Calinski–Harabasz (càng cao càng tốt)"), vcol: str(hdb.get("calinski_harabasz"))},
            ]
        ),
        hide_index=True, width="stretch",
    )
    kmeans = run_csv(f"{platform}_report_kmeans_sweep.csv")
    if not kmeans.empty:
        fig = go.Figure()
        fig.add_scatter(x=kmeans["k"], y=kmeans["silhouette"], name=t("k-means silhouette", "silhouette k-means"), mode="lines+markers", line_color=PALETTE[0])
        if hdb.get("silhouette"):
            fig.add_hline(
                y=hdb["silhouette"], line_dash="dash", line_color=PALETTE[3],
                annotation_text=f"HDBSCAN {hdb['silhouette']}", annotation_position="top left",
            )
        fig.update_layout(height=300, margin=dict(t=30, b=10), xaxis_title="k", yaxis_title="silhouette")
        c.plotly_chart(fig, width="stretch")
        c.caption(
            t(
                "A k-means sweep is run only as a reference. Forcing every journey into a sphere never "
                "beats the density model, which is allowed to leave genuinely mixed journeys unassigned.",
                "Sweep k-means chỉ chạy để làm mốc tham chiếu. Ép mọi journey vào một hình cầu không bao giờ "
                "thắng được model mật độ — vốn được phép để trống những journey thực sự pha trộn.",
            )
        )

    if not split.empty:
        st.subheader(t("Train / holdout split", "Chia train / holdout"))
        st.dataframe(split, hide_index=True, width="stretch")
        st.caption(
            t(
                "The split is **chronological and session-complete**: the last ~20% of sessions by time are "
                "held out and never contribute a single event to training (`session_overlap = 0`). "
                "This is what makes the holdout numbers below a real generalisation test rather than a re-fit.",
                "Cách chia là **theo thời gian và trọn vẹn từng session**: ~20% session cuối theo thời gian được "
                "giữ lại và không đóng góp bất kỳ event nào cho quá trình train (`session_overlap = 0`). "
                "Nhờ vậy các con số holdout bên dưới là phép kiểm tra tổng quát hóa thật, không phải fit lại.",
            )
        )

    hold = pdata.get("holdout", {})
    if hold:
        assign = hold.get("assignment_type", {})
        total = sum(assign.values()) or 1
        assigned = total - assign.get("unassigned_novel", 0)
        kpi_row(
            [
                (t("Holdout journeys", "Journey holdout"), fmt_int(hold.get("n_journeys")), None),
                (t("Holdout sessions", "Session holdout"), fmt_int(hold.get("n_sessions")), None),
                (
                    t("Assigned to a known journey type", "Gán được vào journey type đã biết"),
                    fmt_pct(assigned / total),
                    t("Scored within the cluster's distance limit", "Nằm trong distance limit của cluster"),
                ),
                (
                    t("Novel / unassigned", "Mới / chưa gán được"),
                    fmt_pct(assign.get("unassigned_novel", 0) / total),
                    t("Beyond every centroid's distance limit", "Vượt distance limit của mọi centroid"),
                ),
            ]
        )
        st.markdown(
            t(
                "Note the contrast with the **training** noise share above: HDBSCAN refuses to label "
                "ambiguous journeys while *fitting*, but at *scoring* time a journey is assigned to its "
                "nearest centroid whenever it falls inside that cluster's learned distance limit. "
                "That is why coverage jumps from roughly half to the vast majority.",
                "Hãy so với tỷ lệ noise khi **train** ở trên: HDBSCAN từ chối gán nhãn cho các journey mơ hồ "
                "trong lúc *fit*, nhưng khi *scoring*, một journey sẽ được gán vào centroid gần nhất miễn là nó "
                "nằm trong distance limit đã học của cluster đó. Đó là lý do độ phủ nhảy từ khoảng một nửa lên "
                "gần như toàn bộ.",
            )
        )
        if hold.get("distance_q"):
            dq = hold["distance_q"]
            st.dataframe(
                pd.DataFrame(
                    {
                        t("quantile", "quantile"): list(dq.keys()),
                        t("distance to centroid", "khoảng cách tới centroid"): [float(v) for v in dq.values()],
                    }
                ),
                hide_index=True, width="stretch",
            )
            st.caption(
                t(
                    "Distance from each holdout journey to its assigned centroid. Journeys beyond the "
                    "cluster's learned limit are the novel ones.",
                    "Khoảng cách từ mỗi journey holdout tới centroid được gán. Journey vượt quá giới hạn đã học "
                    "của cluster chính là các journey mới.",
                )
            )
