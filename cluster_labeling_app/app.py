"""Standalone Streamlit app for taxonomy and cluster-name business review."""

from __future__ import annotations

import math
import os
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

import database as sqlite_database

from core import (
    PLATFORMS,
    assignment_status,
    catalog_row,
    cluster_ngrams,
    export_named_clusters_csv,
    export_taxonomy_csv,
    filter_named_clusters,
    load_platform_evidence,
    normalize_taxonomy,
    taxonomy_choices,
    validate_taxonomy,
)
APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
DB_PATH = Path(os.environ.get("HIFPT_SQLITE_PATH", APP_DIR / "db.sqlite")).resolve()
PLATFORM_LABELS = {"android": "Android", "ios": "iOS"}
CONFIDENCE_ORDER = ["low", "medium", "high", "not_applicable"]


st.set_page_config(
    page_title="HiFPT Taxonomy & Cluster Naming",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def setting(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value.strip()
    try:
        return str(st.secrets.get(name, "")).strip()
    except FileNotFoundError:
        return ""


SUPABASE_URL = setting("SUPABASE_URL")
SUPABASE_KEY = setting("SUPABASE_SECRET_KEY") or setting("SUPABASE_KEY")
if bool(SUPABASE_URL) != bool(SUPABASE_KEY):
    st.error("Cần cấu hình đồng thời SUPABASE_URL và SUPABASE_SECRET_KEY/SUPABASE_KEY.")
    st.stop()
if SUPABASE_URL:
    import supabase_database as database_api
    from supabase_database import SupabaseTarget

    DATABASE_TARGET = SupabaseTarget(SUPABASE_URL, SUPABASE_KEY)
    DATABASE_LABEL = "Supabase PostgreSQL"
    DATABASE_LOCATION = SUPABASE_URL
else:
    database_api = sqlite_database
    DATABASE_TARGET = DB_PATH
    DATABASE_LABEL = "SQLite local"
    DATABASE_LOCATION = str(DB_PATH)

st.markdown(
    """
    <style>
    :root { --brand:#0B63CE; --ink:#172033; --muted:#667085; --line:#E5EAF2; --soft:#F5F8FC; }
    .block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { color:var(--ink); letter-spacing:-.02em; }
    h1 { font-size:2.2rem; margin-bottom:.15rem; }
    div[data-testid="stMetric"] { border:1px solid var(--line); border-radius:14px; padding:14px 16px; background:white; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line); border-radius:16px; background:white; }
    .review-lead { color:var(--muted); font-size:1.03rem; margin:0 0 1.25rem 0; }
    .step { border-left:3px solid var(--brand); padding:.2rem 0 .2rem .9rem; margin:.75rem 0; }
    .plain-note { background:var(--soft); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    [data-testid="stTabs"] button { font-weight:650; }
    .ngram-table { border:1px solid var(--line); border-radius:12px; overflow:hidden; margin-bottom:1rem; }
    .ngram-row { display:grid; grid-template-columns:44px minmax(0, 1fr) 72px 72px; gap:10px; align-items:start; padding:10px 12px; border-top:1px solid var(--line); }
    .ngram-row:first-child { border-top:0; }
    .ngram-head { color:var(--muted); background:var(--soft); font-size:.78rem; font-weight:700; text-transform:uppercase; }
    .ngram-flow { display:flex; flex-wrap:wrap; align-items:center; gap:5px; min-width:0; }
    .ngram-token { background:#EEF4FF; border:1px solid #D6E4FF; border-radius:6px; color:#1849A9; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.79rem; line-height:1.35; padding:3px 6px; max-width:100%; overflow-wrap:anywhere; word-break:break-word; }
    .ngram-separator { color:#98A2B3; font-weight:800; }
    .ngram-number { color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }
    @media (max-width: 700px) {
        .ngram-row { grid-template-columns:minmax(0, 1fr); }
        .ngram-number { display:none; }
        .ngram-head > div:not(:nth-child(2)) { display:none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Đang đọc catalog và n-grams…")
def load_evidence() -> dict[str, object]:
    return {
        platform: load_platform_evidence(ASSET_DIR, platform)
        for platform in PLATFORMS
    }


try:
    EVIDENCE = load_evidence()
    DB_STATS = database_api.initialize_database(DATABASE_TARGET, ASSET_DIR)
except Exception as error:  # pragma: no cover - deployment diagnostics
    st.error(f"Không thể khởi tạo dữ liệu ứng dụng: {error}")
    st.stop()


def ordered_confidences(frame: pd.DataFrame) -> list[str]:
    present = [value for value in frame["naming_confidence"].dropna().unique() if value]
    return [value for value in CONFIDENCE_ORDER if value in present] + sorted(
        set(present) - set(CONFIDENCE_ORDER)
    )


def current_or_choices(current: str, values: list[str], include_blank: bool = False) -> list[str]:
    options = ([""] if include_blank else []) + list(values)
    if current and current not in options:
        options.insert(1 if include_blank else 0, current)
    return list(dict.fromkeys(options))


def safe_selectbox(
    label: str,
    options: list[str],
    current: str,
    *,
    key: str,
    invalid: set[str] | None = None,
    disabled: bool = False,
    on_change=None,
    args: tuple[object, ...] = (),
) -> str:
    invalid = invalid or set()
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    selectbox_options: dict[str, int] = {}
    if key not in st.session_state:
        selectbox_options["index"] = options.index(current) if current in options else 0
    return st.selectbox(
        label,
        options,
        key=key,
        disabled=disabled,
        on_change=on_change,
        args=args,
        format_func=lambda value: (
            "— Chưa chọn —"
            if value == ""
            else f"{value} (ngoài taxonomy active)"
            if value in invalid
            else value
        ),
        **selectbox_options,
    )


def reset_mapping_children(platform: str, cluster_id: int, level: str) -> None:
    """Clear dependent draft selections when a parent dropdown changes."""

    prefix = f"map::{platform}::{cluster_id}"
    if level == "family":
        st.session_state[f"{prefix}::submodule"] = ""
    st.session_state[f"{prefix}::detail"] = ""


def render_ngram_table(ngrams: pd.DataFrame) -> None:
    """Render tokenized n-grams with visible separators and natural line wrapping."""

    rows = [
        '<div class="ngram-row ngram-head"><div>Hạng</div><div>Chuỗi hành vi</div>'
        '<div>Lift</div><div>Mass</div></div>'
    ]
    for item in ngrams.itertuples(index=False):
        tokens = str(item.ngram).split()
        flow: list[str] = []
        for index, token in enumerate(tokens):
            if index:
                flow.append('<span class="ngram-separator" aria-hidden="true">→</span>')
            flow.append(f'<span class="ngram-token">{escape(token)}</span>')
        lift = "—" if pd.isna(item.lift) else f"{float(item.lift):.3f}"
        mass = "—" if pd.isna(item.cluster_mass) else f"{float(item.cluster_mass):.3f}"
        rows.append(
            '<div class="ngram-row">'
            f'<div class="ngram-number">{int(item.rank)}</div>'
            f'<div class="ngram-flow">{"".join(flow)}</div>'
            f'<div class="ngram-number">{lift}</div>'
            f'<div class="ngram-number">{mass}</div>'
            "</div>"
        )
    st.markdown(f'<div class="ngram-table">{"".join(rows)}</div>', unsafe_allow_html=True)


def render_cluster_card(
    platform: str,
    row: pd.Series,
    taxonomy: pd.DataFrame,
) -> None:
    cluster_id = int(row["cluster_id"])
    raw = catalog_row(EVIDENCE[platform].catalog, cluster_id)
    choices = taxonomy_choices(taxonomy)
    family = str(row["business_family"] or "")
    submodule = str(row["business_submodule"] or "")
    detail = str(row["business_detail"] or "")
    valid, validation_message = assignment_status(taxonomy, family, submodule, detail)

    with st.container(border=True):
        title_col, state_col = st.columns([5, 1.3], vertical_alignment="center")
        title_col.subheader(f"Cluster {cluster_id} · {row['cluster_name']}")
        state_col.markdown("**Cần review**" if bool(row["needs_review"]) else "Đã review")
        metrics = st.columns(6)
        metrics[0].metric("Score rows", f"{int(row['size']):,}")
        metrics[1].metric("Tỷ trọng score", f"{float(row['share']):.2%}")
        metrics[2].metric("Độ dài trung vị", f"{float(raw.get('median_length', 0)):.1f}")
        metrics[3].metric("Thời gian trung vị", f"{float(raw.get('median_span_s', 0)):.1f}s")
        metrics[4].metric("Back rate", f"{float(raw.get('mean_back_rate', 0)):.1%}")
        metrics[5].metric("Loop journey", f"{float(raw.get('loop_journey_share', 0)):.1%}")

        if not valid:
            st.warning(validation_message + " Nhãn cũ vẫn được giữ để người review xử lý.")

        st.markdown("##### Top n-grams")
        render_ngram_table(cluster_ngrams(EVIDENCE[platform].ngrams, cluster_id))

        with st.expander("Xem bằng chứng taxonomy và medoid path"):
            evidence_cols = st.columns(4)
            evidence_cols[0].metric("Taxonomy ID", row["taxonomy_id"] or "—")
            evidence_cols[1].metric("Nguồn đặt tên", row["naming_source"])
            evidence_cols[2].metric(
                "Score share",
                "—" if pd.isna(row["score_share"]) else f"{float(row['score_share']):.1%}",
            )
            evidence_cols[3].metric(
                "Mass coverage",
                "—"
                if pd.isna(row["evidence_mass_coverage"])
                else f"{float(row['evidence_mass_coverage']):.1%}",
            )
            st.markdown(f"**Supporting evidence:** {row['supporting_evidence'] or 'Không có'}")
            st.code(str(raw.get("medoid_path", "")), language=None)

        st.markdown(f"##### Nhãn nghiệp vụ lưu trong {DATABASE_LABEL}")
        family_key = f"map::{platform}::{cluster_id}::family"
        submodule_key = f"map::{platform}::{cluster_id}::submodule"
        detail_key = f"map::{platform}::{cluster_id}::detail"
        confidence_key = f"map::{platform}::{cluster_id}::confidence"
        review_key = f"map::{platform}::{cluster_id}::review"

        if family_key not in st.session_state:
            st.session_state[family_key] = family
        if submodule_key not in st.session_state:
            st.session_state[submodule_key] = submodule
        if detail_key not in st.session_state:
            st.session_state[detail_key] = detail

        first, second, third = st.columns(3)
        draft_family = str(st.session_state[family_key] or "")
        family_options = current_or_choices(draft_family, list(choices["families"]))
        if draft_family == "Chưa phân loại" and draft_family not in family_options:
            family_options.insert(0, draft_family)
        invalid_family = (
            {draft_family}
            if draft_family
            and draft_family != "Chưa phân loại"
            and draft_family not in choices["families"]
            else set()
        )
        with first:
            selected_family = safe_selectbox(
                "Business Family",
                family_options,
                draft_family,
                key=family_key,
                invalid=invalid_family,
                on_change=reset_mapping_children,
                args=(platform, cluster_id, "family"),
            )
        family = selected_family

        submodule = str(st.session_state[submodule_key] or "")
        available_submodules = list(choices["submodules"].get(family, []))
        submodule_options = current_or_choices(submodule, available_submodules, include_blank=True)
        invalid_submodule = (
            {submodule}
            if submodule and family != "Chưa phân loại" and (family, submodule) not in choices["pairs"]
            else set()
        )
        with second:
            selected_submodule = safe_selectbox(
                "Business Submodule",
                submodule_options,
                submodule,
                key=submodule_key,
                invalid=invalid_submodule,
                disabled=not family,
                on_change=reset_mapping_children,
                args=(platform, cluster_id, "submodule"),
            )
        submodule = selected_submodule

        detail = str(st.session_state[detail_key] or "")
        available_details = list(choices["details"].get((family, submodule), []))
        detail_options = current_or_choices(detail, available_details, include_blank=True)
        invalid_detail = (
            {detail}
            if detail and (family, submodule, detail) not in choices["triples"]
            else set()
        )
        with third:
            selected_detail = safe_selectbox(
                "Business Detail",
                detail_options,
                detail,
                key=detail_key,
                invalid=invalid_detail,
                disabled=not submodule or (not available_details and not detail),
            )
        detail = selected_detail

        meta_left, meta_right = st.columns([1, 2])
        confidence_options = list(
            dict.fromkeys(CONFIDENCE_ORDER + [str(row["naming_confidence"])])
        )
        with meta_left:
            confidence = safe_selectbox(
                "Confidence sau review",
                confidence_options,
                str(row["naming_confidence"]),
                key=confidence_key,
            )
        with meta_right:
            needs_review = st.toggle(
                "Cluster này vẫn cần review",
                value=bool(row["needs_review"]),
                key=review_key,
                help="Tắt khi người nghiệp vụ đã xác nhận nhãn.",
            )

        changed = any(
            (
                family != str(row["business_family"] or ""),
                submodule != str(row["business_submodule"] or ""),
                detail != str(row["business_detail"] or ""),
                confidence != str(row["naming_confidence"] or ""),
                needs_review != bool(row["needs_review"]),
            )
        )
        save_col, status_col = st.columns([1, 3], vertical_alignment="center")
        save_clicked = save_col.button(
            "Lưu vào database",
            key=f"map::{platform}::{cluster_id}::save",
            type="primary",
            disabled=not changed,
            use_container_width=True,
        )
        if changed:
            status_col.caption("Có thay đổi chưa lưu.")
        else:
            status_col.caption(f"Nhãn hiện tại đã đồng bộ với {DATABASE_LABEL}.")

        if save_clicked:
            try:
                database_api.update_named_cluster(
                    DATABASE_TARGET,
                    platform,
                    cluster_id,
                    family=family,
                    submodule=submodule,
                    detail=detail,
                    confidence=confidence,
                    needs_review=needs_review,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state["mapping_flash"] = (
                    f"Đã lưu {PLATFORM_LABELS[platform]} cluster {cluster_id} vào {DATABASE_LABEL}."
                )
                st.rerun()


st.title("HiFPT Taxonomy & Cluster Naming")
st.markdown(
    f'<p class="review-lead">Taxonomy 3 cấp và named clusters được lưu trực tiếp trong {DATABASE_LABEL}.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(DATABASE_LABEL)
    st.code(DATABASE_LOCATION, language=None)
    stats = database_api.database_stats(DATABASE_TARGET)
    st.caption(
        f"{stats['taxonomy_features']:,} taxonomy details\n\n"
        f"{stats['android_clusters']:,} Android clusters\n\n"
        f"{stats['ios_clusters']:,} iOS clusters\n\n"
        f"{stats['audit_events']:,} audit events"
    )
    st.info(
        f"Taxonomy hợp lệ được tự lưu; mapping cluster chỉ ghi {DATABASE_LABEL} khi bấm "
        "Lưu vào database."
    )

info_tab, taxonomy_tab, review_tab = st.tabs(
    ["1. Hướng dẫn", "2. Taxonomy 3 cấp", "3. Review cluster"]
)

with info_tab:
    st.header("Mục tiêu và dữ liệu chuẩn")
    st.write(
        "App dùng output của taxonomy naming pipeline làm source of truth. Taxonomy có ID ổn định, "
        f"mapping cluster có evidence/coverage, backend hiện tại là {DATABASE_LABEL}."
    )
    summary_cols = st.columns(5)
    summary_cols[0].metric("Business families", "16")
    summary_cols[1].metric("Submodules", "85")
    summary_cols[2].metric("Business details", f"{stats['taxonomy_features']:,}")
    summary_cols[3].metric("Android clusters", f"{stats['android_clusters']:,}")
    summary_cols[4].metric("iOS clusters", f"{stats['ios_clusters']:,}")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Luồng cập nhật")
        st.markdown(
            """
            <div class="step"><b>1.</b> Tab Taxonomy đọc bảng <code>taxonomy_features</code> từ database.</div>
            <div class="step"><b>2.</b> Sửa một feature hợp lệ sẽ tự ghi DB và cập nhật mọi named cluster đang tham chiếu taxonomy_id đó.</div>
            <div class="step"><b>3.</b> Tab Review dùng dropdown Family → Submodule → Detail từ các taxonomy row đang active.</div>
            <div class="step"><b>4.</b> Chọn mapping/confidence/need review rồi bấm <b>Lưu vào database</b> để ghi bảng <code>named_clusters</code>.</div>
            <div class="step"><b>5.</b> Mọi thay đổi tạo một record trong <code>change_audit</code>.</div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Quy tắc review")
        st.markdown(
            """
            - Ưu tiên cluster `needs_review=true` và confidence thấp.
            - Dùng top n-gram có mass lớn, không quyết định từ một token hiếm.
            - Detail được chọn sẽ lưu `taxonomy_id` tương ứng.
            - Có thể để trống Detail nếu bằng chứng chỉ đủ tới cấp Submodule.
            - Thay đổi taxonomy làm cluster liên quan quay lại trạng thái cần review.
            """
        )
        st.markdown(
            f'<div class="plain-note"><b>Persistence:</b> refresh trình duyệt không mất thay đổi vì dữ liệu nằm trong {DATABASE_LABEL}, không còn chỉ ở session state.</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Thay đổi gần đây")
    audit = database_api.load_recent_audit(DATABASE_TARGET, limit=10)
    if audit.empty:
        st.caption("Chưa có thay đổi nghiệp vụ.")
    else:
        st.dataframe(audit, hide_index=True, width="stretch")

with taxonomy_tab:
    st.header("Taxonomy tính năng HiFPT")
    st.write(
        f"Mỗi dòng là một feature 3 cấp. Sửa/xóa/thêm một dòng hợp lệ sẽ tự lưu vào {DATABASE_LABEL}. "
        "Taxonomy ID do hệ thống quản lý; dòng mới sẽ nhận ID dạng `USR.000001`."
    )
    if "taxonomy_flash" in st.session_state:
        st.success(st.session_state.pop("taxonomy_flash"))
    taxonomy_db = database_api.load_taxonomy(DATABASE_TARGET)
    edited_taxonomy = st.data_editor(
        taxonomy_db,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key="taxonomy_editor",
        disabled=["taxonomy_id"],
        column_config={
            "taxonomy_id": st.column_config.TextColumn("Taxonomy ID", width="small"),
            "business_family": st.column_config.TextColumn("Business Family", required=True),
            "business_submodule": st.column_config.TextColumn("Business Submodule", required=True),
            "business_detail": st.column_config.TextColumn("Business Detail", required=True),
        },
    )
    normalized_editor = normalize_taxonomy(edited_taxonomy)
    taxonomy_errors = validate_taxonomy(normalized_editor)
    if taxonomy_errors:
        for error in taxonomy_errors:
            st.error(error)
        st.caption(f"{DATABASE_LABEL} chưa được cập nhật vì taxonomy hiện tại chưa hợp lệ.")
    elif not normalized_editor.equals(normalize_taxonomy(taxonomy_db)):
        try:
            result = database_api.save_taxonomy(DATABASE_TARGET, normalized_editor)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["taxonomy_flash"] = (
                f"Đã lưu taxonomy vào {DATABASE_LABEL}: "
                f"{result['inserted']} thêm, {result['updated']} sửa, "
                f"{result['deactivated']} ngừng dùng; "
                f"{result['propagated_clusters']} cluster được đồng bộ."
            )
            st.session_state.pop("taxonomy_editor", None)
            st.rerun()
    else:
        st.success(
            f"{DATABASE_LABEL} đang có {len(taxonomy_db):,} details, "
            f"{taxonomy_db['business_family'].nunique():,} families và "
            f"{taxonomy_db[['business_family', 'business_submodule']].drop_duplicates().shape[0]:,} submodules."
        )
    st.download_button(
        "Tải taxonomy_features.csv",
        data=export_taxonomy_csv(taxonomy_db),
        file_name="taxonomy_features.csv",
        mime="text/csv",
        disabled=bool(taxonomy_errors),
        type="primary",
    )

with review_tab:
    st.header("Review và đặt tên cluster")
    st.caption(
        f"Dropdown đọc taxonomy active từ {DATABASE_LABEL}. Chỉnh nhãn rồi bấm Lưu vào database ở đúng cluster."
    )
    if "mapping_flash" in st.session_state:
        st.success(st.session_state.pop("mapping_flash"))
    platform = st.radio(
        "Platform",
        list(PLATFORMS),
        format_func=lambda value: PLATFORM_LABELS[value],
        horizontal=True,
        key="review_platform",
    )
    taxonomy = database_api.load_taxonomy(DATABASE_TARGET)
    mapping = database_api.load_named_clusters(DATABASE_TARGET, platform)
    confidence_options = ordered_confidences(mapping)
    filter_a, filter_b, filter_c, filter_d = st.columns([1.1, 1.5, 2.3, .8])
    with filter_a:
        review_mode = st.selectbox(
            "Need review",
            ["all", "yes", "no"],
            format_func=lambda value: {"all": "Tất cả", "yes": "Cần review", "no": "Đã review"}[value],
            key=f"review_filter_{platform}",
        )
    with filter_b:
        selected_confidences = st.multiselect(
            "Confidence",
            confidence_options,
            default=confidence_options,
            key=f"confidence_filter_{platform}",
        )
    with filter_c:
        query = st.text_input(
            "Tìm cluster ID, taxonomy ID, tên hoặc evidence",
            key=f"query_filter_{platform}",
        )
    with filter_d:
        page_size = st.selectbox(
            "Số cluster/trang", [5, 10], key=f"page_size_{platform}"
        )

    review_values = {"all": [True, False], "yes": [True], "no": [False]}[review_mode]
    filtered = filter_named_clusters(
        mapping,
        review_values=review_values,
        confidence_values=selected_confidences,
        query=query,
    )
    signature = (platform, review_mode, tuple(selected_confidences), query, page_size)
    signature_key = f"page_signature_{platform}"
    page_key = f"page_{platform}"
    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state[page_key] = 0
    total_pages = max(1, math.ceil(len(filtered) / page_size))
    current_page = min(int(st.session_state.get(page_key, 0)), total_pages - 1)
    st.session_state[page_key] = current_page
    nav_left, nav_mid, nav_right = st.columns([1, 3, 1])
    if nav_left.button(
        "Trang trước",
        disabled=current_page <= 0,
        key=f"prev_{platform}",
        width="stretch",
    ):
        st.session_state[page_key] = current_page - 1
        st.rerun()
    nav_mid.markdown(
        f"<div style='text-align:center;padding:.55rem'><b>{len(filtered):,}</b> cluster · "
        f"Trang <b>{current_page + 1}/{total_pages}</b></div>",
        unsafe_allow_html=True,
    )
    if nav_right.button(
        "Trang sau",
        disabled=current_page >= total_pages - 1,
        key=f"next_{platform}",
        width="stretch",
    ):
        st.session_state[page_key] = current_page + 1
        st.rerun()

    if filtered.empty:
        st.info("Không có cluster phù hợp với bộ lọc hiện tại.")
    else:
        start = current_page * page_size
        for _, cluster_row in filtered.iloc[start : start + page_size].iterrows():
            render_cluster_card(platform, cluster_row, taxonomy)

    st.divider()
    st.subheader(f"Xuất named clusters từ {DATABASE_LABEL}")
    export_columns = st.columns(2)
    for index, export_platform in enumerate(PLATFORMS):
        export_frame = database_api.load_named_clusters(DATABASE_TARGET, export_platform)
        with export_columns[index]:
            st.download_button(
                f"Tải {PLATFORM_LABELS[export_platform]} named clusters",
                data=export_named_clusters_csv(export_frame),
                file_name=f"{export_platform}_named_clusters.csv",
                mime="text/csv",
                type="primary" if export_platform == platform else "secondary",
                width="stretch",
                key=f"download_mapping_{export_platform}",
            )
