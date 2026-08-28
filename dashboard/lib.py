"""Shared data loading, i18n and formatting helpers for the journey-clustering dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

# The dashboard reuses the *production* segmentation code rather than restating its rules,
# so the EDA walkthrough can never drift from what the pipeline actually does. Keep the
# repository root on the path so package imports such as `src.segment` resolve correctly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CACHE_DIR = ROOT / "output" / "dashboard_cache"
# The dashboard deliberately keeps the three data products separate:
#
# * TRAIN_RUN_DIR contains the fitted artefacts and the 500k train/holdout metadata;
# * RUN_DIR contains the full-data scored inference bundle and its authoritative names;
# * SUMMARY_DIR contains compact aggregates generated from the full-data inference CSVs.
#
# Keeping these paths explicit is important: a train catalog and a full-data inference
# summary answer different questions and must not be mixed into one KPI.
RUN_DIR = ROOT / "output" / "scores" / "pca48_ngrams12_500"
TRAIN_RUN_DIR = ROOT / "output" / "partitioned_runs" / "pca48_svd48_ngrams12_500k" / "latest"
MODEL_ROOT = RUN_DIR
SUMMARY_DIR = RUN_DIR / "html_dashboard_summary"
POST_ANALYSIS_DIR = RUN_DIR / "post_analysis"
EDA_SUMMARY_FILE = POST_ANALYSIS_DIR / "eda" / "eda_summary.json"
TEST_DIR = ROOT / "output" / "test"
NAMING_FILE = RUN_DIR / "Cluster_naming.csv"
NAMING_AUDIT_FILE = RUN_DIR / "cluster_naming_audit.csv"
CUSTOMER_ANALYSIS_DIR = RUN_DIR / "shareholder_analysis"


def platform_root(platform: str) -> Path:
    return MODEL_ROOT / platform / "model_version=latest" / f"platform={platform}"


def model_dir(platform: str) -> Path:
    return platform_root(platform) / "model_output"


def train_model_dir(platform: str) -> Path:
    """Return the canonical fitted-model directory for the selected train run."""
    return TRAIN_RUN_DIR / platform


def inspection_dir(platform: str) -> Path:
    return platform_root(platform) / "inspection"


def inference_dir(platform: str) -> Path:
    return platform_root(platform)

PLATFORMS = ["android", "ios"]
PLATFORM_LABEL = {"android": "Android", "ios": "iOS"}

# Plot palette (colour-blind safe, works on light and dark themes)
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]


# ======================================================================================
# look & feel
# ======================================================================================
# Everything here is cosmetic. Colours are expressed with `currentColor` / rgba so the
# same rules read correctly on both the light and the dark Streamlit theme.
_CSS = """
<style>
@keyframes jc-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes jc-fade { from { opacity: 0; } to { opacity: 1; } }

/* page content eases in instead of snapping in on every rerun */
[data-testid="stMain"] .block-container > div { animation: jc-rise .38s cubic-bezier(.22,.61,.36,1) both; }
[data-testid="stSidebar"] { animation: jc-fade .5s ease both; }

h1 {
  background: linear-gradient(90deg, #4C78A8 0%, #72B7B2 45%, #F58518 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -.02em;
}
h2, h3 { letter-spacing: -.01em; }

/* metric tiles become cards that lift on hover */
div[data-testid="stMetric"] {
  position: relative;
  border: 1px solid rgba(128,128,128,.22);
  border-radius: 14px;
  padding: .85rem 1rem .7rem 1.15rem;
  background: linear-gradient(180deg, rgba(128,128,128,.06), rgba(128,128,128,0));
  overflow: hidden;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div[data-testid="stMetric"]::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: linear-gradient(180deg, #4C78A8, #72B7B2);
  opacity: .85;
}
div[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: rgba(76,120,168,.55);
  box-shadow: 0 10px 24px -14px rgba(0,0,0,.55);
}
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

/* tabs: soft pill + animated underline */
button[data-baseweb="tab"] { transition: color .18s ease, background .18s ease; border-radius: 8px 8px 0 0; }
button[data-baseweb="tab"]:hover { background: rgba(128,128,128,.10); }

/* tables and charts share one card language */
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
  border-radius: 12px; overflow: hidden;
  transition: box-shadow .2s ease;
}
div[data-testid="stDataFrame"]:hover { box-shadow: 0 8px 22px -16px rgba(0,0,0,.6); }
div[data-testid="stPlotlyChart"] { animation: jc-fade .5s ease both; }

hr { background: linear-gradient(90deg, rgba(76,120,168,.55), rgba(128,128,128,.12) 60%, transparent); height: 1px; border: none; }

/* reusable card used by the EDA walkthrough */
.jc-cards { display: flex; flex-wrap: wrap; gap: .6rem; margin: .2rem 0 .9rem; }
.jc-card {
  flex: 1 1 190px;
  border: 1px solid rgba(128,128,128,.22);
  border-left: 3px solid var(--jc-accent, #4C78A8);
  border-radius: 12px;
  padding: .7rem .85rem;
  background: linear-gradient(180deg, rgba(128,128,128,.07), rgba(128,128,128,0));
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.jc-card:hover { transform: translateY(-3px); box-shadow: 0 10px 24px -16px rgba(0,0,0,.6); }
.jc-card .jc-title { font-weight: 600; font-size: .93rem; margin-bottom: .18rem; }
.jc-card .jc-body { font-size: .82rem; opacity: .82; line-height: 1.35; }
.jc-card .jc-tag {
  display: inline-block; margin-top: .45rem; padding: .05rem .45rem;
  border-radius: 999px; font-size: .72rem; font-family: ui-monospace, monospace;
  background: rgba(128,128,128,.16);
}
</style>
"""


def inject_css() -> None:
    """Apply the dashboard's cosmetic layer. Safe to call once per rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)


def cards(items: list[tuple[str, str, str | None, str]]) -> None:
    """Render a row of hover cards. items = [(title, body, tag, accent_colour)]"""
    html = ['<div class="jc-cards">']
    for title, body, tag, accent in items:
        tag_html = f'<div class="jc-tag">{tag}</div>' if tag else ""
        html.append(
            f'<div class="jc-card" style="--jc-accent:{accent}">'
            f'<div class="jc-title">{title}</div>'
            f'<div class="jc-body">{body}</div>{tag_html}</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ======================================================================================
# i18n
# ======================================================================================
def lang() -> str:
    """Current UI language: 'EN' or 'VI'."""
    return st.session_state.get("lang", "EN")


def is_vi() -> bool:
    return lang() == "VI"


def t(en: str, vi: str) -> str:
    """Pick the string for the active language. Technical terms stay in English in VI copy."""
    return vi if is_vi() else en


# Business families are stored in English snake_case (cluster_name_mapping.csv) but the
# same file also carries Vietnamese labels; these maps cover both directions.
FAMILY_EN = {
    "quản lý thiết bị": "device management",
    "thanh toán": "payment",
    "hợp đồng": "contract",
    "tương tác": "engagement",
    "điều hướng": "navigation",
    "hỗ trợ": "support",
    "thương mại": "commerce",
    "chưa phân loại": "unclassified",
    "tài khoản": "account",
    "xác thực": "authentication",
    "giải trí": "entertainment",
    "khuyến mãi": "promotion",
}

FAMILY_VI = {
    "device management": "quản lý thiết bị",
    "device_management": "quản lý thiết bị",
    "payment": "thanh toán",
    "payments": "thanh toán",
    "contract": "hợp đồng",
    "contracts": "hợp đồng",
    "engagement": "tương tác",
    "navigation": "điều hướng",
    "support": "hỗ trợ",
    "commerce": "thương mại",
    "unclassified": "chưa phân loại",
    "unknown": "chưa phân loại",
    "novel": "hành vi mới",
    "account": "tài khoản",
    "authentication": "xác thực",
    "entertainment": "giải trí",
    "promotion": "khuyến mãi",
    "service management": "quản lý dịch vụ",
    "service_management": "quản lý dịch vụ",
    "feedback": "phản hồi",
}


def pretty_family(value) -> str:
    """Render a business family in the active language."""
    v = str(value)
    if is_vi():
        key = FAMILY_EN.get(v, v).replace("_", " ")
        return FAMILY_VI.get(key, FAMILY_VI.get(v, key))
    return FAMILY_EN.get(v, v).replace("_", " ")


# Each friction signal, described three ways: a plain title, what it means for the
# customer, and the statistical rule that raised it. `metric` names the fitted threshold
# in thresholds.json (see src/score.py, where the flags are set).
FRICTION_META = {
    "excessive_back": {
        "title_en": "Kept pressing back",
        "title_vi": "Bấm back liên tục",
        "meaning_en": "The customer repeatedly backed out of screens. Usually what they landed on was not what they expected, so they retreated and tried again.",
        "meaning_vi": "Khách hàng liên tục bấm back để thoát khỏi màn hình. Thường là vì màn hình mở ra không đúng thứ họ mong đợi, nên họ lùi lại và thử đường khác.",
        "metric": "back_rate_p90",
        "rule_en": "More than {v} of the steps in the journey were back taps — a level only the top 10% of journeys reach.",
        "rule_vi": "Hơn {v} số bước trong journey là thao tác back — mức mà chỉ 10% journey cao nhất mới chạm tới.",
        "fmt": "pct",
    },
    "screen_thrash": {
        "title_en": "Went in circles between screens",
        "title_vi": "Đi vòng vòng giữa các màn hình",
        "meaning_en": "The customer kept coming back to screens they had already seen — the signature of hunting for something they could not find.",
        "meaning_vi": "Khách hàng cứ quay lại những màn hình đã xem rồi — dấu hiệu điển hình của việc đang tìm thứ gì đó mà không thấy.",
        "metric": "revisit_p90",
        "rule_en": "More than {v} of the steps revisited a screen already seen in the same journey — the top 10% most repetitive journeys.",
        "rule_vi": "Hơn {v} số bước quay lại một màn hình đã xem trong cùng journey — thuộc nhóm 10% journey lặp lại nhiều nhất.",
        "fmt": "pct",
    },
    "navigation_loop": {
        "title_en": "Stuck in a loop",
        "title_vi": "Kẹt trong một vòng lặp",
        "meaning_en": "The same short screen cycle repeated over and over. This is what a dead end in the interface looks like from the data side.",
        "meaning_vi": "Một vòng vài màn hình lặp đi lặp lại. Đây chính là hình ảnh của một ngõ cụt trong giao diện khi nhìn từ phía dữ liệu.",
        "metric": "loops_p90",
        "rule_en": "The journey contained a repeating screen cycle at all. Nine out of ten journeys contain none, so any loop is already unusual.",
        "rule_vi": "Journey có chứa một vòng lặp màn hình. Chín trên mười journey không hề có vòng lặp nào, nên chỉ cần có là đã bất thường.",
        "fmt": "none",
    },
    "slow_journey": {
        "title_en": "Took much longer than normal",
        "title_vi": "Mất nhiều thời gian hơn hẳn bình thường",
        "meaning_en": "The task dragged on well past how long this kind of task usually takes — hesitation, waiting, or repeated attempts.",
        "meaning_vi": "Task kéo dài vượt xa thời gian mà loại task này thường mất — do phân vân, phải chờ, hoặc thử đi thử lại.",
        "metric": "span_p95",
        "rule_en": "Lasted longer than {v} — the slowest 5% of all journeys.",
        "rule_vi": "Kéo dài hơn {v} — thuộc nhóm 5% journey chậm nhất.",
        "fmt": "seconds",
    },
    "improbable_transitions": {
        "title_en": "Took a path this flow almost never takes",
        "title_vi": "Đi một đường mà flow này gần như không bao giờ đi",
        "meaning_en": "The journey belongs to a known task, but the customer moved between screens in an order that almost nobody else uses — they were off the beaten path.",
        "meaning_vi": "Journey vẫn thuộc một task đã biết, nhưng khách hàng di chuyển giữa các màn hình theo thứ tự mà gần như không ai khác dùng — họ đã đi chệch khỏi lối mòn.",
        "metric": "markov_p05",
        "rule_en": "Among the 5% least likely step orders, measured against the pattern this task normally follows (step log-probability below {v}).",
        "rule_vi": "Nằm trong 5% thứ tự bước ít khả năng xảy ra nhất, so với lối đi mà task này thường theo (log-probability mỗi bước thấp hơn {v}).",
        "fmt": "logprob",
    },
    "unknown_archetype": {
        "title_en": "Behaviour we have never seen",
        "title_vi": "Hành vi chưa từng thấy",
        "meaning_en": "This journey did not resemble any known task closely enough to be called one. Often a new feature, a campaign, or something broken.",
        "meaning_vi": "Journey này không đủ giống bất kỳ task nào đã biết để được gọi tên. Thường là do một tính năng mới, một chiến dịch, hoặc một thứ gì đó đang lỗi.",
        "metric": "distance_p95",
        "rule_en": "No known task was close enough to claim it — the journey sits beyond every type's own boundary (model distance above ≈ {v}).",
        "rule_vi": "Không task nào đã biết đủ gần để nhận nó — journey nằm ngoài ranh giới của mọi type (khoảng cách trong model vượt ≈ {v}).",
        "fmt": "raw",
    },
}


def _fmt_threshold(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "seconds":
        return f"{value:.0f} " + t("seconds", "giây")
    if kind == "logprob":
        return f"{value:.1f}"
    if kind == "raw":
        return f"{value:.2f}"
    return ""


def friction_title(flag: str) -> str:
    meta = FRICTION_META.get(flag)
    if not meta:
        return flag
    return meta["title_vi"] if is_vi() else meta["title_en"]


def friction_meaning(flag: str) -> str:
    meta = FRICTION_META.get(flag)
    if not meta:
        return ""
    return meta["meaning_vi"] if is_vi() else meta["meaning_en"]


def friction_rule(flag: str, platforms: list[str] | None = None) -> str:
    """The statistical rule behind a flag, with the run's real cut-off filled in.

    Cut-offs are fitted per platform; when both are in scope, both are shown.
    """
    meta = FRICTION_META.get(flag)
    if not meta:
        return ""
    template = meta["rule_vi"] if is_vi() else meta["rule_en"]
    if meta["fmt"] == "none" or "{v}" not in template:
        return template
    th = load_thresholds()
    picks = [p for p in (platforms or list(th)) if p in th]
    values = [(p, th[p].get(meta["metric"])) for p in picks]
    values = [(p, v) for p, v in values if v is not None]
    if not values:
        return template.replace("{v}", "—")
    if len(values) == 1:
        return template.replace("{v}", _fmt_threshold(values[0][1], meta["fmt"]))
    shown = " / ".join(f"{PLATFORM_LABEL.get(p, p.title())} {_fmt_threshold(v, meta['fmt'])}" for p, v in values)
    return template.replace("{v}", shown)


def friction_explain() -> dict:
    """Backwards-compatible plain-meaning lookup."""
    return {flag: friction_meaning(flag) for flag in FRICTION_META}


# ======================================================================================
# loaders
# ======================================================================================
def missing_cache_notice(what: str) -> None:
    st.error(
        t(
            f"Cache for **{what}** not found. Build it first:",
            f"Chưa có cache cho **{what}**. Hãy build trước:",
        )
        + "\n\n```bash\npython dashboard/prepare_dashboard_data.py\n```"
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_eda() -> dict:
    # Raw EDA is a deliberate, separate preparation job.  Never fall back to
    # prepared-journey parquet footer counts or an old cache: those are not raw
    # event statistics and caused the dashboard conflict this contract fixes.
    if EDA_SUMMARY_FILE.exists():
        payload = json.loads(EDA_SUMMARY_FILE.read_text(encoding="utf-8"))
        # A capped EDA is useful for validating the parser, but must never be
        # presented as the production-wide raw-data profile.
        if payload.get("schema_version") == "raw-eda-v1" and payload.get("status") == "ready" and payload.get("mode") == "full":
            return payload
    return {
        "schema_version": "raw-eda-v1",
        "status": "not_prepared",
        "source": "raw event EDA has not been prepared",
        "per_file": {},
        "platform_vocab_overlap": {},
    }


@st.cache_data(show_spinner=False)
def load_customer_analysis_csv(name: str) -> pd.DataFrame:
    """Load one customer-based artefact generated beside the scored bundle."""
    path = CUSTOMER_ANALYSIS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    # Older shareholder scripts called the stable key object_id. Keep old cached
    # outputs readable while the canonical outputs use the requested customer_id.
    if "customer_id" not in frame.columns and "object_id" in frame.columns:
        frame["customer_id"] = frame["object_id"]
    return frame


@st.cache_data(show_spinner=False)
def load_summary_csv(name: str) -> pd.DataFrame:
    """Load one compact aggregate from the full-data inference summary.

    The summary directory is the dashboard's primary analytics source.  It is generated
    out-of-core from the multi-GB named inference CSVs, so opening the app never requires
    materialising all scored parquet partitions in memory.
    """
    path = SUMMARY_DIR / name
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    return frame


@st.cache_data(show_spinner=False)
def load_summary_manifest() -> dict:
    path = SUMMARY_DIR / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_inference_profile(platform: str) -> dict:
    """Read the scorer's small per-platform inspection summary."""
    path = inspection_dir(platform) / f"{platform}_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_train_run() -> dict:
    """Return the training metadata from the selected partitioned model bundle."""
    cached = CACHE_DIR / "train_run.json"
    if cached.exists() and not any((model_dir(p) / f"{p}_run_config.json").exists() for p in PLATFORMS):
        return json.loads(cached.read_text(encoding="utf-8"))

    out = {"run_slug": TRAIN_RUN_DIR.parent.name, "source": str(TRAIN_RUN_DIR), "platforms": {}}
    for platform in PLATFORMS:
        # The partitioned training output is canonical.  The scored bundle contains a
        # convenient copy of these files, but it is not the source of truth for training.
        cfg_path = train_model_dir(platform) / f"{platform}_run_config.json"
        if not cfg_path.exists():
            cfg_path = model_dir(platform) / f"{platform}_run_config.json"
        summary_path = inspection_dir(platform) / f"{platform}_summary.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        manifest = cfg.get("training_manifest", {})
        hdb = cfg.get("hdbscan", {})
        holdout = train_model_dir(platform) / f"{platform}_scored_holdout.csv"
        if not holdout.exists():
            holdout = model_dir(platform) / f"{platform}_scored_holdout.csv"
        hold = {
            "n_journeys": int(manifest.get("holdout_journeys", 0) or 0),
            "n_sessions": 0,
            "assignment_type": {},
        }
        if holdout.exists():
            # Keep this summary bounded: the detailed rows remain available on the inference
            # page, while training only needs the holdout headline metrics.
            sample = pd.read_csv(
                holdout,
                usecols=lambda c: c in {"session_id", "cluster", "assignment_type", "distance_to_centroid"},
                low_memory=False,
            )
            hold["n_journeys"] = len(sample)
            hold["n_sessions"] = int(sample["session_id"].nunique()) if "session_id" in sample else 0
            if "assignment_type" in sample:
                assignment = sample["assignment_type"].fillna("").astype(str)
            elif "cluster" in sample:
                # The current scorer's holdout CSV predates the named inference export and
                # has no assignment_type.  Cluster -1 is the exact unresolved convention.
                cluster = pd.to_numeric(sample["cluster"], errors="coerce")
                assignment = pd.Series(np.where(cluster.eq(-1), "unassigned_novel", "C_primary"), index=sample.index)
            else:
                assignment = pd.Series(dtype=str)
            if not assignment.empty:
                hold["assignment_type"] = {str(k): int(v) for k, v in assignment.value_counts(dropna=False).items()}
            if "distance_to_centroid" in sample:
                distance = pd.to_numeric(sample["distance_to_centroid"], errors="coerce").dropna()
                hold["distance_q"] = {
                    str(k): round(float(v), 3)
                    for k, v in distance.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).items()
                }
        out["platforms"][platform] = {
            "run_config": cfg,
            "holdout": hold,
            "summary": summary,
            # The bundle does not duplicate a full training summary; leave this optional
            # section empty so pages do not present partial shape statistics as if they were
            # complete.
            "journeys": {},
        }
    return out


@st.cache_data(show_spinner=False)
def load_inference(platform: str) -> pd.DataFrame:
    """Load only bounded inference examples for audit/detail views.

    Aggregate pages must use :func:`load_summary_csv`.  This function intentionally does
    not scan the 24 GB partition lake; it returns the bounded examples generated by the
    summary extractor and keeps a legacy cache fallback for older local checkouts.
    """
    example_path = SUMMARY_DIR / "journey_examples.csv"
    if example_path.exists():
        df = pd.read_csv(example_path, encoding="utf-8-sig", low_memory=False)
        df = df[df.get("platform", platform).eq(platform)].copy() if "platform" in df.columns else df
    else:
        legacy = CACHE_DIR / f"inference_{platform}.parquet"
        if not legacy.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(legacy)
        except (ImportError, RuntimeError, ValueError):
            return pd.DataFrame()

    # Examples already carry their names.  The fallback normalisation below keeps older
    # cached examples and current bounded examples on the same schema.
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")

    # Keep one stable schema for both the current partitions and older named exports.
    aliases = {
        "next_action": "effective_next_action",
        "next_action_share": "effective_next_action_share",
    }
    for source, target in aliases.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]
    if "assignment_type" not in df.columns:
        cluster_series = df["cluster"] if "cluster" in df.columns else pd.Series(-1, index=df.index)
        df["assignment_type"] = np.where(cluster_series.eq(-1), "unassigned_novel", "C_primary")
    if "behavioral_friction_flags" not in df.columns:
        df["behavioral_friction_flags"] = df.get("friction_flags", "")
    for col, default in {
        "cluster_name": "unclassified.dynamic",
        "cluster_name_vi": "Chưa phân loại",
        "business_family": "unclassified",
        "business_family_vi": "Chưa phân loại",
        "naming_confidence": "not_applicable",
    }.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    df["start_ts"] = pd.to_datetime(df["start_ts"], errors="coerce", utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_thresholds() -> dict:
    """Fitted percentile cut-offs the scorer uses to raise each friction flag."""
    path = CACHE_DIR / "thresholds.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_showcase() -> dict:
    """One real session carried end to end, used by the overview page."""
    if (MODEL_ROOT / "Cluster_naming.csv").exists():
        # The scored bundle intentionally contains journey outputs, not a second raw-event
        # extract. Do not show the old cache's unrelated showcase beside the new model run.
        return {}
    path = CACHE_DIR / "showcase.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_run_csv(name: str) -> pd.DataFrame:
    """Resolve a run artefact from the partitioned bundle.

    Callers still use the old logical names (``android_cluster_ngrams.csv`` etc.); this
    resolver keeps those page APIs stable while pointing them at ``model_output``.
    """
    candidates = [RUN_DIR / name]
    for platform in PLATFORMS:
        if name.startswith(f"{platform}_"):
            candidates.extend([train_model_dir(platform) / name, model_dir(platform) / name, inspection_dir(platform) / name])
    if name == "cluster_name_mapping.csv":
        candidates.insert(0, NAMING_FILE)
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


@st.cache_data(show_spinner=False)
def load_cluster_names() -> pd.DataFrame:
    if not NAMING_FILE.exists():
        missing_cache_notice(t("cluster name mapping", "bảng tên cluster"))
    df = pd.read_csv(NAMING_FILE, encoding="utf-8-sig", low_memory=False)
    # Cluster_naming.csv is intentionally business-facing and stores the readable label in
    # Vietnamese.  Preserve that label while exposing a stable English/code label to the
    # existing bilingual page code.
    raw_name = df.get("cluster_name_vi", df["cluster_name"]).fillna("Unclassified / mixed journeys")
    df["cluster_name_vi"] = raw_name
    df["cluster_name"] = (
        df.get("canonical_level_2_code", raw_name)
        .fillna(raw_name)
        .astype(str)
        .str.replace(".", " · ", regex=False)
    )
    raw_family = df["business_family"].fillna("Chưa phân loại")
    df["business_family_vi"] = raw_family
    if "business_family_code" in df.columns:
        family_code = df["business_family_code"]
    else:
        # The authoritative naming file currently stores the canonical level-2 code
        # (for example ``contract.document``) but not a separate family-code column.
        # Derive the stable English family key from that code so the cluster page does
        # not accidentally display Vietnamese labels in its EN mode.
        canonical = df.get("canonical_level_2_code", pd.Series(index=df.index, dtype="object"))
        family_code = canonical.astype("string").str.split(".").str[0]
        family_code = family_code.where(family_code.notna() & family_code.ne(""), raw_family)
    df["business_family"] = family_code.fillna("unclassified")
    if "cluster_id" in df.columns:
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
        df["cluster"] = df["cluster_id"]
    if df.empty:
        missing_cache_notice(t("cluster name mapping", "bảng tên cluster"))
    return df


@st.cache_data(show_spinner=False)
def load_shareholder_catalog() -> dict:
    path = inspection_dir("android") / "shareholder_cluster_catalog.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_shareholder_catalog_vi() -> dict:
    path = inspection_dir("android") / "shareholder_cluster_catalog_vi.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ======================================================================================
# formatting
# ======================================================================================
def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """items = [(label, value, help_text)]"""
    cols = st.columns(len(items))
    for col, (label, value, helptext) in zip(cols, items):
        col.metric(label, value, help=helptext)


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(x, digits: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def hist_to_df(h: dict, value_name: str = "count") -> pd.DataFrame:
    edges = h["edges"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    return pd.DataFrame({"bin": centers, value_name: h["counts"]})


def quantile_table(qdict: dict, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"quantile": list(qdict.keys()), label: list(qdict.values())}
    )


def explode_flags(series: pd.Series) -> pd.Series:
    """friction_flags is a '|'-joined string; return exploded flag counts."""
    s = series.fillna("").astype(str)
    s = s[s != ""]
    if s.empty:
        return pd.Series(dtype="int64")
    return s.str.split("|").explode().str.strip().value_counts()
