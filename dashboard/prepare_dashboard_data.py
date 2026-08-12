"""Precompute aggregates for the Streamlit dashboard.

The raw production sample (data/train_data/raw_data_production/data_raw_sample) is
~410 MB across 6 CSVs and the training run writes multi-hundred-MB artifacts, so the
dashboard never touches them directly. This script reduces everything to a small
JSON cache under output/dashboard_cache/.

Run:
    python dashboard/prepare_dashboard_data.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "train_data"
RUN_DIR = (
    ROOT
    / "output"
    / "EXACT_ch-c45i30o20_ng1-3_svd64_fdf3_mf20000_nw0p35_mcs100_ms5_sel-eom_gap90_jmin4_tdf3_ent0_chr1_boot0"
)
CACHE_DIR = ROOT / "output" / "dashboard_cache"

RAW_FILES = {
    "android_t3_1k.csv": ("android", "T3"),
    "android_t4_1k.csv": ("android", "T4"),
    "android_t5_1k.csv": ("android", "T5"),
    "ios_t3_1k.csv": ("ios", "T3"),
    "ios_t4_1k.csv": ("ios", "T4"),
    "ios_t5_1k.csv": ("ios", "T5"),
}

RAW_USECOLS = [
    "device_id",
    "session_id",
    "customer_id",
    "platform",
    "key",
    "segmentation_name",
    "client_time",
]

UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
LONG_NUM_RE = re.compile(r"\b\d{4,}\b")
QUANTILES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def q(series: pd.Series) -> dict:
    if len(series) == 0:
        return {str(x): None for x in QUANTILES}
    vals = series.quantile(QUANTILES)
    return {str(k): round(float(v), 3) for k, v in vals.items()}


def hist(series: pd.Series, bins) -> dict:
    counts, edges = np.histogram(series.dropna().to_numpy(), bins=bins)
    return {"counts": counts.tolist(), "edges": [float(e) for e in edges]}


def parse_client_time(s: pd.Series) -> pd.Series:
    """android = ISO8601 strings, ios = epoch milliseconds."""
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.9:
        return pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


# --------------------------------------------------------------------------------------
# 1. Raw-data EDA
# --------------------------------------------------------------------------------------
def profile_raw_file(path: Path, platform: str, week: str) -> dict:
    df = pd.read_csv(path, usecols=lambda c: c in RAW_USECOLS, low_memory=False)
    df["ts"] = parse_client_time(df["client_time"])
    df = df.dropna(subset=["ts", "session_id"])
    df["segmentation_name"] = df["segmentation_name"].fillna("<missing>").astype(str)
    df["key"] = df["key"].fillna("<missing>").astype(str)
    df = df.sort_values(["session_id", "ts"], kind="stable")

    seg_counts = df["segmentation_name"].value_counts()
    session_len = df.groupby("session_id", observed=True).size()
    span = df.groupby("session_id", observed=True)["ts"].agg(["min", "max"])
    session_span_s = (span["max"] - span["min"]).dt.total_seconds()

    gaps = df.groupby("session_id", observed=True)["ts"].diff().dt.total_seconds()
    gaps = gaps[gaps.notna() & (gaps >= 0)]

    # taxonomy pressure signals measured on the distinct vocabulary, not on events
    vocab = pd.Series(seg_counts.index.astype(str))
    depth = vocab.str.count("/")
    signals = {
        "vocab_size": int(vocab.size),
        "singletons": int((seg_counts == 1).sum()),
        "singleton_share": round(float((seg_counts == 1).mean()), 4),
        "top100_event_coverage": round(float(seg_counts.head(100).sum() / seg_counts.sum()), 4),
        "top500_event_coverage": round(float(seg_counts.head(500).sum() / seg_counts.sum()), 4),
        "with_uuid": int(vocab.str.contains(UUID_RE).sum()),
        "with_long_number": int(vocab.str.contains(LONG_NUM_RE).sum()),
        "with_url": int(vocab.str.contains("http", case=False).sum()),
        "with_query_string": int(vocab.str.contains(r"\?").sum()),
        "depth_hist": Counter(depth.clip(upper=6).tolist()),
    }
    signals["depth_hist"] = {str(int(k)): int(v) for k, v in sorted(signals["depth_hist"].items())}

    per_day = df.groupby(df["ts"].dt.date, observed=True).size()
    hour_hist = df["ts"].dt.hour.value_counts().sort_index()

    # one representative raw session, untouched, to show what a "sequence" looks like
    demo_session = session_len[(session_len >= 12) & (session_len <= 40)]
    demo_rows = []
    if len(demo_session):
        sid = demo_session.index[0]
        sub = df[df["session_id"] == sid].head(30)
        demo_rows = [
            {
                "key": r.key,
                "segmentation_name": r.segmentation_name,
                "client_time": r.ts.isoformat(),
                "gap_s": None,
            }
            for r in sub.itertuples()
        ]
        for i in range(1, len(demo_rows)):
            prev = pd.Timestamp(demo_rows[i - 1]["client_time"])
            cur = pd.Timestamp(demo_rows[i]["client_time"])
            demo_rows[i]["gap_s"] = round((cur - prev).total_seconds(), 2)

    action_top = (
        df.loc[df["key"] == "action", "segmentation_name"].value_counts().head(25)
    )
    view_top = df.loc[df["key"] == "view", "segmentation_name"].value_counts().head(25)

    return {
        "file": path.name,
        "platform": platform,
        "week": week,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "rows": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()),
        "n_devices": int(df["device_id"].nunique()),
        "n_customers": int(df["customer_id"].nunique()),
        "ts_min": df["ts"].min().isoformat(),
        "ts_max": df["ts"].max().isoformat(),
        "key_counts": {str(k): int(v) for k, v in df["key"].value_counts().items()},
        "action_share": round(float((df["key"] == "action").mean()), 4),
        "session_len_q": q(session_len),
        "session_span_q": q(session_span_s),
        "gap_q": q(gaps),
        "gap_over_90s_share": round(float((gaps > 90).mean()), 4),
        "gap_over_1800s_share": round(float((gaps > 1800).mean()), 4),
        "sessions_per_customer_q": q(df.groupby("customer_id", observed=True)["session_id"].nunique()),
        "events_per_day": {str(k): int(v) for k, v in per_day.items()},
        "hour_hist": {str(k): int(v) for k, v in hour_hist.items()},
        "top_segments": [[str(k), int(v)] for k, v in seg_counts.head(40).items()],
        "top_action_tokens": [[str(k), int(v)] for k, v in action_top.items()],
        "top_view_tokens": [[str(k), int(v)] for k, v in view_top.items()],
        "taxonomy_signals": signals,
        "session_len_hist": hist(session_len.clip(upper=400), bins=40),
        "demo_session": demo_rows,
        "_vocab": set(vocab.tolist()),  # stripped before serialisation
    }


def build_raw_eda() -> dict:
    per_file = {}
    vocab_by_platform: dict[str, set] = {}
    for fname, (platform, week) in RAW_FILES.items():
        path = RAW_DIR / fname
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        print(f"  profiling {fname} ...")
        stats = profile_raw_file(path, platform, week)
        vocab_by_platform.setdefault(platform, set()).update(stats.pop("_vocab"))
        per_file[fname] = stats

    overlap = {}
    if "android" in vocab_by_platform and "ios" in vocab_by_platform:
        a, i = vocab_by_platform["android"], vocab_by_platform["ios"]
        overlap = {
            "android_only": len(a - i),
            "ios_only": len(i - a),
            "shared": len(a & i),
            "jaccard": round(len(a & i) / max(len(a | i), 1), 4),
            "shared_examples": sorted(list(a & i))[:15],
        }
    return {"per_file": per_file, "platform_vocab_overlap": overlap}


# --------------------------------------------------------------------------------------
# 2. Training-run artefacts
# --------------------------------------------------------------------------------------
def profile_journeys(path: Path) -> dict:
    cols = [
        "journey_id", "session_id", "device_id", "customer_id", "boundary_reason",
        "n_events_raw", "n_events_final", "n_unique_tokens", "n_dedup_removed",
        "n_loop_removed", "action_ratio", "back_rate", "revisit_ratio",
        "span_seconds", "median_gap_s", "cluster", "markov_logprob", "anomaly_flag",
        "entry_token", "exit_token",
    ]
    sessions, devices = set(), set()
    n = 0
    boundary = Counter()
    clusters = Counter()
    entry = Counter()
    exit_ = Counter()
    numeric_parts = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in cols, chunksize=200_000, low_memory=False):
        n += len(chunk)
        sessions.update(chunk["session_id"].unique().tolist())
        devices.update(chunk["device_id"].unique().tolist())
        boundary.update(chunk["boundary_reason"].value_counts().to_dict())
        clusters.update(chunk["cluster"].value_counts().to_dict())
        entry.update(chunk["entry_token"].value_counts().head(60).to_dict())
        exit_.update(chunk["exit_token"].value_counts().head(60).to_dict())
        numeric_parts.append(
            chunk[[
                "n_events_raw", "n_events_final", "n_unique_tokens", "n_dedup_removed",
                "n_loop_removed", "action_ratio", "back_rate", "revisit_ratio",
                "span_seconds", "median_gap_s", "markov_logprob",
            ]].astype("float32")
        )
    num = pd.concat(numeric_parts, ignore_index=True)
    return {
        "n_journeys": n,
        "n_sessions": len(sessions),
        "n_devices": len(devices),
        "boundary_reason": {str(k): int(v) for k, v in boundary.most_common()},
        "cluster_sizes": {str(k): int(v) for k, v in clusters.most_common()},
        "top_entry_tokens": [[str(k), int(v)] for k, v in entry.most_common(20)],
        "top_exit_tokens": [[str(k), int(v)] for k, v in exit_.most_common(20)],
        "length_hist": hist(num["n_events_final"].clip(upper=80), bins=39),
        "span_hist": hist(num["span_seconds"].clip(upper=600), bins=40),
        "action_ratio_hist": hist(num["action_ratio"], bins=20),
        "markov_hist": hist(num["markov_logprob"].dropna().clip(lower=-10), bins=40),
        "quantiles": {
            c: q(num[c]) for c in ["n_events_final", "span_seconds", "action_ratio", "back_rate", "revisit_ratio"]
        },
        "compression": {
            "events_raw": int(num["n_events_raw"].sum()),
            "events_final": int(num["n_events_final"].sum()),
            "dedup_removed": int(num["n_dedup_removed"].sum()),
            "loop_removed": int(num["n_loop_removed"].sum()),
        },
    }


def profile_holdout(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    out = {"n_journeys": int(len(df)), "n_sessions": int(df["session_id"].nunique())}
    for col in ["assignment_type", "geometric_anomaly", "generative_anomaly", "severe_anomaly"]:
        if col in df.columns:
            out[col] = {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}
    if "cluster" in df.columns:
        out["cluster_sizes"] = {str(k): int(v) for k, v in df["cluster"].value_counts().items()}
    if "distance_to_centroid" in df.columns:
        out["distance_q"] = q(df["distance_to_centroid"])
    return out


def build_train_artifacts() -> dict:
    out = {"run_slug": RUN_DIR.name, "platforms": {}}
    cfg = RUN_DIR / "experiment_config.json"
    if cfg.exists():
        out["experiment_config"] = json.loads(cfg.read_text(encoding="utf-8"))
    for platform in ["android", "ios"]:
        entry: dict = {}
        rc = RUN_DIR / f"{platform}_run_config.json"
        if rc.exists():
            entry["run_config"] = json.loads(rc.read_text(encoding="utf-8"))
        jp = RUN_DIR / f"{platform}_journeys.csv"
        if jp.exists():
            print(f"  profiling {jp.name} ...")
            entry["journeys"] = profile_journeys(jp)
        hp = RUN_DIR / f"{platform}_scored_holdout.csv"
        if hp.exists():
            print(f"  profiling {hp.name} ...")
            entry["holdout"] = profile_holdout(hp)
        out["platforms"][platform] = entry
    return out


# --------------------------------------------------------------------------------------
# 3. Inference on output/test
# --------------------------------------------------------------------------------------
TEST_FILES = {
    "android": ROOT / "output" / "test" / "android_t5_1k_android_scored_named.csv",
    "ios": ROOT / "output" / "test" / "ios_t5_1k_ios_scored_named.csv",
}


def build_inference_slim() -> None:
    """Write a slim per-platform table the inference page can load in full."""
    name_map = pd.read_csv(RUN_DIR / "cluster_name_mapping.csv", encoding="utf-8-sig")
    keep = [
        "journey_id", "session_id", "device_id", "customer_id", "start_ts", "end_ts",
        "boundary_reason", "n_events_final", "n_unique_tokens", "action_ratio",
        "back_rate", "revisit_ratio", "span_seconds", "entry_token", "exit_token",
        "cluster", "nearest_cluster", "distance_to_centroid", "distance_limit",
        "markov_logprob", "geometric_anomaly", "generative_anomaly", "severe_anomaly",
        "friction_flags", "behavioral_friction_flags", "assignment_type",
        "effective_next_action", "effective_next_action_share", "sequence",
    ]
    for platform, path in TEST_FILES.items():
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        print(f"  slimming {path.name} ...")
        df = pd.read_csv(path, usecols=lambda c: c in keep, encoding="utf-8-sig", low_memory=False)
        # The scored files carry a broken cluster_name_en column (every row inherits the
        # noise-cluster label), so names are re-joined from the run's name mapping.
        nm = name_map[name_map["platform"] == platform][
            ["cluster_id", "cluster_name", "cluster_name_vi", "business_family",
             "business_family_vi", "naming_confidence"]
        ].rename(columns={"cluster_id": "cluster"})
        df = df.merge(nm, on="cluster", how="left")
        df["cluster_name"] = df["cluster_name"].fillna("Unclassified / mixed journeys")
        df["business_family"] = df["business_family"].fillna("unknown")
        df["naming_confidence"] = df["naming_confidence"].fillna("not_applicable")
        df["sequence"] = df["sequence"].astype(str).str.slice(0, 400)
        out = CACHE_DIR / f"inference_{platform}.parquet"
        df.to_parquet(out, index=False)
        print(f"    -> {out.name} ({len(df):,} journeys, {out.stat().st_size/1e6:.1f} MB)")


# --------------------------------------------------------------------------------------
# 4. End-to-end showcase: one real session, from raw events to scored journeys
# --------------------------------------------------------------------------------------
def build_showcase(platform: str = "android") -> dict:
    """Pick one real session and carry it through the whole pipeline.

    The overview page uses this to show input and output side by side on the *same*
    session, so the story is traceable rather than illustrative.
    """
    inf_path = CACHE_DIR / f"inference_{platform}.parquet"
    raw_path = RAW_DIR / f"{platform}_t5_1k.csv"
    if not inf_path.exists() or not raw_path.exists():
        return {}

    df = pd.read_parquet(inf_path)
    grp = df.groupby("session_id").agg(
        journeys=("journey_id", "count"),
        names=("cluster_name", "nunique"),
        families=("business_family", "nunique"),
        events=("n_events_final", "sum"),
        flagged=("friction_flags", lambda s: s.notna().sum()),
        predicted=("effective_next_action", lambda s: s.notna().sum()),
    )
    # A good teaching example: one session holding two different journey types from two
    # different business families, one of them flagged, one with a predicted next action.
    cand = grp[
        (grp["journeys"] == 2)
        & (grp["names"] == 2)
        & (grp["families"] == 2)
        & grp["events"].between(14, 22)
        & (grp["flagged"] >= 1)
        & (grp["predicted"] >= 1)
    ]
    if cand.empty:
        cand = grp[(grp["journeys"] == 2) & (grp["names"] == 2)]
    if cand.empty:
        return {}
    session_id = sorted(cand.index)[0]

    journeys = df[df["session_id"] == session_id].sort_values("start_ts")
    keep_cols = [
        "journey_id", "cluster", "cluster_name", "cluster_name_vi", "business_family",
        "business_family_vi", "naming_confidence", "boundary_reason", "n_events_raw",
        "n_events_final", "span_seconds", "action_ratio", "back_rate", "entry_token",
        "exit_token", "sequence", "assignment_type", "distance_to_centroid",
        "distance_limit", "markov_logprob", "friction_flags", "behavioral_friction_flags",
        "effective_next_action", "effective_next_action_share", "start_ts", "end_ts",
    ]
    keep_cols = [c for c in keep_cols if c in journeys.columns]
    journeys_out = journeys[keep_cols].astype(object).where(pd.notna(journeys[keep_cols]), None)

    raw_rows = []
    for chunk in pd.read_csv(raw_path, chunksize=200_000, low_memory=False):
        hit = chunk[chunk["session_id"] == session_id]
        if len(hit):
            raw_rows.append(hit)
    raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    if not raw.empty:
        raw["ts"] = parse_client_time(raw["client_time"])
        raw = raw.sort_values("ts")
        raw["gap_s"] = raw["ts"].diff().dt.total_seconds().round(2)
        raw = raw[["key", "segmentation_name", "ts", "gap_s"]].head(40)
        raw["ts"] = raw["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "platform": platform,
        "session_id": session_id,
        "source_file": f"{platform}_t5_1k.csv",
        "raw_events": raw.astype(object).where(pd.notna(raw), None).to_dict("records") if not raw.empty else [],
        "raw_event_count": int(len(raw)),
        "journeys": [
            {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in rec.items()}
            for rec in journeys_out.to_dict("records")
        ],
    }


# --------------------------------------------------------------------------------------
# 5. Fitted friction thresholds, so the dashboard can quote the real rule
# --------------------------------------------------------------------------------------
def build_thresholds() -> dict:
    """Read the percentile cut-offs the scorer actually uses to raise each friction flag.

    They live inside the fitted scorer pickle, which needs src/ on the path to unpickle.
    """
    import dataclasses
    import pickle

    sys.path.insert(0, str(ROOT / "src"))
    out: dict = {}
    for platform in ["android", "ios"]:
        path = RUN_DIR / f"{platform}_journey_scorer.pkl"
        if not path.exists():
            continue
        try:
            with path.open("rb") as fh:
                scorer = pickle.load(fh)
            th = scorer.thresholds
            out[platform] = dataclasses.asdict(th) if dataclasses.is_dataclass(th) else dict(vars(th))
        except Exception as exc:  # noqa: BLE001 - the dashboard degrades gracefully without this
            print(f"    ! could not read thresholds for {platform}: {exc}")
    return out


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] raw-data EDA")
    eda = build_raw_eda()
    eda["generated_at"] = datetime.now(timezone.utc).isoformat()
    (CACHE_DIR / "eda.json").write_text(json.dumps(eda, ensure_ascii=False), encoding="utf-8")

    print("[2/5] training-run artefacts")
    train = build_train_artifacts()
    train["generated_at"] = datetime.now(timezone.utc).isoformat()
    (CACHE_DIR / "train_run.json").write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")

    print("[3/5] inference tables")
    build_inference_slim()

    print("[4/5] end-to-end showcase session")
    showcase = build_showcase()
    if showcase:
        (CACHE_DIR / "showcase.json").write_text(json.dumps(showcase, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"    -> session {showcase['session_id']} "
              f"({showcase['raw_event_count']} raw events -> {len(showcase['journeys'])} journeys)")
    else:
        print("    ! could not build a showcase session")

    print("[5/5] fitted friction thresholds")
    thresholds = build_thresholds()
    if thresholds:
        (CACHE_DIR / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False), encoding="utf-8")
        print(f"    -> {', '.join(thresholds)}")

    print(f"\nCache written to {CACHE_DIR}")


if __name__ == "__main__":
    main()
