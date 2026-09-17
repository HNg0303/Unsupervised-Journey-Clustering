#!/usr/bin/env python3
"""Build one self-contained July journey intelligence HTML report."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/lake/raw_events/july"
JOURNEYS = ROOT / "data/lake/journeys/july"
TRAIN = ROOT / "output/partitioned_runs/july_pca48_svd48_ngram12_1000/latest"
SCORES = ROOT / "output/scores/july"
OUT = SCORES / "july_journey_intelligence.html"


def records(cur: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    result = cur.execute(sql)
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row)) for row in result.fetchall()]


def scalar(cur: duckdb.DuckDBPyConnection, sql: str):
    return cur.execute(sql).fetchone()[0]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mapping_rows(platform: str) -> list[dict]:
    path = SCORES / platform / "model_version=latest" / f"{platform}_scores_named.cluster_mapping.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_data() -> dict:
    con = duckdb.connect()
    con.execute("PRAGMA threads=6")
    con.execute("PRAGMA memory_limit='6GB'")
    data: dict[str, object] = {"generated": "2026-09-15", "period": "July data bundle"}

    raw_summary, raw_top, raw_daily = [], [], []
    journey_summary, boundary, journey_hour, journey_daily = [], [], [], []
    prod_summary, prod_family, prod_hierarchy, prod_type, prod_daily, prod_friction, prod_next = [], [], [], [], [], [], []
    customer_summary, customer_bands, customer_type_reach = [], [], []
    learned_summary, learned_top, naming_quality, train_timing = [], [], [], []

    for platform in ("android", "ios"):
        raw_glob = (RAW / f"platform={platform}" / "*" / "*.parquet").as_posix()
        journey_glob = (JOURNEYS / f"platform={platform}" / "*.parquet").as_posix()
        score_glob = (SCORES / platform / "model_version=latest" / f"platform={platform}" / "*.parquet").as_posix()
        map_path = (SCORES / platform / "model_version=latest" / f"{platform}_scores_named.cluster_mapping.csv").as_posix()

        raw_summary += records(con, f"""
            WITH x AS (
              SELECT *, to_timestamp(try_cast(client_time AS DOUBLE) / 1000.0) ts
              FROM read_parquet('{raw_glob}')
            )
            SELECT '{platform}' platform, count(*) events,
              count(DISTINCT session_id) sessions, count(DISTINCT device_id) devices,
              count(DISTINCT nullif(customer_id,'')) customers,
              avg(CASE WHEN lower(key)='action' THEN 1.0 ELSE 0.0 END) action_share,
              sum(CASE WHEN lower(key)='view' THEN 1 ELSE 0 END) view_events,
              sum(CASE WHEN lower(key)='action' THEN 1 ELSE 0 END) action_events,
              sum(CASE WHEN ts IS NULL OR ts < TIMESTAMPTZ '2025-01-01' OR ts >= TIMESTAMPTZ '2028-01-01' THEN 1 ELSE 0 END) suspect_timestamps,
              min(ts) FILTER (WHERE ts >= TIMESTAMPTZ '2025-01-01' AND ts < TIMESTAMPTZ '2028-01-01') first_ts,
              max(ts) FILTER (WHERE ts >= TIMESTAMPTZ '2025-01-01' AND ts < TIMESTAMPTZ '2028-01-01') last_ts
            FROM x
        """)
        raw_top += records(con, f"""
            SELECT '{platform}' platform, lower(key) event_kind, segmentation_name AS "label", count(*) events
            FROM read_parquet('{raw_glob}')
            WHERE lower(key) IN ('view','action') AND segmentation_name IS NOT NULL
            GROUP BY 1,2,3 QUALIFY row_number() OVER (PARTITION BY lower(key) ORDER BY count(*) DESC) <= 10
            ORDER BY event_kind, events DESC
        """)
        raw_daily += records(con, f"""
            SELECT '{platform}' platform, cast(to_timestamp(try_cast(client_time AS DOUBLE) / 1000.0) AS DATE) date, count(*) events
            FROM read_parquet('{raw_glob}')
            WHERE to_timestamp(try_cast(client_time AS DOUBLE) / 1000.0) >= TIMESTAMPTZ '2025-01-01'
              AND to_timestamp(try_cast(client_time AS DOUBLE) / 1000.0) < TIMESTAMPTZ '2028-01-01'
            GROUP BY 1,2 ORDER BY 2
        """)

        journey_summary += records(con, f"""
            SELECT '{platform}' platform, count(*) journeys, count(DISTINCT session_id) sessions,
              count(DISTINCT device_id) devices, count(DISTINCT nullif(customer_id,'')) customers,
              avg(CASE WHEN model_eligible THEN 1.0 ELSE 0.0 END) eligible_rate,
              median(n_events_final) median_steps, quantile_cont(n_events_final,.9) p90_steps,
              avg(n_events_final) mean_steps, median(span_seconds) median_seconds,
              quantile_cont(span_seconds,.9) p90_seconds, avg(back_rate) mean_back_rate,
              avg(revisit_ratio) mean_revisit_ratio, avg(n_loop_removed) mean_loops_removed,
              min(start_ts) first_ts, max(start_ts) last_ts
            FROM read_parquet('{journey_glob}')
        """)
        boundary += records(con, f"""
            SELECT '{platform}' platform, coalesce(boundary_reason,'unknown') AS "label", count(*) journeys
            FROM read_parquet('{journey_glob}') GROUP BY 1,2 ORDER BY journeys DESC LIMIT 10
        """)
        journey_hour += records(con, f"""
            SELECT '{platform}' platform, extract(hour FROM start_ts + INTERVAL 7 HOUR)::INTEGER AS "hour", count(*) journeys
            FROM read_parquet('{journey_glob}') GROUP BY 1,2 ORDER BY 2
        """)
        journey_daily += records(con, f"""
            SELECT '{platform}' platform, cast(start_ts AS DATE) date, count(*) journeys
            FROM read_parquet('{journey_glob}')
            WHERE start_ts >= TIMESTAMPTZ '2025-01-01' AND start_ts < TIMESTAMPTZ '2028-01-01'
            GROUP BY 1,2 ORDER BY 2
        """)

        con.execute(f"""
          CREATE OR REPLACE TEMP VIEW prod_{platform} AS
          SELECT s.*, coalesce(m.cluster_name,s.cluster_name,'Chưa phân loại') journey_type,
                 coalesce(m.business_family,s.business_family,'Chưa phân loại') biz_family,
                 coalesce(m.business_submodule,'') submodule,
                 coalesce(m.business_detail,'') detail,
                 coalesce(m.naming_confidence,s.naming_confidence,'low') confidence
          FROM read_parquet('{score_glob}') s
          LEFT JOIN read_csv_auto('{map_path}', header=true, all_varchar=true) m
            ON s.cluster = try_cast(m.cluster_id AS BIGINT)
        """)
        prod_summary += records(con, f"""
          SELECT '{platform}' platform, count(*) journeys, count(DISTINCT session_id) sessions,
            count(DISTINCT device_id) devices, count(DISTINCT nullif(customer_id,'')) customers,
            count(DISTINCT cluster) clusters, count(DISTINCT journey_type) journey_types,
            avg(CASE WHEN cluster<>-1 THEN 1.0 ELSE 0 END) known_rate,
            avg(CASE WHEN coalesce(friction_flags,'')<>'' THEN 1.0 ELSE 0 END) friction_rate,
            avg(CASE WHEN severe_anomaly THEN 1.0 ELSE 0 END) severe_rate,
            avg(CASE WHEN geometric_anomaly THEN 1.0 ELSE 0 END) geometric_rate,
            avg(CASE WHEN generative_anomaly THEN 1.0 ELSE 0 END) generative_rate,
            median(n_events_final) median_steps, median(span_seconds) median_seconds,
            avg(distance_to_centroid) mean_distance, min(start_ts) first_ts, max(start_ts) last_ts
          FROM prod_{platform}
        """)
        prod_family += records(con, f"""
          SELECT '{platform}' platform, biz_family AS "label", count(*) journeys,
            count(DISTINCT nullif(customer_id,'')) customers,
            avg(CASE WHEN coalesce(friction_flags,'')<>'' THEN 1.0 ELSE 0 END) friction_rate
          FROM prod_{platform} GROUP BY 1,2 ORDER BY journeys DESC
        """)
        prod_hierarchy += records(con, f"""
          SELECT '{platform}' platform, biz_family AS "family",
            coalesce(nullif(submodule,''),'Chưa xác định submodule') submodule,
            coalesce(nullif(detail,''),'Không có detail') detail,
            count(*) journeys,
            count(DISTINCT nullif(customer_id,'')) customers,
            avg(CASE WHEN coalesce(friction_flags,'')<>'' THEN 1.0 ELSE 0 END) friction_rate
          FROM prod_{platform} GROUP BY 1,2,3,4 ORDER BY journeys DESC
        """)
        prod_type += records(con, f"""
          SELECT '{platform}' platform, journey_type AS "label", biz_family AS "family", count(*) journeys,
            count(DISTINCT nullif(customer_id,'')) customers,
            avg(CASE WHEN coalesce(friction_flags,'')<>'' THEN 1.0 ELSE 0 END) friction_rate,
            median(n_events_final) median_steps, median(span_seconds) median_seconds
          FROM prod_{platform} GROUP BY 1,2,3 ORDER BY journeys DESC LIMIT 30
        """)
        prod_daily += records(con, f"""
          SELECT '{platform}' platform, cast(start_ts AS DATE) date, count(*) journeys,
            avg(CASE WHEN cluster=-1 THEN 1.0 ELSE 0 END) novel_rate,
            avg(CASE WHEN coalesce(friction_flags,'')<>'' THEN 1.0 ELSE 0 END) friction_rate
          FROM prod_{platform}
          WHERE start_ts >= TIMESTAMPTZ '2025-01-01' AND start_ts < TIMESTAMPTZ '2028-01-01'
          GROUP BY 1,2 ORDER BY 2
        """)
        prod_friction += records(con, f"""
          WITH flags AS (
            SELECT unnest(string_split(coalesce(friction_flags,''),'|')) AS "label"
            FROM prod_{platform}
          ) SELECT '{platform}' platform, "label", count(*) journeys
            FROM flags WHERE "label"<>'' GROUP BY 1,2 ORDER BY journeys DESC LIMIT 12
        """)
        prod_next += records(con, f"""
          SELECT '{platform}' platform, coalesce(nullif(effective_next_action,''),nullif(next_action,'')) AS "label",
            count(*) journeys, avg(coalesce(effective_next_action_share,next_action_share)) confidence
          FROM prod_{platform} WHERE coalesce(nullif(effective_next_action,''),nullif(next_action,'')) IS NOT NULL
          GROUP BY 1,2 ORDER BY journeys DESC LIMIT 12
        """)

        customer_summary += records(con, f"""
          WITH c AS (
            SELECT customer_id, count(*) journeys, count(DISTINCT session_id) sessions,
              count(DISTINCT biz_family) families, count(DISTINCT journey_type) journey_types,
              avg(CASE WHEN cluster<>-1 THEN 1.0 ELSE 0 END) known_rate,
              avg(CASE WHEN coalesce(friction_flags,'')<>'' THEN 1.0 ELSE 0 END) friction_rate
            FROM prod_{platform} WHERE nullif(customer_id,'') IS NOT NULL GROUP BY 1
          ) SELECT '{platform}' platform, count(*) customers, avg(journeys) mean_journeys,
              median(journeys) median_journeys, quantile_cont(journeys,.9) p90_journeys,
              avg(CASE WHEN journeys>=10 THEN 1.0 ELSE 0 END) engaged_10plus_rate,
              avg(CASE WHEN families>=3 THEN 1.0 ELSE 0 END) multi_family_rate,
              avg(CASE WHEN friction_rate>=.3 THEN 1.0 ELSE 0 END) high_friction_customer_rate,
              avg(known_rate) mean_known_rate, avg(friction_rate) mean_friction_rate
          FROM c
        """)
        customer_bands += records(con, f"""
          WITH c AS (SELECT customer_id, count(*) journeys FROM prod_{platform}
            WHERE nullif(customer_id,'') IS NOT NULL GROUP BY 1),
          b AS (SELECT CASE WHEN journeys=1 THEN '1 journey' WHEN journeys<=3 THEN '2–3 journeys'
                    WHEN journeys<=9 THEN '4–9 journeys' WHEN journeys<=29 THEN '10–29 journeys'
                    ELSE '30+ journeys' END AS "label", count(*) customers FROM c GROUP BY 1)
          SELECT '{platform}' platform,* FROM b ORDER BY CASE "label" WHEN '1 journey' THEN 1 WHEN '2–3 journeys' THEN 2 WHEN '4–9 journeys' THEN 3 WHEN '10–29 journeys' THEN 4 ELSE 5 END
        """)
        customer_type_reach += records(con, f"""
          SELECT '{platform}' platform, journey_type AS "label", biz_family AS "family",
            count(DISTINCT customer_id) customers, count(*) journeys,
            count(*)::DOUBLE / nullif(count(DISTINCT customer_id),0) journeys_per_customer
          FROM prod_{platform} WHERE nullif(customer_id,'') IS NOT NULL
          GROUP BY 1,2,3 ORDER BY customers DESC LIMIT 15
        """)

        catalog = load_json(TRAIN / platform / f"{platform}_cluster_catalog.json")
        map_rows = mapping_rows(platform)
        total = sum(int(x["size"]) for x in catalog)
        noise = next((int(x["size"]) for x in catalog if int(x["cluster"]) == -1), 0)
        timing = load_json(TRAIN / platform / f"{platform}_training_timings.json")
        train_items = max(int(x.get("items", 0)) for x in timing.values())
        learned_summary.append({
            "platform": platform, "train_items": train_items, "clusters": len(catalog),
            "named_types": len({r["cluster_name"] for r in map_rows}),
            "families": len({r["business_family"] for r in map_rows}),
            "noise_share": noise / total if total else 0,
            "noise_journeys": noise, "catalog_journeys": total,
        })
        for row in sorted(catalog, key=lambda x: int(x["size"]), reverse=True)[:16]:
            mid = str(row["cluster"])
            mapped = next((m for m in map_rows if m["cluster_id"] == mid), None) or {}
            learned_top.append({"platform": platform, "cluster": int(row["cluster"]),
                "label": mapped.get("cluster_name", f"Cluster {mid}"), "family": mapped.get("business_family", ""),
                "journeys": int(row["size"]), "share": float(row["share"]),
                "median_steps": float(row["median_length"]), "median_seconds": float(row["median_span_s"])})
        counts: dict[str, int] = {}
        for row in map_rows:
            counts[row["naming_confidence"]] = counts.get(row["naming_confidence"], 0) + 1
        for confidence, count in counts.items():
            naming_quality.append({"platform": platform, "label": confidence, "clusters": count})
        for name, stage in timing.items():
            train_timing.append({"platform": platform, "label": name, "seconds": float(stage.get("elapsed_seconds", 0))})

    data.update({
        "raw_summary": raw_summary, "raw_top": raw_top, "raw_daily": raw_daily,
        "journey_summary": journey_summary, "boundary": boundary, "journey_hour": journey_hour, "journey_daily": journey_daily,
        "learned_summary": learned_summary, "learned_top": learned_top, "naming_quality": naming_quality, "train_timing": train_timing,
        "prod_summary": prod_summary, "prod_family": prod_family, "prod_hierarchy": prod_hierarchy, "prod_type": prod_type, "prod_daily": prod_daily,
        "prod_friction": prod_friction, "prod_next": prod_next,
        "customer_summary": customer_summary, "customer_bands": customer_bands, "customer_type_reach": customer_type_reach,
        "overall_counts": {
            "named_types": len({r["cluster_name"] for p in ("android", "ios") for r in mapping_rows(p)}),
            "families": len({r["business_family"] for p in ("android", "ios") for r in mapping_rows(p)}),
        },
    })
    return data


HTML_TEMPLATE = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HiFPT Journey Intelligence · July</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%230b1739'/%3E%3Cpath d='M7 22c4-9 7-12 18-12M8 12h6v6' fill='none' stroke='%236ee7cf' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E">
<style>
:root{--bg:#070d1d;--panel:#0e1830;--panel2:#121f3c;--ink:#f5f7fb;--muted:#9cabca;--line:#263659;--cyan:#6ee7cf;--blue:#70a7ff;--amber:#f4c76f;--red:#ff7a86;--violet:#ad8cff;--shadow:0 20px 60px #03071380}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 80% -10%,#18356a 0,transparent 35%),var(--bg);color:var(--ink);font:16px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.layout{display:grid;grid-template-columns:250px minmax(0,1fr);min-height:100vh}.side{position:sticky;top:0;height:100vh;padding:28px 20px;background:#081127e8;border-right:1px solid var(--line);backdrop-filter:blur(18px)}.brand{font-weight:850;font-size:20px;letter-spacing:-.5px}.brand b{color:var(--cyan)}.side p,.meta{color:var(--muted);font-size:13px}.nav{margin-top:28px}.nav a{display:block;color:#bdc9e3;text-decoration:none;padding:10px 12px;border-radius:9px;margin:3px 0;font-size:14px}.nav a:hover,.nav a.active{background:#182746;color:#fff}.side-foot{position:absolute;bottom:24px;color:#7181a5;font-size:12px}.main{padding:32px 38px 70px;min-width:0;max-width:1680px}.top{display:flex;justify-content:space-between;gap:24px;align-items:flex-start}.eyebrow{text-transform:uppercase;letter-spacing:1.5px;color:var(--cyan);font-weight:800;font-size:12px}.top h1{font-size:34px;letter-spacing:-1.4px;line-height:1.12;margin:7px 0}.top p{margin:0;color:var(--muted)}.control{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 12px}.control label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;font-weight:750}.control select{border:0;outline:0;background:transparent;color:#fff;font:inherit;min-width:155px}.section{scroll-margin-top:24px;margin-top:34px}.section-head{display:flex;justify-content:space-between;align-items:end;margin-bottom:13px}.section-head h2{font-size:20px;margin:0;letter-spacing:-.3px}.section-head p{margin:2px 0 0;color:var(--muted);font-size:13px}.kpis{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:12px}.card,.panel{background:linear-gradient(145deg,#111d37,#0c162d);border:1px solid var(--line);border-radius:15px;box-shadow:var(--shadow)}.kpi{padding:16px}.kpi .label{text-transform:uppercase;letter-spacing:.55px;color:var(--muted);font-size:11px;font-weight:750}.kpi .value{font-size:25px;font-weight:850;letter-spacing:-.6px;margin-top:6px}.kpi .note{font-size:12px;color:var(--muted)}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.grid3{display:grid;grid-template-columns:1.3fr 1fr .8fr;gap:14px}.panel{padding:18px;min-width:0}.panel h3{font-size:15px;margin:0 0 3px}.hint{font-size:12px;color:var(--muted);margin-bottom:14px}.bars{display:grid;gap:9px}.bar-row{display:grid;grid-template-columns:minmax(135px,1.4fr) minmax(100px,2fr) 82px;gap:9px;align-items:center}.bar-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:13px}.track{height:8px;background:#1d2b49;border-radius:20px;overflow:hidden}.fill{height:100%;border-radius:20px;background:linear-gradient(90deg,var(--blue),var(--cyan))}.fill.red{background:linear-gradient(90deg,var(--amber),var(--red))}.fill.violet{background:linear-gradient(90deg,var(--violet),var(--blue))}.bar-value{text-align:right;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}.chart{width:100%;height:270px;display:block}.axis{stroke:#263858;stroke-width:1}.line{fill:none;stroke:var(--cyan);stroke-width:2.5}.line.alt{stroke:var(--red)}.area{fill:#6ee7cf16}.tick{fill:#8292b5;font-size:10px}.hierarchy{width:100%;height:430px;display:block;background:#091329;border-radius:11px}.family-box{fill:none;stroke:#e8efff;stroke-width:2}.family-label{fill:#fff;font-size:13px;font-weight:800;pointer-events:none}.submodule-box{stroke:#091329;stroke-width:2;transition:opacity .15s}.submodule-box:hover,.detail-box:hover{opacity:.78}.submodule-label{fill:#fff;font-size:11px;font-weight:700;pointer-events:none}.detail-box{stroke:#dce8ff55;stroke-width:1;transition:opacity .15s}.detail-label{fill:#fff;font-size:9px;pointer-events:none}.heat-legend{display:flex;align-items:center;justify-content:flex-end;gap:7px;margin-top:9px;color:var(--muted);font-size:11px}.heat-gradient{width:120px;height:8px;border-radius:8px;background:linear-gradient(90deg,#315181,#f4c76f,#ff6574)}.table-wrap{overflow:auto;max-height:560px;border:1px solid var(--line);border-radius:11px}table{width:100%;border-collapse:collapse;font-size:13px}th{position:sticky;top:0;background:#172541;color:#aebbd4;text-transform:uppercase;font-size:10px;letter-spacing:.45px;text-align:left;z-index:1}th,td{padding:9px 10px;border-bottom:1px solid #243351}tbody tr:hover{background:#162441}.pill{display:inline-block;padding:3px 7px;border-radius:20px;background:#173f47;color:#86f1d9;font-size:11px;font-weight:750}.pill.bad{background:#4a2735;color:#ff9eaa}.callout{border-left:3px solid var(--cyan);padding:12px 14px;background:#101f39;border-radius:0 11px 11px 0;color:#dce5f6}.callout strong{color:#fff}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:20px}.flow div{padding:12px;background:#0d1930;border:1px solid var(--line);border-radius:12px;text-align:center;font-size:13px}.flow b{display:block;color:var(--cyan);font-size:12px}.footer{color:#7181a5;font-size:12px;margin-top:40px;text-align:center}.mobile-menu{display:none}.empty{padding:25px;color:var(--muted);text-align:center}@media(max-width:1180px){.kpis{grid-template-columns:repeat(3,1fr)}.grid3{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr}}@media(max-width:850px){.layout{grid-template-columns:1fr}.side{display:none}.main{padding:20px}.top{display:block}.control{margin-top:16px;width:max-content}.grid2,.grid3{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.kpis{grid-template-columns:1fr}.top h1{font-size:28px}.bar-row{grid-template-columns:110px 1fr 68px}.hierarchy{height:360px}}
</style></head><body><div class="layout"><aside class="side"><div class="brand">HiFPT <b>Journey</b></div><p>July · static intelligence report</p><nav class="nav"><a href="#overview">Tổng quan</a><a href="#raw">01 · Raw events</a><a href="#journeys">02 · Journey EDA</a><a href="#learned">03 · Journey đã học</a><a href="#customers">04 · Customer insights</a><a href="#production">05 · Production</a></nav><div class="side-foot">Self-contained HTML<br>Không tải dữ liệu bên ngoài</div></aside><main class="main">
<header class="top"><div><div class="eyebrow">Unsupervised journey clustering</div><h1>Journey Intelligence<br>July data bundle</h1><p>Từ clickstream thô đến customer insights và kết quả production.</p></div><div class="control"><label>Platform</label><select id="platform"><option value="all">Android + iOS</option><option value="android">Android</option><option value="ios">iOS</option></select></div></header>
<section id="overview" class="section"><div class="section-head"><div><h2>Tổng quan dữ liệu</h2><p>Quy mô end-to-end theo phạm vi đang chọn.</p></div></div><div id="overviewKpis" class="kpis"></div><div class="flow"><div><b>01 · Thu thập</b>Raw clickstream</div><div><b>02 · Chuẩn hoá</b>Canonical events</div><div><b>03 · Phân đoạn</b>Journeys theo intent</div><div><b>04 · Học không giám sát</b>TF-IDF · SVD · HDBSCAN</div><div><b>05 · Production</b>Gán type · friction · next action</div></div></section>
<section id="raw" class="section"><div class="section-head"><div><h2>01 · EDA dữ liệu raw</h2><p>Event volume, coverage và những điểm chạm phổ biến.</p></div></div><div id="rawKpis" class="kpis"></div><div class="grid2" style="margin-top:14px"><div class="panel"><h3>Top màn hình</h3><div class="hint">View event theo segmentation_name</div><div id="rawViews" class="bars"></div></div><div class="panel"><h3>Top hành động</h3><div class="hint">Action event theo segmentation_name</div><div id="rawActions" class="bars"></div></div></div><div class="panel" style="margin-top:14px"><h3>Event volume theo ngày</h3><div id="rawTrend"></div></div></section>
<section id="journeys" class="section"><div class="section-head"><div><h2>02 · EDA journey đã phân đoạn</h2><p>Độ dài, thời gian, eligibility và nhịp sử dụng theo giờ Việt Nam.</p></div></div><div id="journeyKpis" class="kpis"></div><div class="grid2" style="margin-top:14px"><div class="panel"><h3>Journey theo giờ</h3><div class="hint">Giờ địa phương UTC+7</div><div id="hourBars" class="bars"></div></div><div class="panel"><h3>Lý do kết thúc journey</h3><div class="hint">Boundary reason từ pipeline segmentation</div><div id="boundaryBars" class="bars"></div></div></div></section>
<section id="learned" class="section"><div class="section-head"><div><h2>03 · Journey types mô hình học được</h2><p>Catalogue HDBSCAN, chất lượng naming và quy mô cụm.</p></div></div><div id="learnedKpis" class="kpis"></div><div class="grid2" style="margin-top:14px"><div class="panel"><h3>Cluster lớn nhất</h3><div class="hint">Theo quy mô catalogue train</div><div id="learnedBars" class="bars"></div></div><div class="panel"><h3>Độ tin cậy khi đặt tên</h3><div class="hint">Số cluster theo naming_confidence</div><div id="confidenceBars" class="bars"></div><h3 style="margin-top:24px">Thời gian xử lý</h3><div id="timingBars" class="bars"></div></div></div></section>
<section id="customers" class="section"><div class="section-head"><div><h2>04 · Customer insights</h2><p>Mức độ quay lại, độ rộng nhu cầu và nhóm có friction cao.</p></div></div><div id="customerKpis" class="kpis"></div><div class="grid2" style="margin-top:14px"><div class="panel"><h3>Phân bố số journey / customer</h3><div id="customerBands" class="bars"></div></div><div class="panel"><h3>Journey type có customer reach lớn</h3><div id="reachBars" class="bars"></div></div></div><div id="customerInsight" class="callout" style="margin-top:14px"></div></section>
<section id="production" class="section"><div class="section-head"><div><h2>05 · Kết quả production</h2><p>Named inference, novel behavior, friction và next action.</p></div></div><div id="prodKpis" class="kpis"></div><div class="grid2" style="margin-top:14px"><div class="panel"><h3>Journey production theo ngày</h3><div id="prodTrend"></div></div><div class="panel"><h3>Coverage nghiệp vụ</h3><div class="hint">Treemap lồng nhau: Family → Submodule → Detail. Diện tích biểu diễn số journey; màu biểu diễn friction rate.</div><div id="familyTreemap"></div><div class="heat-legend"><span>Friction thấp</span><span class="heat-gradient"></span><span>Friction cao</span></div></div></div><div class="grid3" style="margin-top:14px"><div class="panel"><h3>Journey type phổ biến</h3><div id="typeBars" class="bars"></div></div><div class="panel"><h3>Friction signals</h3><div id="frictionBars" class="bars"></div></div><div class="panel"><h3>Next action</h3><div id="nextBars" class="bars"></div></div></div><div class="panel" style="margin-top:14px"><h3>Catalogue production</h3><div class="hint">Top journey type, customer reach và friction rate.</div><div class="table-wrap"><table><thead><tr><th>Journey type</th><th>Family</th><th>Journeys</th><th>Customers</th><th>Friction</th><th>Median steps</th><th>Median duration</th></tr></thead><tbody id="typeTable"></tbody></table></div></div></section>
<div class="footer">Nguồn: July raw events, segmented journeys, July PCA48/SVD48/HDBSCAN run và named production scores · dữ liệu đã tổng hợp sẵn trong file.</div></main></div>
<script id="report-data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('report-data').textContent);let platform='all';const N=v=>Number(v||0),I=v=>N(v).toLocaleString('en-US'),P=v=>(N(v)*100).toFixed(N(v)<.1?1:0)+'%',S=v=>N(v)>=60?(N(v)/60).toFixed(1)+' phút':N(v).toFixed(0)+' giây',E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const rows=k=>platform==='all'?D[k]:D[k].filter(r=>r.platform===platform);const one=k=>{let a=rows(k);if(platform!=='all')return a[0]||{};return a.reduce((x,r)=>{for(const [k,v] of Object.entries(r))if(k!=='platform'&&typeof v==='number')x[k]=(x[k]||0)+v;return x},{platform:'all'})};
function cards(id,items){document.getElementById(id).innerHTML=items.map(x=>`<div class="card kpi"><div class="label">${E(x[0])}</div><div class="value">${x[1]}</div><div class="note">${E(x[2]||'')}</div></div>`).join('')}
function group(data,label='label',value='journeys'){let g={};for(const r of data){let k=r[label]??'Khác';(g[k]??={label:k,value:0,rateN:0,rateD:0});g[k].value+=N(r[value]);g[k].rateN+=N(r.friction_rate)*N(r[value]);g[k].rateD+=N(r[value])}return Object.values(g)}
function bars(id,data,{limit=12,fmt=I,color='',suffix=''}={}){data=[...data].sort((a,b)=>b.value-a.value).slice(0,limit);let m=Math.max(1,...data.map(x=>N(x.value)));document.getElementById(id).innerHTML=data.length?data.map(x=>`<div class="bar-row"><div class="bar-label" title="${E(x.label)}">${E(x.label)}</div><div class="track"><div class="fill ${color}" style="width:${100*N(x.value)/m}%"></div></div><div class="bar-value">${fmt(x.value)}${suffix}</div></div>`).join(''):'<div class="empty">Không có dữ liệu</div>'}
function line(id,data,key='events'){let g={};for(const r of data){let d=String(r.date).slice(0,10);g[d]=(g[d]||0)+N(r[key])}let pts=Object.entries(g).sort().map(([x,v])=>({x,v})),host=document.getElementById(id),W=760,H=270,p={l:55,r:12,t:12,b:32};if(!pts.length){host.innerHTML='<div class="empty">Không có dữ liệu</div>';return}let max=Math.max(...pts.map(x=>x.v),1),sx=i=>p.l+i*(W-p.l-p.r)/Math.max(1,pts.length-1),sy=v=>H-p.b-v*(H-p.t-p.b)/max,path=pts.map((x,i)=>`${i?'L':'M'}${sx(i).toFixed(1)},${sy(x.v).toFixed(1)}`).join(' '),axis='';for(let i=0;i<=4;i++){let y=p.t+i*(H-p.t-p.b)/4,v=max*(1-i/4);axis+=`<line class="axis" x1="${p.l}" x2="${W-p.r}" y1="${y}" y2="${y}"/><text class="tick" x="${p.l-7}" y="${y+3}" text-anchor="end">${I(v)}</text>`}for(let i=0;i<pts.length;i+=Math.max(1,Math.ceil(pts.length/6)))axis+=`<text class="tick" x="${sx(i)}" y="${H-10}" text-anchor="middle">${pts[i].x.slice(5)}</text>`;host.innerHTML=`<svg viewBox="0 0 ${W} ${H}" class="chart">${axis}<path class="area" d="M${sx(0)},${H-p.b} ${path} L${sx(pts.length-1)},${H-p.b}Z"/><path class="line" d="${path}"/></svg>`}
function treeRects(items,x,y,w,h){if(!items.length)return[];if(items.length===1)return[{...items[0],x,y,w,h}];items=[...items].sort((a,b)=>b.value-a.value);let total=items.reduce((s,d)=>s+d.value,0),acc=0,cut=1,best=Infinity;for(let i=1;i<items.length;i++){acc+=items[i-1].value;let delta=Math.abs(total/2-acc);if(delta<best){best=delta;cut=i}}let a=items.slice(0,cut),b=items.slice(cut),ratio=a.reduce((s,d)=>s+d.value,0)/Math.max(1,total);return w>=h?[...treeRects(a,x,y,w*ratio,h),...treeRects(b,x+w*ratio,y,w*(1-ratio),h)]:[...treeRects(a,x,y,w,h*ratio),...treeRects(b,x,y+h*ratio,w,h*(1-ratio))]}
function heatColor(rate,level=0){let t=Math.max(0,Math.min(1,N(rate))),stops=[[49,81,129],[244,199,111],[255,101,116]],a,b,u;if(t<.5){a=stops[0];b=stops[1];u=t*2}else{a=stops[1];b=stops[2];u=(t-.5)*2}let rgb=a.map((v,i)=>Math.round(v+(b[i]-v)*u));if(level)rgb=rgb.map(v=>Math.min(255,Math.round(v+(255-v)*level)));return`rgb(${rgb.join(',')})`}
function familyTreemap(){let source=rows('prod_hierarchy'),families={};for(const r of source){let fk=r.family||'Chưa phân loại',sk=r.submodule||'Chưa xác định submodule',dk=r.detail||'Không có detail',f=families[fk]??={label:fk,value:0,rateN:0,subs:{}},s=f.subs[sk]??={label:sk,value:0,rateN:0,details:{}};f.value+=N(r.journeys);f.rateN+=N(r.friction_rate)*N(r.journeys);s.value+=N(r.journeys);s.rateN+=N(r.friction_rate)*N(r.journeys);let d=s.details[dk]??={label:dk,value:0,rateN:0};d.value+=N(r.journeys);d.rateN+=N(r.friction_rate)*N(r.journeys);s.details[dk]=d;f.subs[sk]=s;families[fk]=f}let fs=Object.values(families);for(const f of fs){f.rate=f.rateN/Math.max(1,f.value);f.subs=Object.values(f.subs);for(const s of f.subs){s.rate=s.rateN/Math.max(1,s.value);s.details=Object.values(s.details).map(d=>({...d,rate:d.rateN/Math.max(1,d.value)}))}}let W=820,H=430,fr=treeRects(fs,0,0,W,H),svg=`<svg class="hierarchy" viewBox="0 0 ${W} ${H}" role="img" aria-label="Treemap coverage theo business family, submodule và detail">`;for(const f of fr){let titleH=f.h>48?23:0;svg+=`<rect class="family-box" x="${f.x+1}" y="${f.y+1}" width="${Math.max(0,f.w-2)}" height="${Math.max(0,f.h-2)}"/><title>${E(f.label)} · ${I(f.value)} journeys · ${P(f.rate)} friction</title>`;if(titleH&&f.w>75)svg+=`<text class="family-label" x="${f.x+7}" y="${f.y+16}">${E(f.label.length>Math.floor(f.w/8)?f.label.slice(0,Math.max(7,Math.floor(f.w/8)-1))+'…':f.label)}</text>`;let sr=treeRects(f.subs,f.x+3,f.y+titleH+2,Math.max(0,f.w-6),Math.max(0,f.h-titleH-5));for(const s of sr){svg+=`<rect class="submodule-box" x="${s.x}" y="${s.y}" width="${Math.max(0,s.w)}" height="${Math.max(0,s.h)}" fill="${heatColor(s.rate)}"><title>${E(f.label)} → ${E(s.label)} · ${I(s.value)} journeys · ${P(s.rate)} friction</title></rect>`;if(s.w>90&&s.h>35)svg+=`<text class="submodule-label" x="${s.x+5}" y="${s.y+14}">${E(s.label.length>Math.floor(s.w/7)?s.label.slice(0,Math.max(8,Math.floor(s.w/7)-1))+'…':s.label)}</text>`;let details=s.details.filter(d=>d.label!=='Không có detail');if(details.length){let dr=treeRects(details,s.x+3,s.y+(s.h>35?19:3),Math.max(0,s.w-6),Math.max(0,s.h-(s.h>35?22:6)));for(const d of dr){svg+=`<rect class="detail-box" x="${d.x}" y="${d.y}" width="${Math.max(0,d.w)}" height="${Math.max(0,d.h)}" fill="${heatColor(d.rate,.10)}"><title>${E(f.label)} → ${E(s.label)} → ${E(d.label)} · ${I(d.value)} journeys · ${P(d.rate)} friction</title></rect>`;if(d.w>72&&d.h>20)svg+=`<text class="detail-label" x="${d.x+4}" y="${d.y+12}">${E(d.label.length>Math.floor(d.w/6)?d.label.slice(0,Math.max(7,Math.floor(d.w/6)-1))+'…':d.label)}</text>`}}}}svg+='</svg>';document.getElementById('familyTreemap').innerHTML=svg}
function weighted(data,field,weight='journeys'){let n=0,d=0;for(const r of data){n+=N(r[field])*N(r[weight]);d+=N(r[weight])}return n/Math.max(1,d)}
function render(){let raw=one('raw_summary'),jour=one('journey_summary'),learn=one('learned_summary'),prod=one('prod_summary'),cust=one('customer_summary');if(platform==='all'){for(const x of [raw,jour,learn,prod,cust])x.platform='all';prod.known_rate=weighted(D.prod_summary,'known_rate');prod.friction_rate=weighted(D.prod_summary,'friction_rate');prod.severe_rate=weighted(D.prod_summary,'severe_rate');prod.median_steps=weighted(D.prod_summary,'median_steps');prod.median_seconds=weighted(D.prod_summary,'median_seconds');prod.journey_types=D.overall_counts.named_types;jour.eligible_rate=weighted(D.journey_summary,'eligible_rate');jour.median_steps=weighted(D.journey_summary,'median_steps');jour.p90_steps=weighted(D.journey_summary,'p90_steps');jour.median_seconds=weighted(D.journey_summary,'median_seconds');jour.p90_seconds=weighted(D.journey_summary,'p90_seconds');jour.mean_back_rate=weighted(D.journey_summary,'mean_back_rate');cust.median_journeys=weighted(D.customer_summary,'median_journeys','customers');cust.p90_journeys=weighted(D.customer_summary,'p90_journeys','customers');cust.engaged_10plus_rate=weighted(D.customer_summary,'engaged_10plus_rate','customers');cust.multi_family_rate=weighted(D.customer_summary,'multi_family_rate','customers');cust.high_friction_customer_rate=weighted(D.customer_summary,'high_friction_customer_rate','customers');learn.named_types=D.overall_counts.named_types;learn.families=D.overall_counts.families;learn.noise_share=D.learned_summary.reduce((s,r)=>s+N(r.noise_journeys),0)/Math.max(1,D.learned_summary.reduce((s,r)=>s+N(r.catalog_journeys),0))}
cards('overviewKpis',[['Raw events',I(raw.events),'clickstream input'],['Segmented journeys',I(jour.journeys),P(jour.eligible_rate)+' model eligible'],['Production journeys',I(prod.journeys),I(prod.sessions)+' sessions'],['Customers',I(prod.customers),'distinct theo platform'],['Learned clusters',I(learn.clusters),I(learn.named_types)+' journey types'],['Known rate',P(prod.known_rate),'ngoài cluster -1']]);
cards('rawKpis',[['Events',I(raw.events),'toàn bộ event'],['Sessions',I(raw.sessions),(N(raw.events)/Math.max(1,N(raw.sessions))).toFixed(1)+' event/session'],['Devices',I(raw.devices),'distinct device'],['Customers',I(raw.customers),'distinct customer'],['Action share',P(raw.action_share||N(raw.action_events)/Math.max(1,N(raw.events))),I(raw.action_events)+' actions'],['Timestamp nghi vấn',I(raw.suspect_timestamps),P(N(raw.suspect_timestamps)/Math.max(1,N(raw.events)))+' tổng event']]);bars('rawViews',group(rows('raw_top').filter(x=>x.event_kind==='view'),'label','events'));bars('rawActions',group(rows('raw_top').filter(x=>x.event_kind==='action'),'label','events'));line('rawTrend',rows('raw_daily'));
cards('journeyKpis',[['Journeys',I(jour.journeys),I(jour.sessions)+' sessions'],['Customers',I(jour.customers),'distinct customer'],['Model eligible',P(jour.eligible_rate),'đủ điều kiện học'],['Median steps',N(jour.median_steps).toFixed(0),N(jour.p90_steps).toFixed(0)+' tại P90'],['Median duration',S(jour.median_seconds),S(jour.p90_seconds)+' tại P90'],['Mean back rate',P(jour.mean_back_rate),'tín hiệu quay lại']]);bars('hourBars',group(rows('journey_hour'),'hour'),{limit:24});bars('boundaryBars',group(rows('boundary')),{limit:10,color:'violet'});
cards('learnedKpis',[['Train items',I(learn.train_items),'cao nhất theo stage'],['Clusters',I(learn.clusters),'bao gồm cluster -1'],['Named types',I(learn.named_types),'tên nghiệp vụ distinct'],['Families',I(learn.families),'nhóm nghiệp vụ'],['Noise share',P(learn.noise_share),'cluster -1 trong train'],['Catalogue journeys',I(learn.catalog_journeys),'quy mô catalogue']]);bars('learnedBars',group(rows('learned_top')),{limit:14});bars('confidenceBars',group(rows('naming_quality'),'label','clusters'),{limit:6,color:'violet'});bars('timingBars',group(rows('train_timing'),'label','seconds'),{limit:10,fmt:v=>N(v).toFixed(0)+'s',color:'violet'});
cards('customerKpis',[['Customers',I(cust.customers),'có customer_id'],['Median journeys',N(cust.median_journeys).toFixed(1),'mỗi customer'],['P90 journeys',N(cust.p90_journeys).toFixed(0),'mỗi customer'],['10+ journeys',P(cust.engaged_10plus_rate),'customer quay lại nhiều'],['3+ families',P(cust.multi_family_rate),'nhu cầu đa nghiệp vụ'],['High friction',P(cust.high_friction_customer_rate),'friction rate ≥ 30%']]);bars('customerBands',group(rows('customer_bands'),'label','customers'),{limit:8,color:'violet'});bars('reachBars',group(rows('customer_type_reach'),'label','customers'),{limit:12});let reach=group(rows('customer_type_reach'),'label','customers').sort((a,b)=>b.value-a.value)[0];document.getElementById('customerInsight').innerHTML=reach?`<strong>${E(reach.label)}</strong> có customer reach lớn nhất trong nhóm top, chạm khoảng <strong>${I(reach.value)}</strong> customer. <strong>${P(cust.engaged_10plus_rate)}</strong> customer có từ 10 journey trở lên và <strong>${P(cust.multi_family_rate)}</strong> đi qua ít nhất 3 business family.`:'Không có dữ liệu customer.';
cards('prodKpis',[['Journeys',I(prod.journeys),I(prod.sessions)+' sessions'],['Customers',I(prod.customers),'distinct customer'],['Clusters',I(prod.clusters),I(prod.journey_types)+' named types'],['Known journey',P(prod.known_rate),'không thuộc noise'],['Có friction',P(prod.friction_rate),'ít nhất một flag'],['Severe anomaly',P(prod.severe_rate),'cần theo dõi']]);line('prodTrend',rows('prod_daily'),'journeys');familyTreemap();bars('typeBars',group(rows('prod_type')),{limit:14});bars('frictionBars',group(rows('prod_friction')),{limit:12,color:'red'});bars('nextBars',group(rows('prod_next')),{limit:12,color:'violet'});let table=rows('prod_type');if(platform==='all'){let g={};for(const r of table){let x=g[r.label]??={...r,journeys:0,customers:0,frN:0,frD:0};x.journeys+=N(r.journeys);x.customers+=N(r.customers);x.frN+=N(r.friction_rate)*N(r.journeys);x.frD+=N(r.journeys);g[r.label]=x}table=Object.values(g).map(x=>({...x,friction_rate:x.frN/Math.max(1,x.frD)}))}table.sort((a,b)=>b.journeys-a.journeys);document.getElementById('typeTable').innerHTML=table.map(r=>`<tr><td><strong>${E(r.label)}</strong></td><td>${E(r.family)}</td><td>${I(r.journeys)}</td><td>${I(r.customers)}</td><td><span class="pill ${N(r.friction_rate)>.3?'bad':''}">${P(r.friction_rate)}</span></td><td>${N(r.median_steps).toFixed(0)}</td><td>${S(r.median_seconds)}</td></tr>`).join('')}
document.getElementById('platform').onchange=e=>{platform=e.target.value;render()};render();
</script></body></html>'''


def main() -> None:
    data = build_data()
    payload = json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "bytes": OUT.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
