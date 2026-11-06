"""Daily job: Kafka events of one day -> journeys scored by the current named model
-> rows in the MariaDB journey_summary schema.

Steps (one batch_date, in OUTPUT_TIMEZONE, default Asia/Ho_Chi_Minh):

1. events   Export [date 00:00, date+1 00:00) from Kafka with event-consumer-fastapi
            (`python -m app.export`, no consumer group, no commit), or read the files already
            in EVENTS_DIR/date=YYYY-MM-DD/ (EVENTS_SOURCE=folder): events.csv of the background
            consumer, or one <topic>.csv per topic fetched by the service POST /pipeline/runs.
            Duplicate record_id are dropped.
2. infer    `journey-infer` per platform with the frozen model (no retraining).
3. summary  Fill taxonomy_id from <NAMES_RUN>/<platform>/<platform>_named_clusters.csv
            (cluster -1 -> NULL), convert to the journey_summary columns: sequence joined
            by ",", no next_action/next_action_share, batch_date added, times in UTC.
4. load     Optional (LOAD_MARIADB=1): replace the rows of (batch_date, model_version)
            in MariaDB, so a rerun of the same day never duplicates.

Output folder: <OUTPUT_ROOT>/<model_version>/<date>/
    events.csv, <platform>_scores.csv, journey_summary.csv, manifest.json
It holds customer_id and device_id: keep it on the machine that runs the job.

Examples:

    python scripts/daily_journey_job.py                      # yesterday, settings from env
    python scripts/daily_journey_job.py --config scripts/daily_job.env --date 2026-03-26
    python scripts/daily_journey_job.py --date 2026-03-26 --skip-export   # reuse events.csv

    # one step at a time, as the combined service (deploy/combined) runs them;
    # a step with nothing to do (no events for a platform, LOAD_MARIADB=0) exits with 99
    python scripts/daily_journey_job.py --date 2026-03-26 --step events
    python scripts/daily_journey_job.py --date 2026-03-26 --step infer --platform android
    python scripts/daily_journey_job.py --date 2026-03-26 --step summary
    python scripts/daily_journey_job.py --date 2026-03-26 --step load

Settings come from environment variables, optionally read from --config (KEY=VALUE lines;
real environment variables win). See scripts/daily_job.env.example for every key.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "PYTHON": sys.executable,
    "CONSUMER_DIR": str(ROOT.parent / "event-consumer-fastapi"),
    "CONSUMER_PYTHON": "",          # default: <CONSUMER_DIR>/.venv/bin/python if present, else PYTHON
    "CONSUMER_ENV_FILE": "",        # default: <CONSUMER_DIR>/.env
    "EVENTS_SOURCE": "export",      # export | folder
    "EVENTS_DIR": "",               # folder mode: the consumer OUTPUT_DIR (date=YYYY-MM-DD/events.csv)
    "MODEL_RUN": "output/partitioned_runs/678/678_20260929_090904",
    "NAMES_RUN": "output/scores/678/678_20260929_090904",
    "MODEL_VERSION": "",            # default: folder name of MODEL_RUN
    "SITEMAP": "taxonomy/hifpt_sitemap_taxonomy.csv",
    "PLATFORMS": "android ios",
    "OUTPUT_TIMEZONE": "Asia/Ho_Chi_Minh",
    "OUTPUT_ROOT": "output/daily",
    "LOAD_MARIADB": "0",
    "MARIADB_HOST": "",
    "MARIADB_PORT": "3306",
    "MARIADB_USER": "",
    "MARIADB_PASSWORD": "",
    "MARIADB_DATABASE": "",
    "MARIADB_TABLE": "journey_summary",
    "MARIADB_BATCH_SIZE": "5000",
}

# journey_summary columns written by the job (id and created_at are filled by MariaDB).
SUMMARY_COLUMNS = (
    "batch_date", "model_version", "platform", "cluster_id", "taxonomy_id", "journey_id",
    "session_id", "device_id", "customer_id", "start_ts", "end_ts", "boundary_reason",
    "n_events_raw", "n_events_final", "n_loop_removed", "action_ratio", "back_rate",
    "revisit_ratio", "median_gap_s", "p90_gap_s", "max_gap_s", "entry_token", "exit_token",
    "sequence", "distance_to_centroid", "markov_logprob", "geometric_anomaly",
    "generative_anomaly", "severe_anomaly", "friction_flags", "scored_at",
)
BOOL_COLUMNS = ("geometric_anomaly", "generative_anomaly", "severe_anomaly")
ID_COLUMNS = ("journey_id", "session_id", "device_id", "customer_id")
EVENT_ID_COLUMNS = ("record_id", "_id", "device_id", "customer_id", "session_id")


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def read_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_settings(config: Path | None) -> dict[str, str]:
    settings = dict(DEFAULTS)
    if config:
        settings.update(read_env_file(config))
    settings.update({key: os.environ[key] for key in DEFAULTS if key in os.environ})
    return settings


def resolve(value: str, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


# --------------------------------------------------------------------------- 1. events

def export_events(cfg: dict[str, str], batch_date: date, target: Path) -> dict:
    """Run event-consumer-fastapi's group-less export for one day."""
    consumer_dir = resolve(cfg["CONSUMER_DIR"])
    venv_python = consumer_dir / ".venv" / "bin" / "python"
    python = cfg["CONSUMER_PYTHON"] or (str(venv_python) if venv_python.exists() else cfg["PYTHON"])
    env = dict(os.environ)
    env_file = resolve(cfg["CONSUMER_ENV_FILE"], consumer_dir) if cfg["CONSUMER_ENV_FILE"] else consumer_dir / ".env"
    if env_file.exists():
        # Variables from the env file win over the consumer's own .env (pydantic-settings).
        env.update(read_env_file(env_file))
    env["OUTPUT_TIMEZONE"] = cfg["OUTPUT_TIMEZONE"]
    raw = target.with_name(f".{target.name}.export")
    command = [
        python, "-m", "app.export",
        "--start", batch_date.isoformat(),
        "--end", (batch_date + timedelta(days=1)).isoformat(),
        "--out", str(raw),
    ]
    log(f"[1/4] export Kafka {batch_date} -> {target} ({env.get('KAFKA_TOPIC', '?')} @ "
        f"{env.get('KAFKA_BOOTSTRAP_SERVERS', '?')})")
    subprocess.run(command, cwd=consumer_dir, env=env, check=True)
    return dedupe_events(raw, target)


def folder_events(cfg: dict[str, str], batch_date: date, target: Path) -> dict:
    """Use the CSVs already stored for that day: events.csv of the background consumer, or one
    <topic>.csv per topic fetched by the service (POST /pipeline/runs)."""
    events_dir = cfg["EVENTS_DIR"] or str(resolve(cfg["CONSUMER_DIR"]) / "data" / "events")
    folder = resolve(events_dir) / f"date={batch_date.isoformat()}"
    sources = sorted(folder.glob("*.csv"))
    if not sources:
        raise SystemExit(f"no event files for {batch_date} in {folder}")
    log(f"[1/4] read {', '.join(path.name for path in sources)} from {folder}")
    return dedupe_events(sources, target, keep_source=True)


def read_events(source: Path) -> pd.DataFrame:
    header = pd.read_csv(source, nrows=0).columns
    dtype = {column: "string" for column in EVENT_ID_COLUMNS if column in header}
    return pd.read_csv(source, dtype=dtype, low_memory=False)


def dedupe_events(source: Path | list[Path], target: Path, keep_source: bool = False) -> dict:
    sources = source if isinstance(source, list) else [source]
    frames = [read_events(path) for path in sources]
    if len(frames) > 1:
        # record_id is partition-offset-index, unique inside one topic only
        for path, frame in zip(sources, frames):
            if "record_id" in frame.columns:
                frame["record_id"] = path.stem + ":" + frame["record_id"]
    events = pd.concat(frames, ignore_index=True)
    rows = len(events)
    key = "record_id" if "record_id" in events.columns else ("_id" if "_id" in events.columns else None)
    if key:
        events = events.drop_duplicates(subset=[key])
    if {"kafka_timestamp", "kafka_partition", "kafka_offset", "record_id"} <= set(events.columns):
        # The export reads partition after partition. The model breaks client_time ties by row
        # order, so put rows back in send order: message time, then position inside the message.
        order = events[["kafka_timestamp", "kafka_partition", "kafka_offset"]].apply(pd.to_numeric, errors="coerce")
        order["event_index"] = pd.to_numeric(events["record_id"].str.rsplit("-", n=1).str[-1], errors="coerce")
        events = events.loc[order.sort_values(list(order.columns), kind="mergesort").index]
    tmp = target.with_name(f".{target.name}.tmp")
    events.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(target)
    if not keep_source:
        for path in sources:
            path.unlink(missing_ok=True)
    missing_session = int(events["session_id"].isna().sum()) if "session_id" in events.columns else rows
    stats = {"events_read": rows, "events": len(events), "duplicates_dropped": rows - len(events),
             "events_without_session_id": missing_session}
    if "platform" in events.columns:
        stats["events_by_platform"] = (
            events["platform"].astype("string").str.strip().str.lower().value_counts().to_dict()
        )
    log(f"      {len(events):,} events ({rows - len(events):,} duplicates dropped)")
    if missing_session:
        log(f"      WARNING: {missing_session:,} events have no session_id; the model cannot "
            "build journeys from them")
    return stats


# --------------------------------------------------------------------------- 2. infer

def infer(cfg: dict[str, str], events: Path, platform: str, model_version: str, work: Path) -> Path:
    target = work / f"{platform}_scores.csv"
    target.unlink(missing_ok=True)
    target.with_suffix(".manifest.json").unlink(missing_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''}"
    command = [
        cfg["PYTHON"], "-m", "journey_clustering.cli.infer",
        "--input", str(events),
        "--platform", platform,
        "--model-run", str(resolve(cfg["MODEL_RUN"]) / platform),
        "--model-version", model_version,
        "--output-root", str(work / ".infer"),
        "--output", str(target),
    ]
    log(f"[2/4] infer {platform} with model {model_version}")
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return target


# --------------------------------------------------------------------------- 3. summary

def load_cluster_naming():
    """Reuse the sitemap and name checks of scripts/map_named_clusters.py."""
    spec = importlib.util.spec_from_file_location("map_named_clusters", ROOT / "scripts" / "map_named_clusters.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utc_text(values: pd.Series, digits: int) -> pd.Series:
    times = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    text = times.dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return text.str[: 20 + digits] if digits else text.str[:19]


def clean_id(values: pd.Series) -> pd.Series:
    out = values.astype("string").str.strip()
    return out.str.replace(r"\.0$", "", regex=True).replace({"": pd.NA, "nan": pd.NA})


def to_summary(scores: pd.DataFrame, *, platform: str, batch_date: date, model_version: str,
               names: dict[int, dict[str, str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=scores.index)
    clusters = scores["cluster"].astype(int)
    out["batch_date"] = batch_date.isoformat()
    out["model_version"] = model_version
    out["platform"] = platform
    out["cluster_id"] = clusters
    # Cluster -1 and clusters missing from the reviewed names have no taxonomy id.
    out["taxonomy_id"] = [
        (cluster != -1 and names.get(cluster, {}).get("taxonomy_id")) or pd.NA for cluster in clusters
    ]
    # Platform first so android and ios journeys never share an id in one batch.
    local_id = scores["journey_id"].astype("string").str.split("::").str[-1]
    out["journey_id"] = f"{batch_date.isoformat()}_{platform}::" + local_id
    for column in ("session_id", "device_id", "customer_id"):
        out[column] = clean_id(scores[column])
    out["start_ts"] = utc_text(scores["start_ts"], 3)
    out["end_ts"] = utc_text(scores["end_ts"], 3)
    for column in (
        "boundary_reason", "n_events_raw", "n_events_final", "n_loop_removed", "action_ratio",
        "back_rate", "revisit_ratio", "median_gap_s", "p90_gap_s", "max_gap_s", "entry_token",
        "exit_token", "distance_to_centroid", "markov_logprob",
    ):
        out[column] = scores[column]
    out["sequence"] = scores["sequence"].fillna("").astype(str).str.split(" -> ").str.join(",")
    for column in BOOL_COLUMNS:
        out[column] = scores[column].astype(str).str.lower().isin({"true", "1"}).astype(int)
    # Behavioural flags only: unknown_archetype / improbable_transitions are the anomaly columns.
    flags = scores["behavioral_friction_flags"] if "behavioral_friction_flags" in scores else scores["friction_flags"]
    out["friction_flags"] = flags.astype("string").replace({"": pd.NA})
    out["scored_at"] = utc_text(scores["scored_at"], 6)
    return out[list(SUMMARY_COLUMNS)]


def build_summary(cfg: dict[str, str], score_files: dict[str, Path], batch_date: date,
                  model_version: str, target: Path) -> dict:
    naming = load_cluster_naming()
    sitemap = naming.load_sitemap(resolve(cfg["SITEMAP"]))
    frames, stats = [], {}
    for platform, path in score_files.items():
        names_path = resolve(cfg["NAMES_RUN"]) / platform / f"{platform}_named_clusters.csv"
        names = naming.load_names(names_path, sitemap)
        try:
            scores = pd.read_csv(path, dtype={c: "string" for c in ID_COLUMNS}, low_memory=False)
        except pd.errors.EmptyDataError:
            scores = pd.DataFrame()
        if "cluster" not in scores.columns:  # events but no journey left after cleaning
            stats[platform] = {"journeys": 0}
            continue
        summary = to_summary(scores, platform=platform, batch_date=batch_date,
                             model_version=model_version, names=names)
        unnamed = int((~summary["cluster_id"].isin(names)).sum())
        stats[platform] = {
            "journeys": len(summary),
            "cluster_minus_1": int(summary["cluster_id"].eq(-1).sum()),
            "without_taxonomy_id": int(summary["taxonomy_id"].isna().sum()),
            "in_clusters_missing_from_names": unnamed,
        }
        frames.append(summary)
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SUMMARY_COLUMNS)
    if summary["journey_id"].duplicated().any():
        raise SystemExit("journey_id is not unique in the batch")
    tmp = target.with_name(f".{target.name}.tmp")
    summary.to_csv(tmp, index=False, lineterminator="\n")  # LF: LOAD DATA would keep a "\r"
    tmp.replace(target)
    log(f"[3/4] journey_summary {len(summary):,} rows -> {target}")
    return stats


# --------------------------------------------------------------------------- 4. load

def load_mariadb(cfg: dict[str, str], summary_path: Path, batch_date: date, model_version: str) -> dict:
    try:
        import pymysql
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise SystemExit("LOAD_MARIADB=1 needs pymysql: pip install pymysql") from exc
    table = cfg["MARIADB_TABLE"]
    if not table.replace("_", "").isalnum():
        raise SystemExit(f"bad MARIADB_TABLE: {table}")
    frame = pd.read_csv(summary_path, dtype={c: "string" for c in ID_COLUMNS}, keep_default_na=False,
                        na_values=[""], low_memory=False)
    frame = frame.astype(object).where(frame.notna(), None)
    columns = ", ".join(SUMMARY_COLUMNS)
    placeholders = ", ".join(["%s"] * len(SUMMARY_COLUMNS))
    batch = int(cfg["MARIADB_BATCH_SIZE"])
    connection = pymysql.connect(
        host=cfg["MARIADB_HOST"], port=int(cfg["MARIADB_PORT"]), user=cfg["MARIADB_USER"],
        password=cfg["MARIADB_PASSWORD"], database=cfg["MARIADB_DATABASE"], charset="utf8mb4",
        autocommit=False,
    )
    try:
        with connection.cursor() as cursor:
            deleted = cursor.execute(
                f"DELETE FROM {table} WHERE batch_date = %s AND model_version = %s",
                (batch_date.isoformat(), model_version),
            )
            rows = frame[list(SUMMARY_COLUMNS)].itertuples(index=False, name=None)
            inserted, chunk = 0, []
            for row in rows:
                chunk.append(row)
                if len(chunk) == batch:
                    inserted += cursor.executemany(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", chunk)
                    chunk = []
            if chunk:
                inserted += cursor.executemany(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", chunk)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    log(f"[4/4] MariaDB {table}: replaced {deleted:,} old rows with {inserted:,}")
    return {"deleted": deleted, "inserted": inserted, "table": table}


# --------------------------------------------------------------------------- main

STEPS = ("events", "infer", "summary", "load")
SKIPPED = 99  # exit code of a step with nothing to do; Airflow marks that task skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="batch date YYYY-MM-DD; default: yesterday in OUTPUT_TIMEZONE")
    parser.add_argument("--config", type=Path, help="KEY=VALUE settings file")
    parser.add_argument("--step", choices=("all", *STEPS), default="all",
                        help="run one step only (one Airflow task each); default: all")
    parser.add_argument("--platform", help="--step infer: only this platform")
    parser.add_argument("--skip-export", action="store_true", help="reuse events.csv of that day")
    parser.add_argument("--no-load", action="store_true", help="never load MariaDB, even if LOAD_MARIADB=1")
    return parser.parse_args()


def update_manifest(work: Path, **fields) -> None:
    path = work / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.update(fields)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def platforms_with_events(events: Path, platforms: list[str]) -> list[str]:
    if not events.exists():
        raise SystemExit(f"{events} does not exist: run --step events first")
    present = pd.read_csv(events, usecols=["platform"], dtype="string")["platform"].str.strip().str.lower()
    return [platform for platform in platforms if present.eq(platform).any()]


def main() -> int:
    args = parse_args()
    cfg = load_settings(args.config)
    tz = ZoneInfo(cfg["OUTPUT_TIMEZONE"])
    batch_date = date.fromisoformat(args.date) if args.date else datetime.now(tz).date() - timedelta(days=1)
    model_version = cfg["MODEL_VERSION"] or resolve(cfg["MODEL_RUN"]).name
    platforms = cfg["PLATFORMS"].split()
    work = resolve(cfg["OUTPUT_ROOT"]) / model_version / batch_date.isoformat()
    work.mkdir(parents=True, exist_ok=True)
    events = work / "events.csv"
    summary_path = work / "journey_summary.csv"
    run_all = args.step == "all"
    started = time.time()
    log(f"daily job step={args.step} batch_date={batch_date} model={model_version} -> {work}")
    if run_all or args.step == "events":
        update_manifest(work, batch_date=batch_date.isoformat(), model_version=model_version,
                        platforms=platforms, status="running", started_at=datetime.now(tz).isoformat())

    if run_all or args.step == "events":
        if args.skip_export:
            if not events.exists():
                raise SystemExit(f"--skip-export but {events} does not exist")
            log(f"[1/4] reuse {events}")
        else:
            source = {"export": export_events, "folder": folder_events}.get(cfg["EVENTS_SOURCE"])
            if source is None:
                raise SystemExit(f"EVENTS_SOURCE must be export or folder, not {cfg['EVENTS_SOURCE']}")
            update_manifest(work, events=source(cfg, batch_date, events))

    if run_all or args.step == "infer":
        present = platforms_with_events(events, platforms)
        todo = [args.platform] if args.platform else platforms
        for platform in todo:
            if platform not in present:
                (work / f"{platform}_scores.csv").unlink(missing_ok=True)
                log(f"[2/4] {platform}: no events on {batch_date}, skipped")
                if not run_all and args.platform:
                    return SKIPPED
                continue
            infer(cfg, events, platform, model_version, work)

    if run_all or args.step == "summary":
        score_files = {}
        for platform in platforms_with_events(events, platforms):
            path = work / f"{platform}_scores.csv"
            if not path.exists():
                raise SystemExit(f"{path} is missing: run --step infer --platform {platform} first")
            score_files[platform] = path
        update_manifest(work, summary=build_summary(cfg, score_files, batch_date, model_version, summary_path),
                        summary_file=str(summary_path))

    if run_all or args.step == "load":
        if cfg["LOAD_MARIADB"].lower() in {"1", "true", "yes"} and not args.no_load:
            if not summary_path.exists():
                raise SystemExit(f"{summary_path} is missing: run --step summary first")
            update_manifest(work, load=load_mariadb(cfg, summary_path, batch_date, model_version))
        else:
            log("[4/4] MariaDB load off (LOAD_MARIADB=0)")
            if not run_all:
                update_manifest(work, status="done", finished_at=datetime.now(tz).isoformat())
                return SKIPPED

    if run_all or args.step == "load":
        update_manifest(work, status="done", finished_at=datetime.now(tz).isoformat())
    log(f"step {args.step} done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
