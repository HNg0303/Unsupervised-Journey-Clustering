"""Point the review app at a newly trained model's clusters.

Usage (from this folder):

    python refresh_clusters.py \
        --training-run ../output/partitioned_runs/678/678_20260929_090904 \
        --scores-run ../output/scores/678/678_20260929_090904

The training run supplies the raw catalog/n-grams, the scores run supplies the
taxonomy-named shareholder catalogs and cluster_mapping.csv. Taxonomy and audit
history stay in the database; the previous named_clusters are backed up to
backups/ before being replaced.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import database as sqlite_database
from core import PLATFORMS, load_platform_evidence

APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
BACKUP_DIR = APP_DIR / "backups"

load_dotenv()

def source_files(training_run: Path, scores_run: Path) -> dict[str, Path]:
    files = {"cluster_mapping.csv": scores_run / "taxonomy_naming" / "cluster_mapping.csv"}
    for platform in PLATFORMS:
        files[f"{platform}_cluster_catalog.json"] = (
            training_run / platform / f"{platform}_cluster_catalog.json"
        )
        files[f"{platform}_cluster_ngrams.csv"] = (
            training_run / platform / f"{platform}_cluster_ngrams.csv"
        )
        files[f"{platform}_shareholder_catalog.json"] = (
            scores_run / platform / f"{platform}_taxonomy_shareholder_catalog.json"
        )
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit("Thiếu file nguồn:\n  " + "\n  ".join(missing))
    return files


def secret(name: str) -> str:
    if os.getenv(name, ""):
        return os.getenv(name).strip()
    path = APP_DIR / ".streamlit" / "secrets.toml"
    if path.is_file():
        return str(tomllib.loads(path.read_text()).get(name, "")).strip()
    return ""


def select_backend(choice: str):
    url = secret("SUPABASE_URL")
    key = secret("SUPABASE_SECRET_KEY") or secret("SUPABASE_KEY")
    if choice == "supabase" or (choice == "auto" and url and key):
        if not (url and key):
            raise SystemExit("Cần SUPABASE_URL và SUPABASE_SECRET_KEY để refresh Supabase.")
        import supabase_database

        return supabase_database, supabase_database.SupabaseTarget(url, key), "Supabase"
    db_path = Path(os.environ.get("HIFPT_SQLITE_PATH", APP_DIR / "db.sqlite")).resolve()
    return sqlite_database, db_path, f"SQLite {db_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--scores-run", type=Path, required=True)
    parser.add_argument("--model-version", help="mặc định: tên thư mục scores run")
    parser.add_argument("--backend", choices=("auto", "sqlite", "supabase"), default="auto")
    args = parser.parse_args()

    model_version = args.model_version or args.scores_run.resolve().name
    if args.training_run.resolve().name != args.scores_run.resolve().name:
        print(
            f"Cảnh báo: training run {args.training_run.name} khác scores run "
            f"{args.scores_run.name}; kiểm tra lại hai thư mục có cùng model.",
            file=sys.stderr,
        )
    files = source_files(args.training_run, args.scores_run)
    database_api, target, label = select_backend(args.backend)

    with tempfile.TemporaryDirectory(dir=APP_DIR) as staging_name:
        staging = Path(staging_name)
        for name, source in files.items():
            shutil.copy2(source, staging / name)
        shutil.copy2(ASSET_DIR / "taxonomy_features.csv", staging / "taxonomy_features.csv")
        for platform in PLATFORMS:
            # Raises when catalog and n-gram cluster IDs disagree.
            load_platform_evidence(staging, platform)

        BACKUP_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for platform in PLATFORMS:
            backup = BACKUP_DIR / f"named_clusters_{platform}_{stamp}.csv"
            database_api.load_named_clusters(target, platform).to_csv(backup, index=False)
            print(f"Backup {platform}: {backup.relative_to(APP_DIR)}")

        counts = database_api.replace_named_clusters(target, staging, model_version)
        for name in files:
            shutil.copy2(staging / name, ASSET_DIR / name)

    print(
        f"{label}: named_clusters → {model_version} "
        + ", ".join(f"{platform} {count:,}" for platform, count in counts.items())
    )
    print("Khởi động lại Streamlit app (hoặc push assets/ khi deploy) để app đọc evidence mới.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
