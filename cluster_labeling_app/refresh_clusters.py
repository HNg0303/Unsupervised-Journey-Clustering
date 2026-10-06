"""Publish a trained model's clusters and evidence into the review app's database.

This is the only path from local pipeline outputs to the database the app reads. Training
or editing files locally changes nothing in the deployed app until you run this script
against Supabase explicitly.

Usage (from this folder):

    python refresh_clusters.py --backend supabase \
        --training-run ../output/partitioned_runs/678/678_20260929_090904 \
        --scores-run ../output/scores/678/678_20260929_090904

The training run supplies the raw catalog/n-grams, the scores run supplies the
taxonomy-named shareholder catalogs and cluster_mapping.csv. On an empty database the
taxonomy is seeded from --taxonomy. Otherwise taxonomy and audit history stay in the
database; the previous named_clusters are backed up to backups/ before being replaced.

--evidence-only loads catalog/n-grams for the model that is already deployed without
touching named_clusters, so reviewed labels are kept.
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
REPO_ROOT = APP_DIR.parent
BACKUP_DIR = APP_DIR / "backups"
DEFAULT_TAXONOMY = (
    REPO_ROOT / "output" / "scores" / "taxonomy_naming" / "hifpt_journey_taxonomy_3_levels.csv"
)

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
    if choice == "supabase":
        url = secret("SUPABASE_URL")
        key = secret("SUPABASE_SECRET_KEY") or secret("SUPABASE_KEY")
        if not (url and key):
            raise SystemExit("Cần SUPABASE_URL và SUPABASE_SECRET_KEY để publish lên Supabase.")
        import supabase_database

        return supabase_database, supabase_database.SupabaseTarget(url, key), f"Supabase {url}"
    db_path = Path(os.environ.get("HIFPT_SQLITE_PATH", APP_DIR / "db.sqlite")).resolve()
    return sqlite_database, db_path, f"SQLite {db_path}"


def display_path(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--scores-run", type=Path, required=True)
    parser.add_argument("--model-version", help="mặc định: tên thư mục scores run")
    parser.add_argument(
        "--backend",
        choices=("sqlite", "supabase"),
        required=True,
        help="database đích; chọn rõ ràng để không vô tình ghi lên Supabase",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
        help="taxonomy CSV, chỉ dùng khi seed database rỗng",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="chỉ nạp catalog/n-grams cho model đang deploy, giữ nguyên named_clusters",
    )
    parser.add_argument("--yes", action="store_true", help="bỏ qua xác nhận khi ghi Supabase")
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
    source = (
        f"training_run={display_path(args.training_run)}; "
        f"scores_run={display_path(args.scores_run)}"
    )

    has_clusters = database_api.has_named_clusters(target)
    if args.evidence_only:
        action = f"nạp cluster_evidence cho model {model_version} (giữ named_clusters)"
        if not has_clusters:
            raise SystemExit(sqlite_database.EMPTY_DATABASE_MESSAGE)
    elif has_clusters:
        action = f"thay named_clusters + cluster_evidence bằng model {model_version}"
    else:
        if not args.taxonomy.is_file():
            raise SystemExit(f"Không tìm thấy taxonomy CSV: {args.taxonomy}")
        action = f"seed taxonomy + model {model_version} vào database rỗng"
        source += f"; taxonomy={display_path(args.taxonomy)}"
    print(f"Đích: {label}\nThao tác: {action}\nNguồn: {source}")
    if args.backend == "supabase" and not args.yes:
        if input("Ghi lên Supabase (app đang deploy sẽ thấy ngay)? [y/N] ").strip().lower() != "y":
            print("Đã hủy.")
            return 1

    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        for name, path in files.items():
            shutil.copy2(path, staging / name)
        for platform in PLATFORMS:
            # Raises when catalog and n-gram cluster IDs disagree.
            load_platform_evidence(staging, platform)

        if args.evidence_only:
            counts = database_api.publish_evidence(target, staging, model_version, source)
        elif not has_clusters:
            shutil.copy2(args.taxonomy, staging / "taxonomy_features.csv")
            stats = database_api.initialize_database(target, staging, model_version, source)
            counts = {platform: stats[f"{platform}_clusters"] for platform in PLATFORMS}
        else:
            BACKUP_DIR.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for platform in PLATFORMS:
                backup = BACKUP_DIR / f"named_clusters_{platform}_{stamp}.csv"
                database_api.load_named_clusters(target, platform).to_csv(backup, index=False)
                print(f"Backup {platform}: {backup.relative_to(APP_DIR)}")
            counts = database_api.replace_named_clusters(target, staging, model_version, source)

    print(
        f"{label}: {model_version} → "
        + ", ".join(f"{name} {count:,}" for name, count in counts.items())
    )
    print("App tự đọc model mới trong vòng 30 giây; không cần redeploy hay đẩy file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
