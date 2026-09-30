# Bundled source snapshots

Read-only snapshot dùng để seed `db.sqlite` lần đầu và hiển thị raw evidence. App không ghi đè các
file này; mọi thay đổi nghiệp vụ được lưu trong SQLite.

- Model hiện tại: `678_20260929_090904` (cập nhật bằng `refresh_clusters.py`)
- Taxonomy: `taxonomy_features.csv` (486 details)
- Cluster mapping: `output/scores/678/678_20260929_090904/taxonomy_naming/cluster_mapping.csv`
- Shareholder catalogs: `output/scores/678/678_20260929_090904/<platform>/<platform>_taxonomy_shareholder_catalog.json`
- Raw catalog/ngrams: `output/partitioned_runs/678/678_20260929_090904/<platform>/`
