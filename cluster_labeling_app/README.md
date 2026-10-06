# HiFPT Taxonomy & Cluster Naming

Standalone Streamlit app để người nghiệp vụ chỉnh taxonomy 3 cấp và review named clusters
Android/iOS. App ưu tiên Supabase PostgreSQL khi có secrets và fallback về SQLite khi chạy local.

## Chạy local

```bash
cd cluster_labeling_app
python -m pip install -r requirements.txt
streamlit run app.py
```

App chỉ đọc database (Supabase hoặc SQLite), không đọc `assets/` hay `output/`. Database rỗng thì
app báo lỗi; publish một model trước bằng `refresh_clusters.py` (xem bên dưới):

```bash
python refresh_clusters.py --backend sqlite \
    --training-run ../output/partitioned_runs/678/678_20260929_090904 \
    --scores-run ../output/scores/678/678_20260929_090904
```

Mặc định app dùng `db.sqlite` trong cùng folder. Có thể đổi đường dẫn database:

```bash
HIFPT_SQLITE_PATH=/path/to/review.sqlite streamlit run app.py
```

## Kết nối Supabase trực tiếp

1. Mở Supabase Dashboard → SQL Editor và chạy toàn bộ `supabase_schema.sql`. File idempotent; chạy
   lại sau khi nâng cấp app để tạo bảng mới (ví dụ `cluster_evidence`).
2. Tạo file `.streamlit/secrets.toml` khi chạy local, dựa trên
   `.streamlit/secrets.example.toml`.
3. Khi deploy Streamlit Community Cloud, nhập cùng hai secrets trong phần App settings → Secrets:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_..."
```

Không commit `secrets.toml`. File này đã được ignore. App cũng chấp nhận `SUPABASE_URL` và
`SUPABASE_SECRET_KEY` từ environment variables.

Publishable key (`sb_publishable_...`) chỉ truy cập những row mà RLS cho phép. Schema đi kèm dùng
secure default: bật RLS nhưng không mở quyền ghi public. Vì đây là công cụ quản trị dữ liệu chạy
server-side trên Streamlit, hãy lưu `sb_secret_...` trong Streamlit Secrets. Không gửi secret key
qua chat và không đưa nó vào source code.

App đang deploy không bao giờ tự seed hay đọc file local: train lại hoặc sửa file trên máy không
ảnh hưởng app cho tới khi chạy `refresh_clusters.py --backend supabase`. Mọi lần publish ghi một dòng
`change_audit` (model version, số cluster/evidence, đường dẫn run nguồn).

**Nâng cấp deployment cũ** (đã có `named_clusters` nhưng chưa có `cluster_evidence`): chạy lại
`supabase_schema.sql`, rồi nạp evidence cho đúng model đang deploy mà không ghi đè nhãn đã review:

```bash
python refresh_clusters.py --backend supabase --evidence-only \
    --training-run ../output/partitioned_runs/678/678_20260929_090904 \
    --scores-run ../output/scores/678/678_20260929_090904
```

## Deploy riêng folder này

Folder tự chứa code và SQLite/Supabase schema; mọi dữ liệu (taxonomy, named clusters, catalog,
n-grams) nằm trong database. Có thể đưa riêng `cluster_labeling_app/` lên Git repository rồi chọn
`app.py` làm main file trên Streamlit.

SQLite phù hợp cho prototype một instance. Khi go-live, cấu hình Supabase để dữ liệu không phụ thuộc
filesystem tạm thời của Streamlit Cloud. Backend được chọn tự động: Supabase khi đủ secrets, SQLite
khi không có secrets.

## Source of truth

Database là nguồn duy nhất app đọc. `refresh_clusters.py` publish từ output pipeline:

- `taxonomy_features`: seed một lần từ `output/scores/taxonomy_naming/hifpt_journey_taxonomy_3_levels.csv`
  (486 details, 16 families, 85 submodules); sau đó chỉ sửa trong app.
- `named_clusters`: `<scores-run>/taxonomy_naming/cluster_mapping.csv` + size/share từ
  `<scores-run>/<platform>/<platform>_taxonomy_shareholder_catalog.json`.
- `cluster_evidence`: metrics, medoid và top n-grams từ
  `<training-run>/<platform>/<platform>_cluster_catalog.json` và `<platform>_cluster_ngrams.csv`.

Các file taxonomy/mapping được lấy từ `output/scores/taxonomy_naming`, tức pipeline taxonomy mới,
không dùng mapping legacy của app cũ.

## Cập nhật clusters khi train model mới

Train/score ở local không thay đổi app đang deploy. Khi muốn đưa model mới lên, publish rõ ràng
(`--backend` là bắt buộc; ghi Supabase sẽ hỏi xác nhận, thêm `--yes` để bỏ qua):

```bash
python refresh_clusters.py --backend supabase \
    --training-run ../output/partitioned_runs/678/678_20260929_090904 \
    --scores-run ../output/scores/678/678_20260929_090904
```

Script kiểm tra catalog/n-grams/mapping cùng cluster_id và taxonomy_id khớp taxonomy active, backup
`named_clusters` hiện tại vào `backups/`, rồi thay `named_clusters` và `cluster_evidence` trong
database đích. Database rỗng thì seed taxonomy (`--taxonomy`) cùng model. Taxonomy và `change_audit`
được giữ nguyên; một record `cluster_refresh` (model version, số cluster/evidence, run nguồn) và
`app_metadata.model_version` ghi lại model đang dùng. App nhận model mới trong khoảng 30 giây, không
cần redeploy. Nhãn người review của model cũ không tự chuyển sang model mới vì cluster_id không
tương ứng giữa các lần train.

## Database tables

SQLite schema nằm trong `schema.sql`; PostgreSQL schema nằm trong `supabase_schema.sql`.

- `taxonomy_features`: feature 3 cấp và `taxonomy_id` ổn định; xóa trong UI là soft-delete.
- `named_clusters`: một dòng cho `(platform, cluster_id)`, gồm mapping, confidence, evidence và
  score volume.
- `cluster_evidence`: catalog (metrics, medoid path) và top n-grams cho mỗi `(platform, cluster_id)`,
  read-only trong app; chỉ `refresh_clusters.py` ghi.
- `change_audit`: lịch sử before/after cho mỗi thay đổi taxonomy hoặc cluster và mỗi lần publish
  (`database_seed`, `cluster_refresh`, `evidence_refresh`).
- `app_metadata`: schema version, `model_version` đang deploy và nguồn seed.

Khi một taxonomy row đổi tên, mọi `named_clusters` đang tham chiếu cùng `taxonomy_id` được cập nhật
trong cùng transaction, chuyển `naming_source` sang `business_review_taxonomy_update` và bật lại
`needs_review`.

## UI workflow

- Tab 1 giải thích evidence và hiển thị audit gần nhất.
- Tab 2 chỉnh taxonomy. Thay đổi hợp lệ tự ghi database; dòng mới nhận ID `USR.000001` trở đi.
- Tab 3 lọc/review 5 hoặc 10 clusters/trang. Dropdown Family → Submodule → Detail lấy trực tiếp từ
  các taxonomy rows đang active. N-gram tự xuống dòng và dùng dấu `→` phân cách từng token.
- Mỗi cluster có nút **Lưu vào database**; thay đổi chỉ ghi bảng `named_clusters` sau khi người review
  chủ động bấm lưu.
- Export CSV luôn đọc trạng thái mới nhất từ backend đang được chọn.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests publish run mới nhất trong `output/` vào SQLite tạm và kiểm tra app không seed từ file,
evidence round-trip qua database, `--evidence-only` giữ nhãn đã review, seed, foreign-key contract, persistence, propagation taxonomy → named clusters,
dropdown updates, audit, filter/export và đủ 8 n-grams cho mỗi cluster.
