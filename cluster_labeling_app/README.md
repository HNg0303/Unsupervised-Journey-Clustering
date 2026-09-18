# HiFPT Taxonomy & Cluster Naming

Standalone Streamlit app để người nghiệp vụ chỉnh taxonomy 3 cấp và review named clusters
Android/iOS. App ưu tiên Supabase PostgreSQL khi có secrets và fallback về SQLite khi chạy local.

## Chạy local

```bash
cd cluster_labeling_app
python -m pip install -r requirements.txt
streamlit run app.py
```

Mặc định app dùng `db.sqlite` trong cùng folder. Có thể đổi đường dẫn database:

```bash
HIFPT_SQLITE_PATH=/path/to/review.sqlite streamlit run app.py
```

## Kết nối Supabase trực tiếp

1. Mở Supabase Dashboard → SQL Editor và chạy toàn bộ `supabase_schema.sql` một lần.
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

Lần kết nối đầu tiên, nếu bốn bảng đang rỗng, app tự seed 486 taxonomy details và toàn bộ named
clusters từ `assets/`. Các lần chạy sau giữ nguyên dữ liệu nghiệp vụ trên Supabase.

## Deploy riêng folder này

Folder tự chứa code, SQLite/Supabase schema, database local đã seed và snapshot nguồn trong `assets/`. Có thể
đưa riêng `cluster_labeling_app/` lên Git repository rồi chọn `app.py` làm main file trên
Streamlit.

SQLite phù hợp cho prototype một instance. Khi go-live, cấu hình Supabase để dữ liệu không phụ thuộc
filesystem tạm thời của Streamlit Cloud. Backend được chọn tự động: Supabase khi đủ secrets, SQLite
khi không có secrets.

## Source of truth

- `assets/taxonomy_features.csv`: 486 taxonomy details, 16 families, 85 submodules.
- `assets/cluster_mapping.csv`: taxonomy mapping cho 1.374 Android và 1.440 iOS clusters.
- `assets/*_shareholder_catalog.json`: size/share từ full July score outputs.
- `assets/*_cluster_catalog.json` và `assets/*_cluster_ngrams.csv`: metrics, medoid và top n-grams.

Các file taxonomy/mapping được lấy từ `output/scores/taxonomy_naming`, tức pipeline taxonomy mới,
không dùng mapping legacy của app cũ.

## Database tables

SQLite schema nằm trong `schema.sql`; PostgreSQL schema nằm trong `supabase_schema.sql`.

- `taxonomy_features`: feature 3 cấp và `taxonomy_id` ổn định; xóa trong UI là soft-delete.
- `named_clusters`: một dòng cho `(platform, cluster_id)`, gồm mapping, confidence, evidence và
  score volume.
- `change_audit`: lịch sử before/after cho mỗi thay đổi taxonomy hoặc cluster.
- `app_metadata`: version và nguồn seed.

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

Tests kiểm tra seed, foreign-key contract, persistence, propagation taxonomy → named clusters,
dropdown updates, audit, filter/export và đủ 8 n-grams cho mỗi cluster.
