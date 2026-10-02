# Mô phỏng local: MongoDB → Debezium → Kafka → PostgreSQL → Airflow

Thư mục này dựng lại luồng data của công ty trên máy cá nhân, để viết và thử
cronjob Airflow trước khi có quyền vào hệ thống thật.

```
 SQL dump / CSV ──sql_to_mongo.py──▶ MongoDB (tracking.events, replica set rs0)
                                          │ change streams
                                          ▼
                              Debezium MongoDB source  ┐
                                          │            │ cùng chạy trong
                                          ▼            │ Kafka Connect
                              Kafka topic app.tracking.events
                                          │            │
                                          ▼            │
                              Debezium JDBC sink       ┘  (ExtractNewDocumentState)
                                          ▼
                              PostgreSQL bảng public.events
                                          │ 02:00 mỗi ngày
                                          ▼
                       Airflow DAG journey_daily_extract
                         ├─ extract: exports/<ngày>/events.csv
                         └─ run_full_pipeline.sh (bật bằng JC_RUN_MODEL=1)
```

| Thành phần local | Tương ứng ở công ty | Khi chuyển sang hệ thống thật |
|---|---|---|
| `sql_to_mongo.py` + MongoDB | App/backend ghi raw event vào MongoDB | Bỏ, data đã có sẵn |
| `connectors/mongo-source.json` | Connector source do team data quản lý | Chỉ để đối chiếu config |
| Kafka + topic `app.tracking.events` | Kafka cluster của công ty | Không cần đụng |
| `connectors/postgres-sink.json` | Connector sink do team data quản lý | Hỏi họ tên bảng, cột thời gian, cách xử lý delete |
| PostgreSQL `public.events` | Bảng sink trong PostgreSQL công ty | Đổi connection `cdc_postgres` + `JC_SOURCE_TABLE` |
| `airflow/dags/journey_daily_extract.py` | Cronjob của bạn | Giữ nguyên, chỉ đổi biến môi trường |

## Chạy

Cần Docker (Compose v2) và Python 3.10+.

```bash
cd local_cdc
pip install pymongo psycopg2-binary

# 1. Bật MongoDB, Kafka, Kafka Connect, PostgreSQL (lần đầu build image Connect,
#    tải plugin Debezium 3.6 từ Maven Central)
docker compose up -d --build --wait

# 2. Tạo SQL dump mẫu (4.999 event từ test_data_android.csv trong repo)
python scripts/make_sample_sql.py            # -> sample/raw_events.sql

# 3. SQL -> JSON -> MongoDB. Xem thử vài document trước:
python scripts/sql_to_mongo.py sample/raw_events.sql --nest-prefix segmentation_ --dry-run 2
python scripts/sql_to_mongo.py sample/raw_events.sql --nest-prefix segmentation_ --drop

# 4. Đăng ký source + sink connector qua REST API của Kafka Connect
./scripts/register_connectors.sh
#   mongo-source: connector=RUNNING tasks=RUNNING
#   postgres-sink: connector=RUNNING tasks=RUNNING

# 5. Kiểm tra data đã xuống PostgreSQL
docker compose exec postgres psql -U cdc -d analytics -c "select count(*), min(client_time), max(client_time) from events"

# 6. Export 1 ngày (chính là việc DAG làm)
python scripts/extract_daily.py --day 2026-03-26 --output-root exports
#   public.events client_time in [2026-03-25, 2026-03-27): 465 rows -> exports/2026-03-26/events.csv
```

File CSV này có đúng các cột như bản export thủ công, nên đưa thẳng vào model:

```bash
cd .. && RAW_INPUT=local_cdc/exports/2026-03-26 scripts/run_full_pipeline.sh
```

### Airflow

```bash
docker compose --profile airflow up -d      # UI: http://localhost:8080 (không cần đăng nhập)
docker compose exec airflow airflow dags unpause journey_daily_extract
docker compose exec airflow airflow dags trigger journey_daily_extract -c '{"day": "2026-03-26"}'
```

DAG chạy lúc 02:00 hằng ngày và lấy **ngày hôm trước** (hoặc ngày trong param
`day`). Cửa sổ là `[ngày - lookback_days, ngày + 1)` theo `client_time`, mặc
định lookback 1 ngày để không sót event đến muộn; trùng lặp được loại theo `_id`.
Task `run_full_pipeline` bị skip cho tới khi worker có repo + thư viện của model
và bạn đặt `JC_RUN_MODEL=1` (và `JC_REPO_DIR`).

Data mẫu nằm trong tháng 3/2026. Muốn lần chạy theo lịch (lấy "hôm qua") có
data thì dời mốc thời gian khi nạp:

```bash
python scripts/sql_to_mongo.py sample/raw_events.sql --nest-prefix segmentation_ --rebase-to $(date -d yesterday +%F)
```

## Quan sát từng tầng

```bash
# Document gốc trong MongoDB (segmentation là sub-document)
docker compose exec mongo mongosh --quiet tracking --eval 'db.events.findOne()'

# Change event thô trong Kafka: envelope có op, after (chuỗi JSON), source.ts_ms
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 --topic app.tracking.events --from-beginning --max-messages 1

# Thử insert / update / delete rồi xem bảng PostgreSQL thay đổi gần như ngay
docker compose exec mongo mongosh --quiet tracking --eval '
  db.events.insertOne({session_id: "demo", key: "action", segmentation: {name: "pay_bill"}, client_time: new Date()});
  db.events.updateOne({session_id: "demo"}, {$set: {"segmentation.name": "pay_bill_vnpay"}});'
docker compose exec postgres psql -U cdc -d analytics \
  -c "select id, segmentation_name, __op, __deleted from events where session_id = 'demo'"
```

Trong bảng `events`:

- `id`: khóa chính, lấy từ key của Kafka record (= `_id` của document).
- `segmentation_name`: `segmentation.name` đã được làm phẳng (`flatten.struct`).
- `__op`: `r` snapshot lần đầu, `c` insert, `u` update, `d` delete.
- `__source_ts_ms`: thời điểm thay đổi xảy ra trong MongoDB.
- `__deleted`: delete được ghi thành soft delete (`delete.tombstone.handling.mode=rewrite`);
  đổi sang `tombstone` nếu muốn xóa hẳn dòng. `extract_daily.py` bỏ qua dòng đã xóa.

## Vì sao có Kafka ở giữa

Debezium source chỉ biết đọc thay đổi và đẩy ra; nó không ghi thẳng vào
PostgreSQL. Kafka là chỗ giữ các thay đổi đó (ở đây 7 ngày) nên:

- PostgreSQL hoặc sink chết vài giờ không mất data: event vẫn nằm trong topic, sink
  chạy lại sẽ đọc tiếp từ offset đã commit. Thử: `docker compose stop postgres`,
  insert vài document, `docker compose start postgres`. Sink task chuyển sang
  `FAILED` khi mất kết nối, nên phải restart nó rồi mới thấy các dòng mới:
  `curl -X POST "localhost:8083/connectors/postgres-sink/restart?includeTasks=true&onlyFailed=true"`.
  Ở công ty, theo dõi trạng thái `FAILED` này là việc của team vận hành Connect.
- Một luồng thay đổi phục vụ được nhiều nơi đọc (PostgreSQL, data lake, service realtime)
  mà MongoDB chỉ bị đọc một lần.
- Thứ tự thay đổi của từng document được giữ nhờ partition theo key.

Cronjob của bạn không cần đọc Kafka: nó đọc bảng PostgreSQL, nơi data đã ổn định.

## Lưu ý

- Hình dạng document thật trong MongoDB của công ty chưa biết. Bản mẫu giả định
  các cột phẳng, chỉ `segmentation_*` là sub-document. Khi có dump thật, chạy
  `--dry-run` để xem và chỉnh `--nest-prefix`, `--table` cho khớp.
- `sql_to_mongo.py` đọc được `INSERT INTO ... VALUES` (MySQL/PostgreSQL/SQL Server),
  khối `COPY ... FROM stdin` của `pg_dump`, và file `.csv`.
- Build image Connect cần truy cập `repo1.maven.org`; sau proxy công ty thì thêm
  `--build-arg HTTPS_PROXY=...`.
- Dọn dẹp: `docker compose --profile airflow down -v`.
