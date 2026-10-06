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

# 1. Bật MongoDB, Kafka, Kafka Connect, PostgreSQL, Kafka UI (lần đầu build image
#    Connect, tải plugin Debezium 3.6 từ Maven Central). Service connect-ops tự
#    đăng ký source + sink connector ngay khi Connect sẵn sàng.
docker compose up -d --build --wait
docker compose logs connect-ops
#   registered mongo-source / registered postgres-sink
#   mongo-source: connector=RUNNING tasks=RUNNING
#   postgres-sink: connector=RUNNING tasks=RUNNING

# 1b. Kiểm tra nhanh cả đường đi: insert / update / delete 1 document trong Mongo
#     và chờ thấy đúng trạng thái trong PostgreSQL
python scripts/smoke_test.py
#   ok   insert (3.0s) ... ok   update ... ok   delete ... smoke test passed

# 2. Tạo SQL dump mẫu (4.999 event từ test_data_android.csv trong repo)
python scripts/make_sample_sql.py            # -> sample/raw_events.sql

# 3. SQL -> JSON -> MongoDB. Xem thử vài document trước:
python scripts/sql_to_mongo.py sample/raw_events.sql --nest-prefix segmentation_ --dry-run 2
python scripts/sql_to_mongo.py sample/raw_events.sql --nest-prefix segmentation_ --drop

# 4. (Không bắt buộc) Sửa file trong connectors/ rồi áp lại config bằng tay
python scripts/connect_ops.py        # hoặc ./scripts/register_connectors.sh

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

Cách dễ nhất là Kafka UI (Kafbat) ở http://localhost:8081: tab **Topics** để xem
message trong `app.tracking.events`, **Consumers** để xem consumer group
`connect-postgres-sink` đọc tới offset nào (lag), **Kafka Connect** để xem trạng
thái, config và restart connector.

Hoặc bằng dòng lệnh:

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
  `FAILED` gần như ngay khi mất kết nối (lỗi xảy ra lúc kiểm tra bảng, trước bước
  flush nên `flush.max.retries` không che được), và Kafka Connect không tự restart
  task. Service `connect-ops` kiểm tra mỗi 30 giây và restart task `FAILED`
  (`docker compose logs connect-ops`), nên khi PostgreSQL lên lại thì các dòng mới
  tự xuống, không cần làm gì. Ở công ty, theo dõi trạng thái `FAILED` này là việc
  của team vận hành Connect.
- Một luồng thay đổi phục vụ được nhiều nơi đọc (PostgreSQL, data lake, service realtime)
  mà MongoDB chỉ bị đọc một lần.
- Thứ tự thay đổi của từng document được giữ nhờ partition theo key.

Cronjob của bạn không cần đọc Kafka: nó đọc bảng PostgreSQL, nơi data đã ổn định.

## Đọc thẳng từ Kafka topic theo khoảng thời gian

Kafka không phải database nên không có `SELECT ... WHERE date BETWEEN`. Mỗi
partition là một log chỉ ghi nối, đọc theo offset. Thứ duy nhất Kafka đánh
index theo thời gian là **timestamp của record** (lúc thay đổi được ghi vào
Kafka). `scripts/read_topic_range.py` gói hai cách lấy "1 ngày / 1 tuần / 1 tháng":

```bash
pip install confluent-kafka

# Theo thời gian sự kiện (client_time): quét toàn bộ topic rồi lọc
python scripts/read_topic_range.py --from 2026-10-01 --to 2026-10-02 --time event   # 1 ngày
python scripts/read_topic_range.py --last 7d --time event --output week.csv         # 1 tuần
python scripts/read_topic_range.py --last 1m --time event --output month.csv        # 30 ngày

# Theo thời gian vào Kafka: nhảy thẳng tới offset đầu tiên >= --from (offsets_for_times)
python scripts/read_topic_range.py --last 1h --time kafka
```

| | `--time kafka` | `--time event` |
|---|---|---|
| Lọc theo | lúc thay đổi vào Kafka | `client_time` trong document |
| Cách đọc | seek tới offset theo timestamp, chỉ đọc đoạn cần | đọc mọi record còn trong topic |
| Bẫy | snapshot lần đầu dồn **toàn bộ** data cũ vào đúng thời điểm connector khởi động | chậm khi topic lớn |

Script bóc envelope Debezium (`payload.after` là chuỗi JSON), làm phẳng field
lồng, giữ bản mới nhất theo `_id` và bỏ document đã bị delete, tức là tự làm lại
đúng việc mà sink làm.

Giới hạn quan trọng: chỉ đọc được data còn trong **retention** của topic (ở đây
7 ngày, `KAFKA_LOG_RETENTION_HOURS`). Muốn lấy 1 tháng thì topic phải giữ ít
nhất 1 tháng, ở công ty là do team vận hành Kafka quyết định. Ngoài ra đọc Kafka
công ty cần thêm quyền (consumer group, ACL trên topic), khác với quyền đọc
PostgreSQL. Vì vậy cho job hằng ngày vẫn nên đọc bảng PostgreSQL; đọc Kafka hợp
khi cần data gần realtime hoặc để đối chiếu.

## Lưu ý

- Hình dạng document thật trong MongoDB của công ty chưa biết. Bản mẫu giả định
  các cột phẳng, chỉ `segmentation_*` là sub-document. Khi có dump thật, chạy
  `--dry-run` để xem và chỉnh `--nest-prefix`, `--table` cho khớp.
- `sql_to_mongo.py` đọc được `INSERT INTO ... VALUES` (MySQL/PostgreSQL/SQL Server),
  khối `COPY ... FROM stdin` của `pg_dump`, và file `.csv`.
- Build image Connect cần truy cập `repo1.maven.org`; sau proxy công ty thì thêm
  `--build-arg HTTPS_PROXY=...`.
- Dọn dẹp: `docker compose --profile airflow down -v`.
