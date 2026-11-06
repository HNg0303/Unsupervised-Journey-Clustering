# Job hằng ngày: Kafka → journey → journey_summary

Một lệnh nối consumer Kafka (`../event-consumer-fastapi`) với model clustering đã đặt tên:

```bash
cp scripts/daily_job.env.example scripts/daily_job.env     # sửa nếu cần
python scripts/daily_journey_job.py --config scripts/daily_job.env               # ngày hôm qua
python scripts/daily_journey_job.py --config scripts/daily_job.env --date 2026-03-25
python scripts/daily_journey_job.py --config scripts/daily_job.env --date 2026-03-25 --step infer --platform ios  # từng bước
```

| Bước | Việc | Ở đâu |
|---|---|---|
| 1. events | Export cả ngày `[D 00:00, D+1 00:00)` giờ VN từ Kafka, không join group, không commit (`python -m app.export` của consumer). Hoặc `EVENTS_SOURCE=folder`: lấy `date=D/events.csv` consumer nền đã ghi. Bỏ dòng trùng `record_id`, xếp lại theo thứ tự gửi | `events.csv` |
| 2. infer | `journey-infer` từng platform bằng model hiện tại (`MODEL_RUN`), không train lại | `<platform>_scores.csv` |
| 3. summary | Điền `taxonomy_id` từ `<NAMES_RUN>/<platform>/<platform>_named_clusters.csv` (kiểm tra với sitemap, cụm -1 = NULL), đổi sang đúng cột bảng `journey_summary`: thêm `batch_date`, `sequence` ngăn bằng `,`, bỏ `next_action`/`next_action_share`, giờ UTC, anomaly 0/1, `journey_id` = `<ngày>_<platform>::J...` | `journey_summary.csv` |
| 4. load | Chỉ khi `LOAD_MARIADB=1`: xoá các dòng cùng `(batch_date, model_version)` rồi insert lại, nên chạy lại 1 ngày không bị trùng. Cần `pip install pymysql` | MariaDB |

Kết quả nằm ở `output/daily/<model_version>/<ngày>/` kèm `manifest.json` (số event, số journey, số cụm -1 từng platform). File có `customer_id`/`device_id`: chỉ để trên máy chạy job.

## Hai service chạy riêng (Docker)

Đang chạy tách để kiểm tra từng phần. Hai service chỉ nối với nhau qua một thư mục:

```text
service Kafka (event-consumer-fastapi, chạy liên tục :8000)
  POST /pipeline/runs {"date": D}  ->  event-consumer-fastapi/data/storage/events/date=D/<topic>.csv
                                                      │ (mount chỉ đọc)
service model (deploy/, chạy 1 ngày rồi thoát)        ▼
  docker compose run --rm journey-model --date D  ->  output/daily/<model>/D/journey_summary.csv (-> MariaDB)
```

```bash
# 1) service Kafka: chỉ lấy data, không chạy model
cd event-consumer-fastapi
cp .env.example .env                # broker, topic, KAFKA_SASL_PLAIN_USERNAME/PASSWORD nếu có
docker compose up -d --build        # Kafka giả lập: thêm -f docker-compose.yml -f docker-compose.sim.yml
curl -X POST localhost:8000/pipeline/runs -H 'Content-Type: application/json' -d '{"date": "2026-03-25"}'
curl localhost:8000/pipeline/runs   # fetch done, các bước model = skipped

# 2) service model: đọc file của ngày đó
cd ../Unsupervised-Journey-Clustering/deploy
cp model.env.example model.env      # model, tên cụm, MariaDB
docker compose build
docker compose run --rm journey-model --date 2026-03-25
```

- Service model không nối Kafka; ngày chưa được lấy về thì báo `no event files for <ngày>`. Đổi thư mục event bằng `EVENTS_PATH=... docker compose run ...`.
- `model.env` và `.env` bị git bỏ qua vì sẽ chứa mật khẩu. MariaDB chạy trên chính máy này thì đặt `MARIADB_HOST=host.docker.internal`.
- Image model cài `scikit-learn==1.9.1`, đúng bản đã pickle model 678. Train model mới bằng bản khác thì đổi số này trong `deploy/Dockerfile`.
- Đã thử với Kafka giả lập ngày 2026-03-25: service Kafka lấy 758 event, service model ra 27 journey android + 27 iOS, giống bản gộp.

### Bản gộp + Airflow (để sau)

`deploy/combined/` giữ bản gộp: một container `journey-service` vừa lấy Kafka vừa chạy model (`PIPELINE_RUN_MODEL=true`), Airflow DAG `journey_daily` (02:00 mỗi ngày) gọi nó, mỗi bước là một task. Dùng khi hai service riêng đã chạy ổn:

```bash
cd Unsupervised-Journey-Clustering/deploy/combined
cp daily.env.example daily.env
docker compose --profile airflow up -d --build      # service :8000 + Airflow :8080
```

## Cron (không dùng Docker)

Chạy 02:00 mỗi ngày cho ngày hôm qua:

```cron
0 2 * * * cd /Users/hoangnguyen/Desktop/hifpt-journey/Unsupervised-Journey-Clustering && /path/to/python scripts/daily_journey_job.py --config scripts/daily_job.env >> output/daily/cron.log 2>&1
```

## Những gì job KHÔNG làm

- Không train lại. Train lại và `scripts/carry_cluster_names.py` là việc định kỳ riêng; xong thì đổi `MODEL_RUN`, `NAMES_RUN` trong `daily_job.env` (hoặc `deploy/model.env`).
- Journey vắt qua nửa đêm bị cắt ở 00:00 (mỗi ngày chấm riêng).
- Event có `client_time` cùng giây được xếp theo thứ tự message Kafka. Đã thử: 34 ngày của `test_data_android.csv` qua topic giả lập cho 370/370 journey giống hệt chấm trực tiếp file gốc.
