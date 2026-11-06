# syntax=docker/dockerfile:1
# journey-model: service model chạy riêng. Mỗi lần chạy = 1 ngày: đọc file event mà service
# Kafka (event-consumer-fastapi) đã lấy về storage, chấm journey bằng model đã train
# (không train lại), điền taxonomy_id, ra journey_summary (-> MariaDB nếu bật).
# Build context = repo này, xem docker-compose.yml. Model, event và kết quả được mount, không đóng gói.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Ho_Chi_Minh

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src src
# Same scikit-learn version the 678 scorers were pickled with (1.9.1); change it with the model.
RUN pip install ".[pipeline]" "scikit-learn==1.9.1" pymysql

COPY scripts scripts
COPY taxonomy taxonomy

ENV EVENTS_SOURCE=folder \
    EVENTS_DIR=/data/events
ENTRYPOINT ["python", "scripts/daily_journey_job.py"]
