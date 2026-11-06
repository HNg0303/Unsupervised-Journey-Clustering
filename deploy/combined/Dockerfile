# syntax=docker/dockerfile:1
# BẢN GỘP (để sau). journey-service: one image for both projects. The FastAPI app of event-consumer-fastapi
# takes POST /pipeline/runs, exports one day of Kafka to storage and runs the model job
# (Unsupervised-Journey-Clustering/scripts/daily_journey_job.py) in the same container.
# Build context = the hifpt-journey folder, see docker-compose.yml. Models, storage and
# daily outputs are mounted, not baked in.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Ho_Chi_Minh

WORKDIR /app

COPY event-consumer-fastapi/requirements.txt event-consumer-fastapi/requirements.txt
RUN pip install -r event-consumer-fastapi/requirements.txt pymysql

COPY Unsupervised-Journey-Clustering/pyproject.toml Unsupervised-Journey-Clustering/README.md Unsupervised-Journey-Clustering/
COPY Unsupervised-Journey-Clustering/src Unsupervised-Journey-Clustering/src
# Same scikit-learn version the 678 scorers were pickled with (1.9.1); change it with the model.
RUN pip install "./Unsupervised-Journey-Clustering[pipeline]" "scikit-learn==1.9.1"

COPY event-consumer-fastapi/app event-consumer-fastapi/app
COPY Unsupervised-Journey-Clustering/scripts Unsupervised-Journey-Clustering/scripts
COPY Unsupervised-Journey-Clustering/taxonomy Unsupervised-Journey-Clustering/taxonomy

ENV MODEL_JOB_DIR=/app/Unsupervised-Journey-Clustering \
    STORAGE_DIR=/app/event-consumer-fastapi/data/storage
WORKDIR /app/event-consumer-fastapi
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
