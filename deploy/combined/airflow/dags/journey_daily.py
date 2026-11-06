"""journey_daily: mỗi ngày 02:00 (giờ VN) nhờ journey-service lấy event Kafka của ngày hôm qua
về storage rồi chạy model; mỗi bước của service là 1 task Airflow.

    start_run -> wait_fetch -> wait_prepare -> wait_infer -> wait_summary -> wait_load

Việc nặng chạy trong service (POST /pipeline/runs). Airflow chỉ gọi HTTP, nên image Airflow
không cần thư viện của model hay Kafka.

Chạy tay 1 ngày khác: Trigger DAG w/ config {"date": "2026-03-25", "topics": ["..."]}.
Chạy lại 1 ngày lỗi: Clear cả DAG run (start_run tạo run mới vì run cũ đã kết thúc).

Biến môi trường của Airflow:
    JOURNEY_SERVICE_URL      mặc định http://journey-service:8000
    JOURNEY_SERVICE_API_KEY  (hoặc API_KEY) bằng API_KEY của service, nếu có đặt
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

TZ = "Asia/Ho_Chi_Minh"
STAGES = ("fetch", "prepare", "infer", "summary", "load")
# Run lúc 02:00 ngày D có data_interval_start = 02:00 ngày D-1, tức ngày hôm qua theo giờ VN.
BATCH_DATE = (
    "{{ (dag_run.conf or {}).get('date') "
    "or data_interval_start.in_timezone('Asia/Ho_Chi_Minh').to_date_string() }}"
)


def call(method: str, path: str, body: dict | None = None) -> dict:
    base = os.environ.get("JOURNEY_SERVICE_URL", "http://journey-service:8000").rstrip("/")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("JOURNEY_SERVICE_API_KEY") or os.environ.get("API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{base}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def start_run(batch_date: str, dag_run=None, **_) -> str:
    body = {"date": batch_date}
    topics = (dag_run.conf or {}).get("topics") if dag_run else None
    if topics:
        body["topics"] = topics
    run = call("POST", "/pipeline/runs", body)
    print(f"service run {run['id']} for {batch_date}, topics {run['topics']}")
    return run["id"]


def stage_finished(stage: str, ti=None, **_) -> bool:
    run = call("GET", f"/pipeline/runs/{ti.xcom_pull(task_ids='start_run')}")
    state = run["stages"][stage]["state"]
    print(f"run {run['id']}: {stage} {state}, run {run['state']}")
    if state == "done":
        details = {k: v for k, v in run["stages"][stage].items() if k != "manifest"}
        print(json.dumps(details, ensure_ascii=False))
        return True
    if state == "skipped":
        raise AirflowSkipException(f"{stage} skipped by the service")
    if state in {"failed", "cancelled"} or run["state"] in {"failed", "interrupted"}:
        raise AirflowFailException(f"{stage} {state}: {run.get('error')} (log {run.get('log')})")
    return False


with DAG(
    dag_id="journey_daily",
    description="Kafka theo ngày -> storage -> model -> journey_summary (qua journey-service)",
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2026, 10, 1, tz=TZ),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["journey", "kafka"],
    doc_md=__doc__,
) as dag:
    previous = PythonOperator(task_id="start_run", python_callable=start_run,
                              op_kwargs={"batch_date": BATCH_DATE})
    for stage in STAGES:
        sensor = PythonSensor(
            task_id=f"wait_{stage}",
            python_callable=stage_finished,
            op_kwargs={"stage": stage},
            mode="reschedule",
            poke_interval=30,
            timeout=6 * 3600,
        )
        previous >> sensor
        previous = sensor
