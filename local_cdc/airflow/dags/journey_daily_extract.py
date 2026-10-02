"""Daily job: pull yesterday's raw events from the PostgreSQL sink table and
(optionally) run the journey clustering pipeline on them.

Locally the connection ``cdc_postgres`` points at the simulated sink
(see docker-compose.yml). On the company system only that connection and
``JC_SOURCE_TABLE`` change.

Trigger a specific day by hand with the ``day`` param (YYYY-MM-DD), e.g. a day
inside the sample data: ``airflow dags trigger journey_daily_extract -c '{"day": "2026-03-26"}'``.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowSkipException
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Connection, Param, dag, get_current_context, task

sys.path.insert(0, os.environ.get("JC_SCRIPTS_DIR", "/opt/airflow/local_cdc_scripts"))
from extract_daily import extract_day  # noqa: E402

OUTPUT_ROOT = Path(os.environ.get("JC_OUTPUT_ROOT", "/opt/airflow/exports"))


@dag(
    schedule="0 2 * * *",  # 02:00 every day, after the previous day is complete
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
    params={
        "day": Param(None, type=["null", "string"], description="YYYY-MM-DD; empty = the day before the run"),
        "lookback_days": Param(1, type="integer", minimum=0),
    },
    tags=["journey-clustering", "cdc"],
)
def journey_daily_extract():
    @task
    def extract() -> str:
        context = get_current_context()
        params = context["params"]
        if params.get("day"):
            day = date.fromisoformat(params["day"])
        else:
            day = (context["logical_date"] - timedelta(days=1)).date()
        dsn = Connection.get("cdc_postgres").get_uri().replace("postgres://", "postgresql://", 1)
        target = extract_day(
            dsn=dsn,
            day=day,
            output_root=OUTPUT_ROOT,
            table=os.environ.get("JC_SOURCE_TABLE", "public.events"),
            time_column=os.environ.get("JC_TIME_COLUMN", "client_time"),
            lookback_days=int(params["lookback_days"]),
        )
        return str(target.parent)

    @task
    def model_enabled() -> None:
        # The model needs the repo and its Python deps on the worker; turn this
        # on with JC_RUN_MODEL=1 once the worker has them.
        if os.environ.get("JC_RUN_MODEL") != "1":
            raise AirflowSkipException("JC_RUN_MODEL is not 1; export only")

    export_dir = extract()
    run_model = BashOperator(
        task_id="run_full_pipeline",
        bash_command=(
            "cd ${JC_REPO_DIR:-/opt/journey-clustering} && "
            "RAW_INPUT={{ ti.xcom_pull(task_ids='extract') }} "
            "RUN_NAME=daily_{{ ds_nodash }} "
            "scripts/run_full_pipeline.sh"
        ),
    )
    export_dir >> model_enabled() >> run_model


journey_daily_extract()
