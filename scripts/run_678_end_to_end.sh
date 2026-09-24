#!/usr/bin/env bash
# End-to-end pipeline for every CSV below data/giga_data/678:
# CSV -> session-safe parquet -> journey preparation/training -> inference -> naming.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "/opt/miniconda3/envs/cluster/bin/python" ]]; then
  PYTHON_BIN="/opt/miniconda3/envs/cluster/bin/python"
else
  PYTHON_BIN="python"
fi
RUN_NAME="${RUN_NAME:-678_$(date +%Y%m%d_%H%M%S)}"
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/data/giga_data/678}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/lake/raw_events/678/${RUN_NAME}}"
JOURNEYS_ROOT="${JOURNEYS_ROOT:-${REPO_ROOT}/data/lake/journeys/678/${RUN_NAME}}"
TRAIN_ROOT="${TRAIN_ROOT:-${REPO_ROOT}/output/partitioned_runs/678}"
SCORES_ROOT="${SCORES_ROOT:-${REPO_ROOT}/output/scores/678/${RUN_NAME}}"
NAMING_SCRIPT="${NAMING_SCRIPT:-${REPO_ROOT}/output/scores/cluster_naming_pipeline.py}"
SITEMAP="${SITEMAP:-${REPO_ROOT}/output/scores/sitemap.csv}"

CHUNKSIZE="${CHUNKSIZE:-500000}"
BUCKETS="${BUCKETS:-256}"
MAX_TRAINING_JOURNEYS="${MAX_TRAINING_JOURNEYS:-1000000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
TEST_SIZE="${TEST_SIZE:-0.2}"

TRAIN_RUN_DIR="${TRAIN_ROOT}/${RUN_NAME}"
LOG_DIR="${TRAIN_RUN_DIR}/logs"
mkdir -p "${LOG_DIR}" "${SCORES_ROOT}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}" >&2
  exit 1
fi
if ! find "${INPUT_DIR}" -type f -iname '*.csv' -print -quit | grep -q .; then
  echo "No CSV files found below: ${INPUT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${NAMING_SCRIPT}" ]]; then
  echo "Naming script does not exist: ${NAMING_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${SITEMAP}" ]]; then
  echo "Sitemap does not exist: ${SITEMAP}" >&2
  exit 1
fi

echo "Run name: ${RUN_NAME}"
echo "Input:    ${INPUT_DIR}"
echo "Raw:      ${RAW_ROOT}"
echo "Journeys: ${JOURNEYS_ROOT}"
echo "Models:   ${TRAIN_RUN_DIR}"
echo "Scores:   ${SCORES_ROOT}"

echo "[1/4] Partitioning all CSV files into session-safe parquet..."
"${PYTHON_BIN}" scripts/partition_raw_events.py \
  --input "${INPUT_DIR}" \
  --output "${RAW_ROOT}" \
  --chunksize "${CHUNKSIZE}" \
  --buckets "${BUCKETS}" \
  --platforms android ios \
  2>&1 | tee "${LOG_DIR}/01_partition.log"

echo "[2/4] Preparing journeys and training Android/iOS models..."
TRAIN_ARGS=(
  --input "${RAW_ROOT}"
  --journeys-root "${JOURNEYS_ROOT}"
  --output-root "${TRAIN_ROOT}"
  --run-name "${RUN_NAME}"
  --mode all
  --platforms android ios
  --sample-seed "${SAMPLE_SEED}"
  --sampling-strategy customer_stratified
  --test-size "${TEST_SIZE}"
  --log-file "${LOG_DIR}/02_training_internal.log"
  --drop-boot
)
if [[ "${MAX_TRAINING_JOURNEYS}" != "all" && -n "${MAX_TRAINING_JOURNEYS}" ]]; then
  TRAIN_ARGS+=(--max-training-journeys "${MAX_TRAINING_JOURNEYS}")
fi
"${PYTHON_BIN}" scripts/run_partitioned_journey_pipeline.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/02_training.log"

echo "[3/4] Running inference over every raw parquet partition..."
AVAILABLE_PLATFORMS=()
for PLATFORM in android ios; do
  MODEL_DIR="${TRAIN_RUN_DIR}/${PLATFORM}"
  if [[ ! -f "${MODEL_DIR}/${PLATFORM}_journey_scorer.pkl" ]]; then
    echo "Skip ${PLATFORM}: no trained scorer found in ${MODEL_DIR}"
    continue
  fi
  AVAILABLE_PLATFORMS+=("${PLATFORM}")
  "${PYTHON_BIN}" scripts/score_partitioned_events.py \
    --input "${RAW_ROOT}" \
    --platform "${PLATFORM}" \
    --single-run "${MODEL_DIR}" \
    --model-version "${RUN_NAME}" \
    --output-root "${SCORES_ROOT}" \
    2>&1 | tee "${LOG_DIR}/03_inference_${PLATFORM}.log"
done

if [[ ${#AVAILABLE_PLATFORMS[@]} -eq 0 ]]; then
  echo "No Android or iOS scorer was produced; stopping before naming." >&2
  exit 1
fi

echo "[4/4] Consolidating inference partitions and applying cluster names..."
for PLATFORM in "${AVAILABLE_PLATFORMS[@]}"; do
  MODEL_DIR="${TRAIN_RUN_DIR}/${PLATFORM}"
  PARTITION_DIR="${SCORES_ROOT}/${PLATFORM}/model_version=${RUN_NAME}/platform=${PLATFORM}"
  CONSOLIDATED_CSV="${SCORES_ROOT}/${PLATFORM}/${PLATFORM}_scores.csv"
  NAMED_CSV="${SCORES_ROOT}/${PLATFORM}/${PLATFORM}_scores_named.csv"
  MAPPING_CSV="${SCORES_ROOT}/${PLATFORM}/${PLATFORM}_cluster_mapping.csv"
  CATALOG_JSON="${SCORES_ROOT}/${PLATFORM}/${PLATFORM}_shareholder_catalog.json"

  PARTITION_DIR="${PARTITION_DIR}" CONSOLIDATED_CSV="${CONSOLIDATED_CSV}" \
    "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

source = Path(os.environ["PARTITION_DIR"])
target = Path(os.environ["CONSOLIDATED_CSV"])
paths = sorted(path for path in source.glob("*.parquet") if path.is_file())
if not paths:
    raise SystemExit(f"No inference parquet files found in {source}")
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f".{target.name}.tmp")
try:
    for index, path in enumerate(paths):
        frame = pd.read_parquet(path)
        frame.to_csv(
            temporary,
            mode="w" if index == 0 else "a",
            header=index == 0,
            index=False,
        )
    temporary.replace(target)
finally:
    temporary.unlink(missing_ok=True)
print(f"Consolidated {len(paths)} parquet files into {target}")
PY

  "${PYTHON_BIN}" "${NAMING_SCRIPT}" \
    --platform "${PLATFORM}" \
    --training-run "${MODEL_DIR}" \
    --sitemap "${SITEMAP}" \
    --input "${CONSOLIDATED_CSV}" \
    --output "${NAMED_CSV}" \
    --mapping-output "${MAPPING_CSV}" \
    --catalog-output "${CATALOG_JSON}" \
    --overwrite \
    2>&1 | tee "${LOG_DIR}/04_naming_${PLATFORM}.log"
done

echo
echo "Pipeline completed successfully."
echo "Training artifacts: ${TRAIN_RUN_DIR}"
echo "Named inference:    ${SCORES_ROOT}/<platform>/<platform>_scores_named.csv"
