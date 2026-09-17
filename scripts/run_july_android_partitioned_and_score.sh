#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-cluster}"
if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
elif [[ -x "/opt/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/opt/miniconda3/bin/conda"
else
    echo "ERROR: conda was not found." >&2
    exit 1
fi

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${CONDA_ENV}"
PYTHON_BIN="${PYTHON_BIN:-python}"

JOURNEYS_ROOT="${JOURNEYS_ROOT:-${REPO_ROOT}/data/lake/journeys/july}"
EVENTS_CSV="${EVENTS_CSV:-${REPO_ROOT}/data/giga_data/july/android_events_all-t7.csv}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/output/partitioned_runs/july_pca48_svd48_ngram12_1000}"
SCORES_ROOT="${SCORES_ROOT:-${REPO_ROOT}/output/scores/july}"
RUN_NAME="${RUN_NAME:-latest}"
MAX_TRAINING_JOURNEYS="${MAX_TRAINING_JOURNEYS:-1000000}"

MODEL_DIR="${MODEL_ROOT}/${RUN_NAME}/android"
SCORER_FILE="${MODEL_DIR}/android_journey_scorer.pkl"
LOG_FILE="${MODEL_ROOT}/${RUN_NAME}/android_training.log"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: file not found: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "ERROR: directory not found: $1" >&2
        exit 1
    fi
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
}
if ! "${PYTHON_BIN}" -c "import numpy, pandas" >/dev/null 2>&1; then
    echo "ERROR: Conda environment '${CONDA_ENV}' is missing project dependencies." >&2
    exit 1
fi
require_dir "${JOURNEYS_ROOT}/platform=android"
require_file "${EVENTS_CSV}"

cd "${REPO_ROOT}"

echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "[1/2] Training the July Android partitioned journey model..."
"${PYTHON_BIN}" scripts/run_partitioned_journey_pipeline.py \
    --journeys-root "${JOURNEYS_ROOT}" \
    --output-root "${MODEL_ROOT}" \
    --run-name "${RUN_NAME}" \
    --mode train \
    --platforms android \
    --max-training-journeys "${MAX_TRAINING_JOURNEYS}" \
    --ngram-min 1 \
    --ngram-max 2 \
    --svd-dim 48 \
    --global-pca-components 48 \
    --min-cluster-size 100 \
    --log-file "${LOG_FILE}"

require_file "${SCORER_FILE}"

echo "[2/2] Scoring all Android events..."
"${PYTHON_BIN}" scripts/score_partitioned_events.py \
    --input "${EVENTS_CSV}" \
    --platform android \
    --single-run "${MODEL_DIR}" \
    --model-version "${RUN_NAME}" \
    --output-root "${SCORES_ROOT}" \
    --parquet

echo "Done."
echo "Model: ${MODEL_DIR}"
echo "Scores: ${SCORES_ROOT}/android/model_version=${RUN_NAME}/platform=android"
