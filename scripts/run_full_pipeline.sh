#!/usr/bin/env bash
# Reuse partitioned data -> train -> infer (all data) -> taxonomy naming for android and ios.
#
# By default the latest existing partition under data/lake/raw_events/<RUN_NAME>/
# (and its prepared journeys) is reused; a fresh partition is only built when
# none exists. Each invocation trains into a new model RUN_ID, so the data
# partition (DATA_ID) and the model run are tracked separately.
#
# Training drops boot/splash and chrome screens and samples at most
# MAX_TRAINING_JOURNEYS eligible journeys per platform to bound memory.
#
# Usage:
#   scripts/run_full_pipeline.sh
#   DATA_ID=678_20260928_150018 scripts/run_full_pipeline.sh
#   MAX_TRAINING_JOURNEYS=500000 scripts/run_full_pipeline.sh
#   RUN_ID=678_20260929_090000 SKIP_TRAIN=1 scripts/run_full_pipeline.sh
#
# Environment overrides:
#   RAW_INPUT              CSV file or directory of raw CSV exports (default: data/giga_data/678)
#   RUN_NAME               experiment group folder                  (default: basename of RAW_INPUT)
#   DATA_ID                partition folder to reuse under RUN_NAME (default: latest existing, else new)
#   RUN_ID                 model/scores run folder under RUN_NAME   (default: <RUN_NAME>_<timestamp>)
#   MAX_TRAINING_JOURNEYS  sampling cap per platform for training   (default: 1000000)
#   TAXONOMY               3-level taxonomy CSV or Markdown         (default: output/scores/taxonomy_naming/hifpt_journey_taxonomy_3_levels.csv)
#   PLATFORMS              space-separated platforms                (default: "android ios")
#   PYTHON                 python interpreter                       (default: python)
#   FORCE_PARTITION=1      re-partition RAW_INPUT into a new DATA_ID even if one exists
#   SKIP_TRAIN / SKIP_INFER / SKIP_NAMING=1 to reuse outputs of an earlier RUN_ID

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
RAW_INPUT="${RAW_INPUT:-data/giga_data/678}"
RUN_NAME="${RUN_NAME:-$(basename "${RAW_INPUT%.csv}")}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${RUN_NAME}_${TIMESTAMP}}"
MAX_TRAINING_JOURNEYS="${MAX_TRAINING_JOURNEYS:-1000000}"
TAXONOMY="${TAXONOMY:-output/scores/taxonomy_naming/hifpt_journey_taxonomy_3_levels.csv}"
read -r -a PLATFORM_LIST <<< "${PLATFORMS:-android ios}"

# Pick the partition to reuse: explicit DATA_ID, else the newest complete one.
RAW_EVENTS_ROOT="data/lake/raw_events/${RUN_NAME}"
if [[ -z "${DATA_ID:-}" && -z "${FORCE_PARTITION:-}" && -d "$RAW_EVENTS_ROOT" ]]; then
  latest_manifest="$(ls -1d "${RAW_EVENTS_ROOT}"/*/partition_manifest.json 2>/dev/null | sort | tail -n 1 || true)"
  [[ -n "$latest_manifest" ]] && DATA_ID="$(basename "$(dirname "$latest_manifest")")"
fi
DATA_ID="${DATA_ID:-${RUN_NAME}_${TIMESTAMP}}"

RAW_EVENTS="${RAW_EVENTS_ROOT}/${DATA_ID}"
JOURNEYS="data/lake/journeys/${RUN_NAME}/${DATA_ID}"
MODEL_ROOT="output/partitioned_runs/${RUN_NAME}"
MODEL_RUN="${MODEL_ROOT}/${RUN_ID}"
SCORES="output/scores/${RUN_NAME}/${RUN_ID}"
LOGS="${MODEL_RUN}/logs"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$LOGS"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

[[ -n "${SKIP_NAMING:-}" || -f "$TAXONOMY" ]] || { echo "TAXONOMY not found: $TAXONOMY" >&2; exit 1; }

log "run_id=${RUN_ID} data_id=${DATA_ID} platforms=${PLATFORM_LIST[*]} max_training_journeys=${MAX_TRAINING_JOURNEYS}"

# 1. Partition raw CSVs into session-safe parquet buckets (only when not already done).
if [[ -f "${RAW_EVENTS}/partition_manifest.json" ]]; then
  log "[1/4] reuse partition ${RAW_EVENTS}"
else
  [[ -e "$RAW_INPUT" ]] || { echo "RAW_INPUT not found: $RAW_INPUT" >&2; exit 1; }
  log "[1/4] partition ${RAW_INPUT} -> ${RAW_EVENTS}"
  "$PYTHON" scripts/partition_raw_events.py \
    --input "$RAW_INPUT" \
    --output "$RAW_EVENTS" \
    --platforms "${PLATFORM_LIST[@]}" \
    2>&1 | tee "${LOGS}/01_partition.log"
fi

# 2. Train one global model per platform on the partitioned journeys
#    (drop boot + chrome screens, cap the training sample, rest default).
#    Journey preparation is skipped when the journeys manifest already exists.
if [[ -z "${SKIP_TRAIN:-}" ]]; then
  if [[ -f "${JOURNEYS}/journey_manifest.json" ]]; then
    train_mode=train
    log "[2/4] reuse journeys ${JOURNEYS}; train -> ${MODEL_RUN}"
  else
    train_mode=all
    log "[2/4] prepare journeys -> ${JOURNEYS}; train -> ${MODEL_RUN}"
  fi
  "$PYTHON" scripts/run_partitioned_journey_pipeline.py \
    --input "$RAW_EVENTS" \
    --journeys-root "$JOURNEYS" \
    --output-root "$MODEL_ROOT" \
    --run-name "$RUN_ID" \
    --platforms "${PLATFORM_LIST[@]}" \
    --mode "$train_mode" \
    --max-training-journeys "$MAX_TRAINING_JOURNEYS" \
    --drop-boot \
    --log-file "${LOGS}/02_training_internal.log" \
    2>&1 | tee "${LOGS}/02_training.log"
fi

# 3. Score every journey in the partitioned dataset, then flatten to one CSV
#    per platform for the naming step.
if [[ -z "${SKIP_INFER:-}" ]]; then
  for platform in "${PLATFORM_LIST[@]}"; do
    log "[3/4] infer ${platform} -> ${SCORES}/${platform}"
    "$PYTHON" scripts/score_partitioned_events.py \
      --input "$RAW_EVENTS" \
      --platform "$platform" \
      --model-run "${MODEL_RUN}/${platform}" \
      --model-version "$RUN_ID" \
      --output-root "$SCORES" \
      2>&1 | tee "${LOGS}/03_inference_${platform}.log"

    log "[3/4] merge ${platform} parquet scores -> ${SCORES}/${platform}/${platform}_scores.csv"
    "$PYTHON" - "${SCORES}/${platform}" "${SCORES}/${platform}/${platform}_scores.csv" <<'PY'
import sys
from pathlib import Path

import pyarrow.csv as pacsv
import pyarrow.parquet as pq

root, target = Path(sys.argv[1]), Path(sys.argv[2])
# Partitions without journeys carry no typed columns; take the schema from a
# non-empty one so the merged CSV keeps every column.
files = [
    path for path in sorted(root.glob("model_version=*/platform=*/*.parquet"))
    if pq.ParquetFile(path).metadata.num_rows
]
if not files:
    raise SystemExit(f"no scored journeys below {root}")
schema = pq.read_schema(files[0]).remove_metadata()
tmp = target.with_name(f".{target.name}.tmp")
rows = 0
with pacsv.CSVWriter(tmp, schema) as writer:
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=200_000):
            writer.write_batch(batch.select(schema.names))
            rows += batch.num_rows
tmp.replace(target)
print(f"wrote {rows:,} rows from {len(files)} files to {target}")
PY
  done
fi

# 4. Name clusters with the 3-level business taxonomy (both platforms at once).
#    Names are written once per cluster (<platform>_cluster_mapping.csv), not into
#    every scored row; rows join to them on (model_version, platform, cluster_id).
if [[ -z "${SKIP_NAMING:-}" ]]; then
  log "[4/4] taxonomy naming -> ${SCORES}/taxonomy_naming"
  naming_args=()
  for platform in android ios; do
    naming_args+=(--"${platform}"-ngrams "${MODEL_RUN}/${platform}/${platform}_cluster_ngrams.csv")
  done
  for platform in "${PLATFORM_LIST[@]}"; do
    naming_args+=(
      --"${platform}"-input "${SCORES}/${platform}/${platform}_scores.csv"
      --"${platform}"-mapping-output "${SCORES}/${platform}/${platform}_cluster_mapping.csv"
      --"${platform}"-catalog-output "${SCORES}/${platform}/${platform}_taxonomy_shareholder_catalog.json"
    )
  done
  "$PYTHON" scripts/taxonomy_cluster_naming_pipeline.py \
    --taxonomy "$TAXONOMY" \
    --output-dir "${SCORES}/taxonomy_naming" \
    "${naming_args[@]}" \
    --overwrite \
    2>&1 | tee "${LOGS}/04_naming.log"
fi

log "done: data=${RAW_EVENTS} models=${MODEL_RUN} scores=${SCORES}"
