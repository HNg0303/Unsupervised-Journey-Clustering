#!/usr/bin/env bash
# Infer a new raw CSV folder with a frozen model (no retrain), then apply final names
# (taxonomy_id, cluster -1 = noise) from the model run's <platform>_named_clusters.csv.
#
# Usage:
#   scripts/infer_new_data.sh                                # data/giga_data/8910, model 678_20260929_090904
#   RAW_INPUT=data/giga_data/1112 scripts/infer_new_data.sh
#   MODEL_GROUP=678 MODEL_ID=678_20260929_090904 scripts/infer_new_data.sh
# Rerunning resumes: an existing partition is reused and scored parquet files are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src"
PYTHON="${PYTHON:-python}"

RAW_INPUT="${RAW_INPUT:-data/giga_data/8910}"
NAME="$(basename "$RAW_INPUT")"
MODEL_GROUP="${MODEL_GROUP:-678}"
MODEL_ID="${MODEL_ID:-678_20260929_090904}"
MODEL_RUN="output/partitioned_runs/$MODEL_GROUP/$MODEL_ID"
NAMES="output/scores/$MODEL_GROUP/$MODEL_ID"
RAW="${RAW:-data/lake/raw_events/$NAME/${NAME}_20261105}"
SCORES="output/scores/$NAME/$MODEL_ID"
LOGS="$SCORES/logs"
mkdir -p "$LOGS"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

if [[ ! -f "$RAW/partition_manifest.json" ]]; then
  log "[1/4] partition $RAW_INPUT -> $RAW"
  "$PYTHON" scripts/partition_raw_events.py --input "$RAW_INPUT" --output "$RAW" \
    --buckets 128 --platforms android ios 2>&1 | tee "$LOGS/01_partition.log"
else
  log "[1/4] reuse partition $RAW"
fi

for p in android ios; do
  log "[2/4] infer $p"
  "$PYTHON" scripts/score_partitioned_events.py --input "$RAW" --platform "$p" \
    --model-run "$MODEL_RUN/$p" --model-version "$MODEL_ID" --output-root "$SCORES" --resume \
    2>&1 | tee "$LOGS/02_infer_$p.log"

  log "[3/4] merge $p -> $SCORES/$p/${p}_scores.csv"
  "$PYTHON" - "$SCORES/$p" "$SCORES/$p/${p}_scores.csv" <<'PY'
import sys
from pathlib import Path
import pyarrow.csv as pacsv, pyarrow.parquet as pq
root, target = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(root.glob("model_version=*/platform=*/*.parquet"))
if not files:
    raise SystemExit(f"no scored parquet files below {root}")
schema = pq.read_schema(files[0]).remove_metadata()
tmp = target.with_name(f".{target.name}.tmp"); rows = 0
with pacsv.CSVWriter(tmp, schema) as w:
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=200_000):
            w.write_batch(b.select(schema.names)); rows += b.num_rows
tmp.replace(target); print(f"wrote {rows:,} rows from {len(files)} files to {target}")
PY

  log "[4/4] final names $p -> $SCORES/$p/${p}_journeys_named.csv"
  "$PYTHON" scripts/map_named_clusters.py --platform "$p" \
    --named "$NAMES/$p/${p}_named_clusters.csv" \
    --journeys "$SCORES/$p/${p}_scores.csv" \
    --output "$SCORES/$p/${p}_journeys_named.csv" 2>&1 | tee "$LOGS/04_names_$p.log"
done
log "done -> $SCORES"
