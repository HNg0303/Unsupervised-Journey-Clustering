#!/usr/bin/env bash
# Zip android and ios <platform>_journeys_named.csv of one scores run into one archive.
#
# Usage:
#   bash scripts/zip_journeys_named.sh [RUN_DIR] [OUTPUT_ZIP]
#
#   RUN_DIR     default: output/scores/678/678_20260929_090904
#   OUTPUT_ZIP  default: <RUN_DIR>/<run name>_journeys_named.zip
#
# The archive keeps the platform folders (android/..., ios/...). Files over
# 4 GB need zip 3.0 or newer (zip64), which macOS ships. The CSVs hold
# customer_id and device_id: share the zip only where that data is allowed.
set -euo pipefail

RUN_DIR="${1:-output/scores/678/678_20260929_090904}"
RUN_DIR="${RUN_DIR%/}"
OUTPUT_ZIP="${2:-${RUN_DIR}/$(basename "${RUN_DIR}")_journeys_named.zip}"

files=()
for platform in android ios; do
  file="${platform}/${platform}_journeys_named.csv"
  if [[ ! -f "${RUN_DIR}/${file}" ]]; then
    echo "missing ${RUN_DIR}/${file}; run scripts/map_named_clusters.py first" >&2
    exit 1
  fi
  files+=("${file}")
done

output_abs="$(cd "$(dirname "${OUTPUT_ZIP}")" && pwd)/$(basename "${OUTPUT_ZIP}")"
temporary="${output_abs}.partial"
rm -f "${temporary}"
echo "zipping ${files[*]} from ${RUN_DIR}"
# -6 balances speed and size; -q quiet, sizes are printed below.
(cd "${RUN_DIR}" && zip -6 -q "${temporary}" "${files[@]}")
unzip -tq "${temporary}"
mv -f "${temporary}" "${output_abs}"

unzip -l "${output_abs}"
ls -lh "${output_abs}"
