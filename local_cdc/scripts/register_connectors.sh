#!/usr/bin/env bash
# Create or update the source and sink connectors through the Kafka Connect
# REST API (PUT /connectors/<name>/config is idempotent).
set -euo pipefail

CONNECT_URL="${CONNECT_URL:-http://localhost:8083}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../connectors" && pwd)"

for file in "$DIR"/mongo-source.json "$DIR"/postgres-sink.json; do
  name="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$file")"
  config="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["config"]))' "$file")"
  echo "-> $name"
  curl -fsS -X PUT -H 'Content-Type: application/json' \
    --data "$config" "$CONNECT_URL/connectors/$name/config" >/dev/null
done

sleep 3
for name in mongo-source postgres-sink; do
  curl -fsS "$CONNECT_URL/connectors/$name/status" | python3 -c '
import json, sys
s = json.load(sys.stdin)
tasks = ", ".join(t["state"] for t in s["tasks"]) or "no tasks yet"
print(s["name"] + ": connector=" + s["connector"]["state"] + " tasks=" + tasks)'
done
