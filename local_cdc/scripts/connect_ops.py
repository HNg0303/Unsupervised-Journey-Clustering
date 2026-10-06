"""Register the Debezium connectors and keep them running.

Runs as the `connect-ops` service in docker-compose (or by hand from the host):

1. waits for the Kafka Connect REST API,
2. creates or updates every connector in connectors/*.json
   (PUT /connectors/<name>/config is idempotent),
3. with --watch, checks connector status every --interval seconds and restarts
   whatever is FAILED. The Debezium JDBC sink retries a lost PostgreSQL
   connection for a few minutes (flush.max.retries) and then gives up with a
   FAILED task; Kafka Connect never restarts it on its own.

Standard library only, so it runs in a plain python image.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "connectors"


def call(url: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def wait_for_connect(base: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            call(f"{base}/connectors")
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if time.time() > deadline:
                raise SystemExit(f"Kafka Connect not reachable at {base}")
            time.sleep(3)


def register(base: str, files: list[Path]) -> list[str]:
    names = []
    for path in files:
        spec = json.loads(path.read_text())
        call(f"{base}/connectors/{spec['name']}/config", "PUT", spec["config"])
        log(f"registered {spec['name']} ({path.name})")
        names.append(spec["name"])
    return names


def status_line(s: dict) -> str:
    tasks = ", ".join(t["state"] for t in s["tasks"]) or "no tasks yet"
    return f"{s['name']}: connector={s['connector']['state']} tasks={tasks}"


def heal(base: str, names: list[str]) -> None:
    for name in names:
        try:
            s = call(f"{base}/connectors/{name}/status")
        except urllib.error.HTTPError as e:
            log(f"{name}: status HTTP {e.code}")
            continue
        states = [s["connector"]["state"]] + [t["state"] for t in s["tasks"]]
        if "FAILED" in states:
            trace = next((t.get("trace", "") for t in s["tasks"] if t["state"] == "FAILED"), "")
            log(f"{status_line(s)} -> restarting. {trace.splitlines()[0] if trace else ''}")
            call(f"{base}/connectors/{name}/restart?includeTasks=true&onlyFailed=true", "POST")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connect-url", default="http://localhost:8083")
    p.add_argument("--connectors-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--watch", action="store_true", help="keep running and restart FAILED connectors/tasks")
    p.add_argument("--interval", type=int, default=30, help="seconds between checks with --watch")
    p.add_argument("--wait", type=int, default=300, help="seconds to wait for Kafka Connect")
    args = p.parse_args()

    base = args.connect_url.rstrip("/")
    wait_for_connect(base, args.wait)
    names = register(base, sorted(args.connectors_dir.glob("*.json")))

    time.sleep(5)
    for name in names:
        log(status_line(call(f"{base}/connectors/{name}/status")))

    while args.watch:
        time.sleep(args.interval)
        try:
            heal(base, names)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            log(f"Kafka Connect unreachable: {e}")


if __name__ == "__main__":
    main()
