"""Small helpers for observable, timed model-training stages."""

from __future__ import annotations

import logging
import platform
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide this module
    resource = None  # type: ignore[assignment]


def peak_memory_mb() -> float | None:
    """Return the process peak resident memory when the OS exposes it."""
    if resource is None:
        return None
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux reports KiB.
        if platform.system() == "Darwin":
            value /= 1024 * 1024
        else:
            value /= 1024
        return round(value, 1)
    except (AttributeError, OSError, ValueError):
        return None


@contextmanager
def timed_stage(
    logger: logging.Logger,
    stage: str,
    *,
    items: int | None = None,
    heartbeat_seconds: float = 30.0,
) -> Iterator[dict[str, object]]:
    """Log a stage start, heartbeat, and completion metrics.

    A heartbeat is intentionally elapsed-time only: estimators such as
    HDBSCAN do not expose a trustworthy percentage-complete callback.
    ``result`` is populated in-place when the context exits, which makes it
    suitable for persisting in a run manifest.
    """
    started = time.perf_counter()
    result: dict[str, object] = {
        "stage": stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(heartbeat_seconds):
            elapsed = time.perf_counter() - started
            memory = peak_memory_mb()
            logger.info(
                "stage=%s status=running elapsed=%.1fs%s",
                stage,
                elapsed,
                f" peak_rss_mb={memory:.1f}" if memory is not None else "",
            )

    memory = peak_memory_mb()
    logger.info(
        "stage=%s status=started%s%s",
        stage,
        f" items={items:,}" if items is not None else "",
        f" peak_rss_mb={memory:.1f}" if memory is not None else "",
    )
    thread = threading.Thread(target=heartbeat, name=f"timing-{stage}", daemon=True)
    thread.start()
    try:
        yield result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        thread.join(timeout=1.0)
        elapsed = time.perf_counter() - started
        result.setdefault("status", "completed")
        result["elapsed_seconds"] = round(elapsed, 3)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        memory = peak_memory_mb()
        if memory is not None:
            result["peak_rss_mb"] = memory
        rate = items / elapsed if items and elapsed > 0 else None
        logger.info(
            "stage=%s status=%s elapsed=%.1fs%s%s",
            stage,
            result["status"],
            elapsed,
            f" average_items_per_sec={rate:.1f}" if rate is not None else "",
            f" peak_rss_mb={memory:.1f}" if memory is not None else "",
        )
