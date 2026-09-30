"""Partition identity and Parquet storage primitives.

These helpers deliberately know nothing about journey semantics or model
training.  They provide deterministic partition identities and bounded storage
operations used by the aggregate pipelines and repository commands.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd


def require_pyarrow() -> tuple[Any, Any, Any]:
    """Import Parquet dependencies lazily with an actionable error."""
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "Parquet support requires pyarrow. Install the pipeline dependencies "
            "with `pip install -e '.[pipeline]'`."
        ) from exc
    return pa, ds, pq


def stable_session_bucket(platform: object, session_id: object, bucket_count: int) -> int:
    """Return a deterministic bucket for one logical production session."""
    if bucket_count < 1:
        raise ValueError("bucket_count must be positive")
    material = f"{str(platform).strip().lower()}\x1f{str(session_id).strip()}".encode()
    digest = hashlib.sha1(material).digest()
    return int.from_bytes(digest[:8], "big") % bucket_count


def safe_partition_id(value: str) -> str:
    """Make a stable, filesystem- and ID-friendly partition name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))
    return cleaned.strip("_") or "partition"


def parquet_files(root: Path) -> list[Path]:
    """Find Parquet files below a file or directory in deterministic order."""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() == ".parquet" else []
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write one dataframe without relying on pandas' optional engine lookup."""
    pa, _, pq = require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def parquet_dataset(root: Path) -> Any:
    """Return a PyArrow dataset for a Parquet file or directory."""
    _, ds, _ = require_pyarrow()
    paths = parquet_files(root)
    if not paths:
        raise FileNotFoundError(f"no parquet files found below {root}")
    return ds.dataset([str(path) for path in paths], format="parquet")
