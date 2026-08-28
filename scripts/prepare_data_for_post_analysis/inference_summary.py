#!/usr/bin/env python3
"""Canonical entrypoint for compact full-inference dashboard summaries.

The implementation remains in the tested DuckDB builder at
``scripts/extract_inference_dashboard_summaries.py`` for compatibility with
existing automation; this entrypoint places it in the post-analysis preparation
workflow and makes the intended command discoverable.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_inference_dashboard_summaries import build, parse_args  # noqa: E402


if __name__ == "__main__":
    build(parse_args())
