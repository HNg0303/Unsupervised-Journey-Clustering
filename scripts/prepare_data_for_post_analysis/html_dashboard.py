#!/usr/bin/env python3
"""Build static HTML from compact inference summary CSVs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_inference_html_dashboard import build, parse_args  # noqa: E402


if __name__ == "__main__":
    build(parse_args())
