"""Execute generated notebooks without requiring the Jupyter launcher."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path


def execute_notebook(path: Path, repo_root: Path) -> None:
    """Execute code cells in order and write stream/error outputs back to the notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    sys.path.insert(0, str(repo_root))

    for cell_index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        stdout = io.StringIO()
        cell["execution_count"] = cell_index
        cell["outputs"] = []
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(source, f"{path.name}#cell-{cell_index}", "exec"), namespace)
        except Exception as exc:
            text = stdout.getvalue()
            if text:
                cell["outputs"].append({"name": "stdout", "output_type": "stream", "text": text})
            cell["outputs"].append(
                {
                    "ename": exc.__class__.__name__,
                    "evalue": str(exc),
                    "output_type": "error",
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        text = stdout.getvalue()
        if text:
            cell["outputs"].append({"name": "stdout", "output_type": "stream", "text": text})

    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Execute the requested notebooks."""
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repo_root / "notebooks" / "1_data_exploration.ipynb",
        repo_root / "notebooks" / "2_session_based_eda.ipynb",
        repo_root / "notebooks" / "3_Tokenization.ipynb",
    ]
    for notebook_path in notebook_paths:
        print(f"Executing {notebook_path.relative_to(repo_root)}")
        execute_notebook(notebook_path, repo_root)
        print(f"Completed {notebook_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
