"""Add an idempotent, standalone tqdm cell counter to project notebooks."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "# SPECTRA notebook progress: begin\n"
END = "# SPECTRA notebook progress: end\n"


def setup_source(total):
    return START + f'''from tqdm.auto import tqdm

def _spectra_enable_cell_progress():
    from IPython import get_ipython
    _shell = get_ipython()
    if _shell is None:
        return
    _old = getattr(_shell, "_spectra_cell_progress", None)
    if _old is not None:
        _old[0].close()
        for _event, _callback in _old[1:]:
            _shell.events.unregister(_event, _callback)
    _bar = tqdm(total={total}, desc="Notebook cells", unit="cell", mininterval=2)
    def _after_cell(_result):
        if _bar.n < _bar.total:
            _bar.update(1)
        if getattr(_result, "error_in_exec", None) or getattr(_result, "error_before_exec", None):
            _bar.set_description("Notebook cell failed")
            _bar.close()
        elif _bar.n >= _bar.total:
            _bar.close()
    _shell.events.register("post_run_cell", _after_cell)
    _shell._spectra_cell_progress = (_bar, ("post_run_cell", _after_cell))

_spectra_enable_cell_progress()
''' + END + "\n"


def update_notebook(path):
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    if not code:
        return False
    if any(c.get("outputs") or c.get("execution_count") is not None for c in code):
        try:
            relative = path.resolve().relative_to(ROOT)
        except ValueError:
            relative = None
        if relative is not None:
            backup = ROOT / "output/revision/notebook_output_backups" / relative
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(path.read_bytes())
    setup_cell = next((c for c in code if not "".join(c["source"]).lstrip().startswith("%%")), None)
    if setup_cell is None:
        return False
    old = "".join(setup_cell["source"])
    source = old
    if START in source:
        begin = source.index(START)
        end = source.index(END, begin) + len(END)
        source = source[:begin] + source[end:].lstrip("\n")
    # Future imports must remain before executable setup; cell magics must
    # remain the first line of their own cells.
    prefix_lines = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
                prefix_lines = statement.end_lineno
    lines = source.splitlines(keepends=True)
    prefix = "".join(lines[:prefix_lines]).rstrip("\n") + "\n\n" if prefix_lines else ""
    body = "".join(lines[prefix_lines:]).lstrip("\n") if prefix_lines else source.lstrip("\n")
    source = prefix + setup_source(len(code)) + body
    changed = old != source or b"\r\n" in path.read_bytes()
    setup_cell["source"] = source.splitlines(keepends=True)
    # Execution output is a local artifact, not a source-controlled result.
    for cell in code:
        if cell.get("outputs") or cell.get("execution_count") is not None:
            changed = True
        cell["outputs"] = []
        cell["execution_count"] = None
    for index, cell in enumerate(code):
        # Shell/magic cells are valid IPython, not ordinary Python source.
        text = "".join(cell["source"])
        if not any(line.lstrip().startswith(("%", "!")) for line in text.splitlines()):
            compile(text, f"{path}#code{index}", "exec")
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return changed


def project_notebooks(root=ROOT):
    for directory in ("create_datasets_graphs", "experiments", "revision", "spectral_code/templates"):
        yield from (Path(root) / directory).rglob("*.ipynb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted(project_notebooks())
    print(f"Updated {sum(update_notebook(p) for p in paths)} of {len(paths)} notebooks")
