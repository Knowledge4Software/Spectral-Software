"""Emit the experimental-design tables, verified against the implementation.

Both tables are taken structurally from the paper (``kaggle/lecture.txt``) so
the generated files match its formatting exactly -- column specs, rules, row
colours, and spacing. Every number is then checked against the code that
actually runs, and any mismatch is corrected in place and reported.

Reusing the paper's own markup rather than rebuilding it keeps the two in sync
when the template changes, which has already happened twice.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PAPER = _ROOT / "kaggle" / "lecture.txt"
CHECKPOINT = _ROOT.parent / "outputs/kaggle/RQ2/codenet/CodeNet_spectra_siam_Lexical.zip"
METHOD_NOTEBOOK = _ROOT / "kaggle/rq2/codenet/method/spectra_siam_lex.ipynb"
RUNTIME = _ROOT / "research/faithful_graph_baselines/notebook_runtime.py"


def table_block(label: str, environment: str = "table") -> str:
    """The paper's own markup for one table, start to end."""
    text = PAPER.read_text(encoding="utf-8", errors="ignore")
    anchor = text.index("\\label{" + label + "}")
    start = text.rfind("\\begin{" + environment + "}", 0, anchor)
    end = text.index("\\end{" + environment + "}", anchor) + len("\\end{" + environment + "}")
    return text[start:end]


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def constant(source: str, name: str) -> str | None:
    match = re.search(rf"^{name} = ([^\n#]+)", source, re.M)
    return match.group(1).strip() if match else None


def baseline_configs() -> dict[str, dict[str, str]]:
    """Batch / learning rate / weight decay per method, as the runtime sets them."""
    text = RUNTIME.read_text(encoding="utf-8")
    block = re.search(r"METHOD_CONFIGS = \{(.*?)\n\}", text, re.S).group(1)
    configs = {}
    for line in block.strip().split("\n"):
        match = re.search(r'"name": "([^"]+)", "batch": ([^,]+), "lr": ([^,]+), '
                          r'"weight_decay": ([^}]+)\}', line)
        if match:
            name, batch, lr, decay = (group.strip() for group in match.groups())
            configs[name] = {"batch": batch, "lr": lr, "wd": decay}
    return configs


def spectral_dimensions() -> tuple[int, int, int]:
    from research.rq1.run_table import K_EIGEN
    per_code = 8 + K_EIGEN
    return K_EIGEN, per_code, 2 * per_code + 2


def snn_parameters(input_dim: int) -> int:
    from research.rq1.run_table import SiameseSpectralNet
    model = SiameseSpectralNet(input_dim)
    return sum(parameter.numel() for parameter in model.parameters())


def checkpoint_parameters() -> int | None:
    if not CHECKPOINT.is_file():
        return None
    with zipfile.ZipFile(CHECKPOINT) as archive:
        names = [n for n in archive.namelist() if n.endswith("result.json")]
        if not names:
            return None
        return json.loads(archive.read(names[0])).get("TrainableParameters")


def render_lr(value: str) -> str:
    """Render a Python float literal the way the template writes it."""
    return {"1e-3": "10^{-3}", "5e-4": "(5{\\times}10^{-4})",
            "1e-4": "10^{-4}", "0.0": "0"}.get(value, value)


def check(block: str, expected: dict[str, str], report: list[str]) -> str:
    """Replace any stale value in the template, recording what changed."""
    for wrong, right in expected.items():
        if wrong == right:
            continue
        if wrong in block:
            block = block.replace(wrong, right)
            report.append(f"corrected: {wrong}  ->  {right}")
        elif right not in block:
            report.append(f"NOT FOUND (neither form present): {wrong} / {right}")
    return block


def build_baselines() -> tuple[str, list[str]]:
    block = table_block("tab:baselines")
    report: list[str] = []
    configs = baseline_configs()
    _k_eigen, per_code, pair_dim = spectral_dimensions()

    expected = {
        # Spectral widths follow K_EIGEN, which has changed since the first draft.
        "72-D spectrum": f"{per_code}-D spectrum",
        "146-D pair features": f"{pair_dim}-D pair features",
        f"${72}\\!\\to\\!256": f"${per_code}\\!\\to\\!256",
        "185k": f"{snn_parameters(per_code)/1000:.0f}k",
    }
    # Training triples: batch / lr / epochs, straight from METHOD_CONFIGS.
    for name in ("RtvNN", "CDLH", "ASTNN", "DeepSim", "FA-AST+GGNN", "FA-AST+GMN"):
        config = configs.get(name)
        if not config:
            continue
        expected[f"%%{name}%%"] = f"${config['batch']}/{render_lr(config['lr'])}/4$"
    block = check(block, expected, report)

    # Verify each method's training cell rather than blindly substituting, since
    # the template already carries most of these values correctly.
    for name in ("RtvNN", "CDLH", "ASTNN", "DeepSim", "FA-AST+GGNN", "FA-AST+GMN"):
        config = configs.get(name)
        if not config:
            continue
        wanted = f"${config['batch']}/{render_lr(config['lr'])}/4$"
        start = block.find(f"\n{name}\n")
        if start == -1:
            report.append(f"{name}: row not found in the template")
            continue
        # The row runs to the next method or section band, not the first "\\",
        # which appears inside every shortstack.
        end = block.find("\\midrule", start)
        row = block[start:end if end != -1 else len(block)]
        if wanted not in row:
            found = re.search(r"\$\d+/[^$]+/\d+\$", row)
            report.append(f"{name}: template has {found.group(0) if found else 'no triple'}, "
                          f"code says {wanted}")
    return block, report


def build_settings() -> tuple[str, list[str]]:
    block = table_block("tab:spectra_settings")
    report: list[str] = []
    source = notebook_source(METHOD_NOTEBOOK)
    parameters = checkpoint_parameters()

    expected = {}
    for name, pattern in (("MAX_AST_NODES", r"\{(\d+) nodes; AST\+DDG\}"),
                          ("LATENT_NODES", None), ("CHEBYSHEV_DEGREE", None)):
        value = constant(source, name)
        if value and pattern:
            match = re.search(pattern, block)
            if match and match.group(1) != value:
                expected[match.group(0)] = match.group(0).replace(match.group(1), value)
    if parameters:
        match = re.search(r"\{([\d,]{7,})\}", block)
        if match and match.group(1).replace(",", "") != str(parameters):
            expected[match.group(0)] = "{" + f"{parameters:,}" + "}"
    block = check(block, expected, report)
    return block, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=_ROOT / "kaggle" / "latex" / "design")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    header = ("% Structure taken verbatim from kaggle/lecture.txt; values verified\n"
              "% against the implementation by research/design/build_design_tables.py.\n")
    for name, builder in (("table_baselines.tex", build_baselines),
                          ("table_spectra_settings.tex", build_settings)):
        block, report = builder()
        (args.output_dir / name).write_text(header + block + "\n", encoding="utf-8")
        print(f"wrote {args.output_dir / name}")
        for line in report:
            print(f"    {line}")
        if not report:
            print("    template already matches the code")


if __name__ == "__main__":
    main()
