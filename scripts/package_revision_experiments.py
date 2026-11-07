"""Validate and package generated notebooks for manual Kaggle runs."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package(output, stage_one=False):
    root = ROOT / "experiments/kaggle/revision"
    expected = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "spectral_code/revision").glob("*.py")}
    support = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "spectral_code/faithful_graph_baselines").glob("*.py")}
    manifest_name = "stage1_notebook_manifest.csv" if stage_one else "notebook_manifest.csv"
    with (root / manifest_name).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = root / row["notebook"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Notebook changed after generation: {path}")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path}#{index}", "exec")
        tree = ast.parse("".join(notebook["cells"][2]["source"]))
        assignments = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name)}
        if ast.literal_eval(assignments["_sources"]) != expected:
            raise ValueError(f"Stale embedded revision runtime: {path}")
        if "_legacy_sources" in assignments and ast.literal_eval(assignments["_legacy_sources"]) != support:
            raise ValueError(f"Stale baseline encoder support: {path}")
    extras = ["README.md", "RUN_ORDER.md", "REVIEWER_COVERAGE.md", "MANUSCRIPT_CORRECTIONS.md",
              "requirements-transformers.txt", manifest_name, "transfer_plan.csv", "focused_transfer_plan.csv",
              "EXPERIMENT_PROTOCOL.md", "FIRST_RUN.md", "stage1_run_queue.csv", "stage1_protocol.json"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [row["notebook"] for row in rows] + extras:
            archive.write(root / name, "revision/" + name)
    result = {"notebook_archive": str(output.resolve()), "notebooks": len(rows),
              "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "validation": "Every cell compiles; every embedded runtime/support file equals current tested source",
              "gpu_training_executed": False, "dataset_archives": []}
    data = ROOT.parent / "data/benchmarks"
    for name in ("codenet_revision_common_support", "atcoder_revision_common_support_v2", "bigclonebench_revision_common_support_v2"):
        path = data / (name + ".zip")
        if not path.exists():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        recorded = path.with_suffix(".sha256").read_text().split()[0]
        if digest.hexdigest() != recorded:
            raise ValueError(f"Dataset archive checksum mismatch: {path}")
        result["dataset_archives"].append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                                          "sha256": digest.hexdigest()})
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-one", action="store_true", help="Package only 29 seed-42 jobs and the validation collector")
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.stage_one), indent=2))
