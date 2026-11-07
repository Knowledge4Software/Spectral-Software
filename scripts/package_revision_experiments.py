"""Validate and package generated notebooks for manual Kaggle runs."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package(output, stage_one=False):
    from scripts.revision_layout import REVISION_ROOT
    root = REVISION_ROOT
    expected = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "spectral_code/revision").glob("*.py")}
    support = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "spectral_code/faithful_graph_baselines").glob("*.py")}
    manifest_name = "stage1_notebook_manifest.csv" if stage_one else "notebook_manifest.csv"
    with (root / "queues" / manifest_name).open(encoding="utf-8", newline="") as stream:
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
    extras = ["README.md", "START_HERE_FA.md", "requirements-transformers.txt",
              *(p.relative_to(root).as_posix() for p in sorted((root / "00_plan").glob("*.md"))),
              *(p.relative_to(root).as_posix() for p in sorted((root / "datasets").glob("*")) if p.is_file()),
              "queues/" + manifest_name, "queues/transfer_plan.csv", "queues/focused_transfer_plan.csv",
              "queues/stage1_run_queue.csv", "queues/stage1_protocol.json",
              "queues/README.md", "queues/LAUNCH_24.md", "queues/launch_24.csv", "queues/launch_next.csv",
              "queues/path_migration.csv", "results/README.md", "results/run_progress.csv",
              *(p.relative_to(root).as_posix() for p in sorted(root.glob("0*/README.md"))),
              "optional/README.md"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in dict.fromkeys([row["notebook"] for row in rows] + extras):
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
