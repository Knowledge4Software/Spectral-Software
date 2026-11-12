"""Build read-only Kaggle analysis jobs without rewriting frozen training jobs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.build_revision_notebooks import cell
from scripts.sync_notebook_progress import setup_source
from spectral_code.revision.checkpoint_review import CORE_ARMS, REVIEW_VERSION
from spectral_code.revision.protocol import RunConfig, fingerprint, run_identity
from spectral_code.revision.screening import INPUT_V2_HASH, VALIDATION_RECOVERY_HASH
from spectral_code.revision.versioning import TRAINING_SOURCES, training_implementation_hash

ROOT = Path(__file__).resolve().parents[1]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_manifest(root=ROOT):
    root = Path(root)
    output_root = root.parent / "outputs/revesion"
    index = {p.name: p for folder in ("01", "02", "03") for p in (output_root / folder).glob("*.zip")}
    desired = []
    for seed in (42, 123, 231):
        for arm in CORE_ARMS:
            name = ("codenet_classification_only_seed123_input_v2_validation_fix_restart_complete.zip"
                    if arm == "classification_only" and seed == 123 else
                    f"codenet_{arm}_seed{seed}_input_v2_complete.zip")
            desired.append((f"codenet_{arm}_seed{seed}", name))
    desired += [(f"atcoder_full_seed{seed}", f"atcoder_full_seed{seed}_input_v2" +
                 ("" if seed == 42 else "_validation_fix") + "_complete.zip") for seed in (42, 123, 231)]
    entries = []
    for job, name in desired:
        path = index[name]
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.endswith("/progress.pt")]
            if len(members) != 1:
                raise ValueError(f"Expected one project checkpoint in {path}")
            blob = archive.read(members[0])
            with torch.serialization.safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype,
                                                    type(np.dtype("uint32"))]):
                state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
            config = RunConfig(**state["training"])
            assert state["complete"] and state["best_model"] is not None
            assert f"{config.dataset}_{state['arm']}_seed{config.seed}" == job
            assert state["implementation_hash"] in {INPUT_V2_HASH, VALIDATION_RECOVERY_HASH}
            assert state["identity"] == fingerprint(dict(run=run_identity(config, state["model_config"], state["losses"], state["data_hash"]),
                implementation=state["implementation_hash"], arm=state["arm"]))
            for source in TRAINING_SOURCES:
                member = "revision_runtime/spectral_code/revision/" + source
                if member in archive.namelist() and source not in {"training.py", "numerics.py"}:
                    archived = archive.read(member).decode("utf-8").replace("\r\n", "\n")
                    assert archived == (root / "spectral_code/revision" / source).read_text(encoding="utf-8"), source
            entries.append(dict(job_id=job, dataset=config.dataset, arm=state["arm"], seed=config.seed,
                archive=name, archive_relative=path.relative_to(output_root).as_posix(),
                archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), checkpoint_member=members[0],
                checkpoint_sha256=hashlib.sha256(blob).hexdigest(), identity=state["identity"],
                data_hash=state["data_hash"], implementation_hash=state["implementation_hash"],
                model_config_hash=fingerprint(state["model_config"]), losses_hash=fingerprint(state["losses"]),
                training_config_hash=fingerprint(asdict(config)), original_selection_metric=config.selection_metric,
                best_epoch=state["best_epoch"]))
            del state, blob
    return entries


def prepare(root=ROOT, *, entries=None):
    root = Path(root)
    if training_implementation_hash() != VALIDATION_RECOVERY_HASH:
        raise ValueError("Frozen training implementation changed; review before packaging")
    entries = build_manifest(root) if entries is None else entries
    if len(entries) != 21 or len({e["job_id"] for e in entries}) != 21:
        raise ValueError("Manifest must contain 18 CodeNet plus three AtCoder checkpoints")
    expected = {f"codenet_{arm}_seed{seed}" for arm in CORE_ARMS for seed in (42, 123, 231)}
    expected |= {f"atcoder_full_seed{seed}" for seed in (42, 123, 231)}
    if {e["job_id"] for e in entries} != expected:
        raise ValueError("Unexpected checkpoint conditions; do not change the comparison roster")
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "spectral_code/revision").glob("*.py"))}
    bootstrap = '''# Reporting snapshot only; no checkout, pip install or Internet needed.
import sys
_sources = SOURCES_LITERAL
_runtime = Path("/kaggle/working/checkpoint_review_runtime")
_package = _runtime / "spectral_code" / "revision"
_package.mkdir(parents=True, exist_ok=True)
(_package.parent / "__init__.py").write_text("", encoding="utf-8")
for _name, _source in _sources.items():
    (_package / _name).write_text(_source, encoding="utf-8", newline="\\n")
sys.path.insert(0, str(_runtime))
from spectral_code.revision.progress import install_progress
install_progress()
'''.replace("SOURCES_LITERAL", repr(sources))
    jobs = []
    plans = [("01_atcoder_diagnosis", "atcoder", None, True)]
    plans += [(f"02_codenet_f1_seed{seed}", "codenet", seed, False) for seed in (42, 123, 231)]
    for name, dataset, seed, diagnostics in plans:
        batch = [e for e in entries if e["dataset"] == dataset and (seed is None or e["seed"] == seed)]
        title = ("AtCoder: diagnose existing best and last weights across three seeds" if diagnostics else
                 f"CodeNet seed {seed}: validation F1 for all six existing readout/objective conditions")
        config = setup_source(3) + '''from pathlib import Path
import time
SESSION_STARTED = time.monotonic()
SESSION_HOURS = 10.0  # work stops at 9h15, below Kaggle's 12h limit
INPUT_ROOT = Path("/kaggle/input")
CLEAN_DATA_DIR = None  # matching existing dataset; discover by dataset key
CHECKPOINT_PATHS = {}  # optional {job_id: '/kaggle/input/.../progress.pt'}; pinned bytes required
RESUME_REPORTS_DIR = None  # attach earlier analysis output after a budget pause
FINAL_TEST = False
'''
        run = f'''import json
from spectral_code.revision.jobs import find_bundle
from spectral_code.revision.checkpoint_review import run_checkpoint_review
ENTRIES = {batch!r}
if FINAL_TEST is not False:
    raise ValueError("This notebook is validation-only; FINAL_TEST must stay False")
summary = run_checkpoint_review(ENTRIES, find_bundle(CLEAN_DATA_DIR, {dataset!r}),
    Path("/kaggle/working/{name}"), input_root=INPUT_ROOT, diagnostics={diagnostics!r},
    session_hours=SESSION_HOURS, started=SESSION_STARTED, resume_reports=RESUME_REPORTS_DIR,
    checkpoint_paths=CHECKPOINT_PATHS)
print(json.dumps({{k:v for k,v in summary.items() if k != 'completed'}}, indent=2))
from IPython.display import FileLink, display
if summary.get("download"):
    display(FileLink(summary["download"]))
'''
        description = (f"# {title}\n\nAttach the matching **{dataset}** dataset and saved outputs for these "
                       f"**{len(batch)} pinned checkpoints**. GPU on; Internet off. No optimizer or training updates. "
                       "New F1 thresholds use validation at existing weights; no F1-best epoch is recovered. "
                       "Final tests stay off. Download only `_small_reports.zip`.\n\n" +
                       "\n".join(f"- `{e['job_id']}`: `{e['archive_relative']}`" for e in batch))
        jobs.append((name, dataset, batch, [cell("markdown", description), cell("code", config),
                                          cell("code", bootstrap), cell("code", run)]))
    collector_config = setup_source(3) + '''from pathlib import Path
INPUT_ROOT = Path("/kaggle/input")
FINAL_TEST = False
'''
    collector_run = f'''import json
from spectral_code.revision.checkpoint_review import collect_checkpoint_reviews
if FINAL_TEST is not False:
    raise ValueError("No final tests in this collector")
summary = collect_checkpoint_reviews(INPUT_ROOT, Path("/kaggle/working/checkpoint_review_collection"), {entries!r})
print(json.dumps(summary, indent=2))
from IPython.display import FileLink, display
display(FileLink(summary["download"]))
'''
    jobs.append(("03_collect_review_reports", "reports_only", [], [cell("markdown",
        "# Collect checkpoint-review reports\n\nRun on CPU after four GPU analysis jobs. Attach their saved outputs. "
        "No training dataset, training checkpoint or Internet required. Full summaries require all 18 CodeNet reports. "
        "BCB/231 provenance stays unresolved; no attempt is chosen automatically."),
        cell("code", collector_config), cell("code", bootstrap), cell("code", collector_run)]))
    manifest = dict(version=REVIEW_VERSION, runtime_hash=VALIDATION_RECOVERY_HASH, entries=entries,
                    limitations=["Post-hoc validation F1 calibration; epoch selection is preserved.",
                                 "BCB/231 provenance requires Kaggle notebook/version identities, not another training run."])
    _write(root / "revision/queues/checkpoint_review_manifest.json", json.dumps(manifest, indent=2) + "\n")
    notebook_rows, input_rows = [], []
    for name, dataset, batch, cells in jobs:
        path = root / f"revision/05_analysis/continuation/{name}.ipynb"
        notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                                   language_info=dict(name="python", version="3.13")), nbformat=4, nbformat_minor=5)
        for c in cells:
            c["id"] = fingerprint(dict(kind=c["cell_type"], source=c["source"]))[:12]
            if c["cell_type"] == "code":
                compile("".join(c["source"]), path.name, "exec")
        _write(path, json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        notebook_rows.append(dict(notebook=path.relative_to(root / "revision").as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), dataset=dataset, checkpoints=len(batch),
            gpu=dataset != "reports_only", internet=False, final_test=False))
        input_rows += [dict(notebook=name + ".ipynb", job_id=e["job_id"], archive=e["archive_relative"],
                           checkpoint_sha256=e["checkpoint_sha256"], dataset=dataset) for e in batch]
    for name, rows in (("checkpoint_review_notebooks.csv", notebook_rows), ("checkpoint_review_inputs.csv", input_rows)):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        _write(root / "revision/queues" / name, stream.getvalue())
    progress = root / "revision/results/checkpoint_review_progress.csv"
    if not progress.exists():
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["notebook", "status", "evidence", "test_evaluated", "notes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(dict(notebook=r["notebook"], status="ready_not_run" if r["gpu"] else "waiting_for_reports",
            evidence="", test_evaluated=False,
            notes="Read-only validation analysis; reuse pinned checkpoints" if r["gpu"] else "Run after the four GPU analysis outputs are available")
            for r in notebook_rows)
        _write(progress, stream.getvalue())
    print(f"Prepared four GPU inference jobs and one CPU collector; {len(entries)} existing checkpoints, zero training jobs.")
    return notebook_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    prepare(args.output_root)
