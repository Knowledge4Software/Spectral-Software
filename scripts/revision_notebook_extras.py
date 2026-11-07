"""Baseline, cross-benchmark and analysis companions for the revision builder."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_METHODS = ("astnn", "rtvnn", "cdlh", "deepsim", "fa_ast_ggnn", "fa_ast_gmn",
                    "codebert", "c4_adapted", "qwen3_lora")
REFERENCE_METHODS = EXTERNAL_METHODS


def extend(output, template, transfer_keys):
    from scripts.build_revision_notebooks import cell
    written = []
    def write(relative, notebook):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    screening = copy.deepcopy(template)
    screening["cells"][0] = cell("markdown", "# Validation-only screening summary\n\nAttach saved revision outputs. This CPU job reads only status.json; it does not open final tests or checkpoints. Incomplete runs are listed but not ranked.\n")
    screening["cells"][1] = cell("code", '''from pathlib import Path
INPUT_ROOT = Path("/kaggle/input")
QUEUE_CSV = None  # optional attached stage1_run_queue.csv for a planned/pending ledger
''')
    screening["cells"][3] = cell("code", '''from spectral_code.revision.screening import collect_validation
summary = collect_validation(INPUT_ROOT, Path("/kaggle/working/validation_screening"), QUEUE_CSV)
print(summary.to_string(index=False))
''')
    screening["metadata"]["kaggle"]["accelerator"] = "none"
    write(Path("08_analysis/validation_screening.ipynb"), screening)
    support = {p.name: p.read_text(encoding="utf-8") for p in
               (ROOT / "spectral_code/faithful_graph_baselines").glob("*.py")}
    support_bootstrap = f'''
_legacy_sources = {support!r}
_legacy_package = _runtime / "spectral_code" / "faithful_graph_baselines"
_legacy_package.mkdir(parents=True, exist_ok=True)
for _name, _source in _legacy_sources.items():
    (_legacy_package / _name).write_text(_source, encoding="utf-8")
'''
    for dataset in ("codenet", "atcoder", "bigclonebench"):
        methods = EXTERNAL_METHODS if dataset == "codenet" else (
            tuple(method for method in REFERENCE_METHODS if method != "c4_adapted") if dataset == "bigclonebench" else REFERENCE_METHODS)
        for method in methods:
            keys = (None, *transfer_keys) if dataset == "codenet" else (None,)
            for key in keys:
                for seed in (42, 123, 231):
                    notebook = copy.deepcopy(template)
                    is_text = method in {"codebert", "c4_adapted", "qwen3_lora"}
                    is_transfer = key is not None
                    notebook["cells"][0] = cell("markdown", f'''# Revised baseline: {method}, {dataset}, seed {seed}

One model/seed/condition per Save & Run, with a 10-hour session and 45-minute
output reserve. Training uses validation-based selection; final test remains
off while choosing the protocol. Time-limited jobs checkpoint and remain
incomplete; reattach progress.pt to continue. A 100-epoch ceiling does not
guarantee convergence or completion in one session.

Legacy encoders are the project's portable adaptations. C4 is explicitly a
task-masked contrastive adaptation of the authors' CodeBERT method, not an
exact reproduction (https://github.com/Chenning-Tao/C4). Qwen uses LoRA and
classification plus pair-contrastive training. CodeBERT is a symmetric Siamese
classifier. None of these is relabeled as FSD-CLCD, ZC3, or GraphCodeBERT.

Transfer jobs use 6,000 pairs and 30 epochs / 1,410 optimizer steps. Physical
batches differ for memory, so in-batch contrastive negatives are not equivalent
to a 128-example batch; report physical and effective sizes separately.
Main jobs use validation stopping. Learning rate is method-specific and editable;
if tuning it, compare prespecified candidates on validation only and freeze the
choice before final testing and running the remaining seeds.

Use the SAME derived common-support dataset for all compared methods. Missing
required graph views cause an error rather than silent test-pair exclusions.
For transformers enable Internet and install the optional requirements first.
''')
                    epochs, minimum, patience = (30, 30, 1000) if is_transfer else (100, 10, 10)
                    batch = 4 if method in {"qwen3_lora", "fa_ast_gmn", "cdlh"} else 16 if method == "deepsim" else 8
                    notebook["cells"][1] = cell("code", f'''from pathlib import Path
import time
SESSION_STARTED = time.monotonic()
METHOD = {method!r}
DATASET = {dataset!r}
SEED = {seed}
SESSION_HOURS = 10.0
MAX_EPOCHS = {epochs}
MIN_EPOCHS = {minimum}
PATIENCE = {patience}
BATCH_SIZE = {batch}
EFFECTIVE_BATCH_SIZE = 128
LEARNING_RATE = None  # method default; freeze validation-selected changes across seeds
TRANSFER_KEY = {key!r}
MATCHED_TRAIN_PAIRS = 6000
CLEAN_DATA_DIR = None
RESUME_CHECKPOINT = None
ALLOW_NUMERICAL_UPGRADE = False
RESUME_TOKENS_DIR = None
MODEL_REVISION = None  # resolved to a fixed model commit at first token-cache creation
FINAL_TEST = False
MEASURE_COST = True
''')
                    notebook["cells"][2] = cell("code", "".join(template["cells"][2]["source"]) + support_bootstrap)
                    run = '''import json
from spectral_code.revision.jobs import run_external
summary = run_external(method=METHOD, dataset=DATASET, seed=SEED, started=SESSION_STARTED,
                       clean_data=CLEAN_DATA_DIR, session_hours=SESSION_HOURS,
                       max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS, patience=PATIENCE,
                       batch_size=BATCH_SIZE, effective_batch_size=EFFECTIVE_BATCH_SIZE,
                       learning_rate=LEARNING_RATE, transfer_key=TRANSFER_KEY,
                       matched_pairs=MATCHED_TRAIN_PAIRS, resume_checkpoint=RESUME_CHECKPOINT,
                       resume_tokens=RESUME_TOKENS_DIR, model_revision=MODEL_REVISION,
                       final_test=FINAL_TEST, measure_cost=MEASURE_COST,
                       allow_numerical_upgrade=ALLOW_NUMERICAL_UPGRADE)
print(json.dumps(summary, indent=2))
'''
                    if is_text:
                        setup = '''# Kaggle already supplies CUDA PyTorch. Do not replace that installation.
import importlib.util
import subprocess
import sys
from packaging.version import Version
import importlib.metadata
_needs = []
if importlib.util.find_spec("transformers") is None or Version(importlib.metadata.version("transformers")) < Version("4.51"):
    _needs.append("transformers>=4.51,<5")
if METHOD == "qwen3_lora" and importlib.util.find_spec("peft") is None:
    _needs.extend(["peft>=0.15,<1", "accelerate>=1,<2"])
if _needs:
    if "transformers>=4.51,<5" not in _needs:
        _needs.append("transformers>=4.51,<5")
    subprocess.run([sys.executable, "-m", "pip", "install", *_needs], check=True)
'''
                        run = setup + run
                    notebook["cells"][3] = cell("code", run)
                    notebook["metadata"]["kaggle"]["isInternetEnabled"] = is_text
                    group = "05_baselines" if dataset == "codenet" and not key else "06_transfer_baselines" if key else f"07_cross_benchmark/{dataset}/baselines"
                    suffix = "_" + key.replace("|", "_").replace("->", "_") if key else ""
                    write(Path(group) / f"{dataset}_{method}{suffix}_seed{seed}.ipynb", notebook)
        if dataset != "codenet":
            for arm in ("full", "mean_max", "attention"):
                for seed in (42, 123, 231):
                    notebook = copy.deepcopy(template)
                    intro = f'''# {dataset}: {arm}, seed {seed}

Use the derived audited {dataset} common-support bundle, separately from the
original published benchmark. AtCoder carries source problem IDs. BigCloneBench
is a Java-only endpoint/source-disjoint control, not evidence of cross-language
generalization. Missing functionality/clone-type labels are not inferred.
Main training follows the same validation/checkpoint rules as CodeNet. Final
test remains off until choices are frozen. Never mix official-split and derived
common-support scores in the same row.
'''
                    notebook["cells"][0] = cell("markdown", intro)
                    config = "".join(notebook["cells"][1]["source"])
                    config = config.replace("ARM = 'full'", f"ARM = {arm!r}").replace("SEED = 42", f"SEED = {seed}")
                    notebook["cells"][1] = cell("code", config)
                    run = "".join(notebook["cells"][3]["source"])
                    start = run.index("if CLEAN_DATA_DIR is None:")
                    end = run.index("CLEAN_DATA_DIR = Path(CLEAN_DATA_DIR)", start)
                    run = run[:start] + f'from spectral_code.revision.jobs import find_bundle\nCLEAN_DATA_DIR = find_bundle(CLEAN_DATA_DIR, {dataset!r})\n' + run[end:]
                    run = run.replace('RunConfig(seed=SEED,', f'RunConfig(dataset={dataset!r}, seed=SEED,')
                    if dataset == "bigclonebench":
                        run = run.replace('selection_metric="Accuracy"', 'selection_metric="F1"')
                    run = run.replace("prepare_data(CLEAN_DATA_DIR, run_config)", "prepare_data(CLEAN_DATA_DIR, run_config, require_disjoint=False)")
                    run = run.replace('f"revision_{ARM}_seed{SEED}"', f'f"revision_{dataset}_{{ARM}}_seed{{SEED}}"')
                    notebook["cells"][3] = cell("code", run)
                    write(Path("07_cross_benchmark") / dataset / f"{arm}_seed{seed}.ipynb", notebook)
    diagnostics = copy.deepcopy(template)
    diagnostics["cells"][0] = cell("markdown", "# Validation spectral diagnostics\n\nAttach a completed main full-model checkpoint and its exact dataset. This job reads validation only. It exports clone/nonclone distance distributions, size/density-matched summaries and marginal slot profiles. These are empirical diagnostics, not a proof of language invariance.\n")
    diagnostics["cells"][1] = cell("code", '''from pathlib import Path
import time
SESSION_STARTED = time.monotonic()
SESSION_HOURS = 10.0
CLEAN_DATA_DIR = None
CHECKPOINT = None  # required attached main-model progress.pt
MAXIMUM_PAIRS = 4000
''')
    diagnostics["cells"][3] = cell("code", '''import json
from spectral_code.revision.protocol import RunConfig
from spectral_code.revision.training import SessionClock
from spectral_code.revision.jobs import find_bundle
from spectral_code.revision.diagnostics import validation_diagnostics
if CHECKPOINT is None:
    raise ValueError("Set CHECKPOINT to an attached validation-selected full-model progress.pt")
clock = SessionClock(RunConfig(session_hours=SESSION_HOURS), started=SESSION_STARTED)
print(json.dumps(validation_diagnostics(find_bundle(CLEAN_DATA_DIR), CHECKPOINT,
                                       Path("/kaggle/working/validation_diagnostics"), clock=clock,
                                       maximum_pairs=MAXIMUM_PAIRS), indent=2))
''')
    write(Path("08_analysis/validation_spectral_diagnostics.ipynb"), diagnostics)
    statistics = copy.deepcopy(diagnostics)
    statistics["cells"][0] = cell("markdown", "# Paired shared-problem uncertainty\n\nAttach completed final-test outputs only after protocol freeze. Both methods need the same conditions and at least three shared seeds. Resampling shares problem weights across paths and methods; paths are not treated as independent. No significance p-value is inferred.\n")
    statistics["cells"][1] = cell("code", '''from pathlib import Path
import time
SESSION_STARTED = time.monotonic()
SESSION_HOURS = 10.0
INPUT_ROOT = Path("/kaggle/input")
RESUME_BOOTSTRAP = None  # attached bootstrap_progress.json, if needed
METHOD_A = "full"
METHOD_B = "baseline/astnn"
REPETITIONS = 2000
METRIC = "Accuracy"
''')
    statistics["cells"][3] = cell("code", '''import json
import shutil
from spectral_code.revision.protocol import RunConfig
from spectral_code.revision.training import SessionClock
from spectral_code.revision.statistics import paired_comparison
output = Path("/kaggle/working/paired_uncertainty")
output.mkdir(parents=True, exist_ok=True)
if RESUME_BOOTSTRAP and not (output / "bootstrap_progress.json").exists():
    shutil.copyfile(RESUME_BOOTSTRAP, output / "bootstrap_progress.json")
clock = SessionClock(RunConfig(session_hours=SESSION_HOURS), started=SESSION_STARTED)
print(json.dumps(paired_comparison(INPUT_ROOT, METHOD_A, METHOD_B,
                                  output, repetitions=REPETITIONS, metric=METRIC, clock=clock), indent=2))
''')
    statistics["metadata"]["kaggle"]["accelerator"] = "none"
    write(Path("08_analysis/paired_problem_uncertainty.ipynb"), statistics)
    return written
