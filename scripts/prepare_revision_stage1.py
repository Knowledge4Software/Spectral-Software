"""Freeze the first screening queue and its dataset/configuration provenance.

Run after build_revision_notebooks.py. This performs no research training.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.revision_notebook_extras import EXTERNAL_METHODS
from spectral_code.revision.model import RevisionModelConfig
from spectral_code.revision.protocol import RunConfig
from spectral_code.revision.training import LOSS_WEIGHTS
from spectral_code.revision.transformer import TransformerConfig
from spectral_code.revision.numerics import POLICY
from spectral_code.revision.versioning import training_implementation_hash

DATASETS = {
    "codenet": ("codenet_revision_common_support", "ec86d556b37b32180c44be50b3d79336fabae97432fe004a0baaebc7b9e4b46f", (69538, 14896, 14906)),
    "atcoder": ("atcoder_revision_common_support_v2", "3525d3a750d395fd912da6879f96986dc4d9fa639fd78293c4c542c7d62d7318", (70000, 15000, 15000)),
    "bigclonebench": ("bigclonebench_revision_common_support_v2", "b6bc759554b73379f61d1c2e3f0de63addfd28d68f2dae91a9d494c342ce9bd5", (70000, 7129, 7566)),
}


def prepare(root=None):
    root = Path(root) if root else ROOT / "experiments/kaggle/revision"
    with (root / "notebook_manifest.csv").open(encoding="utf-8", newline="") as stream:
        manifest = {row["notebook"]: row for row in csv.DictReader(stream)}
    jobs = []
    settings = {}
    # Pilot hardware order: full model, one tree adaptation, one pretrained model.
    methods = ("astnn", "codebert", *(m for m in EXTERNAL_METHODS if m not in {"astnn", "codebert"}))
    for dataset, (bundle, digest, counts) in DATASETS.items():
        recorded = (ROOT.parent / "data/benchmarks" / (bundle + ".sha256")).read_text().split()[0]
        if recorded != digest:
            raise ValueError(f"Frozen dataset changed: {bundle}")
        for method in ("full", *methods):
            if dataset == "bigclonebench" and method == "c4_adapted":
                continue
            is_full = method == "full"
            if dataset == "codenet":
                relative = ("01_readout_controls/codenet_full_seed42.ipynb" if is_full else
                            f"05_baselines/codenet_{method}_seed42.ipynb")
            else:
                relative = (f"07_cross_benchmark/{dataset}/full_seed42.ipynb" if is_full else
                            f"07_cross_benchmark/{dataset}/baselines/{dataset}_{method}_seed42.ipynb")
            if relative not in manifest:
                raise FileNotFoundError(relative)
            text = method in {"codebert", "c4_adapted", "qwen3_lora"}
            batch = 32 if is_full else 4 if method in {"qwen3_lora", "fa_ast_gmn", "cdlh"} else 16 if method == "deepsim" else 8
            lr = 1e-4 if is_full else (5e-5 if method == "c4_adapted" else 2e-5) if text else {"astnn": 1e-3, "cdlh": 1e-3}.get(method, 5e-4)
            wd = 0.0 if method in {"astnn", "cdlh"} else 1e-3 if method == "deepsim" else 1e-4
            config = RunConfig(dataset=dataset, seed=42, selection_metric="F1" if dataset == "bigclonebench" else "Accuracy",
                               batch_size=batch, learning_rate=lr, weight_decay=wd)
            arm = "full" if is_full else f"baseline/{method}"
            job_id = f"{dataset}_{method}_seed42"
            jobs.append({"order": len(jobs) + 1, "job_id": job_id, "dataset": dataset, "arm": arm,
                         "seed": 42, "notebook": relative, "notebook_sha256": manifest[relative]["sha256"],
                         "primary_metric": config.selection_metric, "dataset_zip": bundle + ".zip",
                         "dataset_sha256": digest, "gpu": True, "internet": text,
                         "final_test": False})
            settings[job_id] = asdict(config)
    with (root / "stage1_run_queue.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(jobs[0]))
        writer.writeheader(); writer.writerows(jobs)
    selected = [manifest[job["notebook"]] for job in jobs]
    selected.append(manifest["08_analysis/validation_screening.ipynb"])
    with (root / "stage1_notebook_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    release = {"protocol": "spectra_revision_stage1_v1", "date": "2026-11-06",
               "training_implementation_hash": training_implementation_hash(),
               "numerical_policy": POLICY,
               "research_jobs": len(jobs), "seed": 42, "test_release": False,
               "tuning": "One prespecified default configuration per method; no hyperparameter search in this stage",
               "scientific_settings": settings, "full_model": asdict(RevisionModelConfig()),
               "full_losses": LOSS_WEIGHTS,
               "text_models": {
                   "codebert": asdict(TransformerConfig()),
                   "c4_adapted": asdict(TransformerConfig(objective="c4_adapted")),
                   "qwen3_lora": asdict(TransformerConfig(model_id="Qwen/Qwen3-Embedding-0.6B",
                                                          pooling="last_token", lora_rank=16,
                                                          objective="pair_bce_contrastive"))},
               "dataset_releases": {d: {"zip": b + ".zip", "sha256": h, "pair_counts": dict(zip(("train", "valid", "test"), c))}
                                    for d, (b, h, c) in DATASETS.items()},
               "pretrained_revisions": "Resolve at first download; copy tokens/tokens.json commit into MODEL_REVISION for continuation and later seeds",
               "remaining_hardware_gate": "No CUDA run validated locally; run CodeNet full seed42 first"}
    (root / "stage1_protocol.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(jobs)} seed-42 research jobs; start only {jobs[0]['notebook']}")
    return release


if __name__ == "__main__":
    prepare()
