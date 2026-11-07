"""Notebook entry points for revised baseline and matched-transfer jobs."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .protocol import RunConfig, fingerprint
from .training import SessionClock, evaluate_final, load_pair_data, train_session
from .transfer import select_transfer_frames


def find_bundle(explicit=None, dataset="codenet"):
    if explicit is not None:
        return Path(explicit)
    candidates = []
    for path in Path("/kaggle/input").rglob("metadata.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("protocol") in {"revision_problem_disjoint_v1", "revision_reference_audited_v1"}:
                if dataset in metadata.get("dataset_key", "codenet"):
                    candidates.append(path.parent)
        except (ValueError, OSError):
            pass
    if len(candidates) != 1:
        raise ValueError(f"Attach exactly one {dataset} revision bundle, or set CLEAN_DATA_DIR: {candidates}")
    return candidates[0]


def run_external(*, method, seed, started, clean_data=None, session_hours=10.0,
                 max_epochs=100, min_epochs=10, patience=10, batch_size=8,
                 effective_batch_size=128, learning_rate=None, transfer_key=None,
                 matched_pairs=6000, resume_checkpoint=None, resume_tokens=None,
                 final_test=False, output_root=Path("/kaggle/working"),
                 dataset="codenet", model_revision=None, measure_cost=True):
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU for these full training jobs")
    is_text = method in {"codebert", "c4_adapted", "qwen3_lora"}
    if learning_rate is None:
        learning_rate = (5e-5 if method == "c4_adapted" else 2e-5) if is_text else {"astnn": 1e-3, "cdlh": 1e-3}.get(method, 5e-4)
    if transfer_key:
        max_epochs, min_epochs, patience = 30, 30, 1000
    config = RunConfig(dataset=dataset, seed=seed, selection_metric="F1" if dataset == "bigclonebench" else "Accuracy",
                       max_epochs=max_epochs, min_epochs=min_epochs, patience=patience,
                       batch_size=batch_size, effective_batch_size=effective_batch_size,
                       weight_decay=0.0 if method in {"astnn", "cdlh"} else 1e-3 if method == "deepsim" else 1e-4,
                       learning_rate=learning_rate, session_hours=session_hours)
    clock = SessionClock(config, started=started)
    root = find_bundle(clean_data, dataset)
    corpus, frames, data_hash = load_pair_data(root, require_disjoint=dataset == "codenet")
    group_hash = data_hash
    label = f"baseline/{method}"
    suffix = ""
    if transfer_key:
        frames, transfer_metadata = select_transfer_frames(frames, transfer_key, matched_pairs, seed)
        data_hash = fingerprint({"bundle": data_hash, "transfer": transfer_metadata})
        label += "/" + transfer_key
        suffix = "_" + transfer_key.replace("|", "_").replace("->", "_")
    output = Path(output_root) / f"revision_{dataset}_{method}{suffix}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if transfer_key:
        (output / "transfer_metadata.json").write_text(json.dumps(transfer_metadata, indent=2), encoding="utf-8")
    if is_text:
        from .transformer import TransformerConfig, TransformerPairModel, prepare_tokens
        settings = TransformerConfig(revision=model_revision)
        if method == "c4_adapted":
            settings = TransformerConfig(revision=model_revision, objective="c4_adapted")
        elif method == "qwen3_lora":
            settings = TransformerConfig(model_id="Qwen/Qwen3-Embedding-0.6B", revision=model_revision,
                                         pooling="last_token", lora_rank=16, objective="pair_bce_contrastive")
        packed, model_config, token_status = prepare_tokens(corpus, frames, settings, output / "tokens",
                                                            clock=clock, resume=resume_tokens)
        if packed is None:
            return {"complete": False, "status": "incomplete_tokenization", "token_status": token_status,
                    "continuation": "Attach output; set RESUME_TOKENS_DIR to its tokens directory"}
        factory, config_factory = TransformerPairModel, TransformerConfig
    else:
        from .legacy import LegacyConfig, prepare_legacy
        packed, model_config, factory = prepare_legacy(root, frames, method, config, output)
        config_factory = LegacyConfig
    summary = train_session(frames, packed, config, model_config, {}, output, data_hash=data_hash,
                            data_group_hash=group_hash, model_factory=factory, clock=clock,
                            resume=Path(resume_checkpoint) if resume_checkpoint else None, experiment_label=label)
    if measure_cost and summary["complete"]:
        from .benchmark import benchmark_checkpoint
        benchmark_checkpoint(output / "progress.pt", frames["train"], packed, config, output / "benchmark.json",
                             model_factory=factory, model_config_factory=config_factory, clock=clock)
    if final_test and summary["complete"]:
        return evaluate_final(output / "progress.pt", frames, packed, config, output,
                              expected_data_hash=data_hash, clock=clock, model_factory=factory,
                              model_config_factory=config_factory)
    return summary
