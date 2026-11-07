"""Resumable training: fixed validation selection and explicit final test release.

Checkpoint only at optimizer boundaries. A resumed run retains optimizer/scaler,
Python/NumPy/Torch/CUDA RNG, epoch permutation position, and validation history.
Time exhaustion produces an incomplete run, never a completed paper result.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .batching import CanonicalPairDataset, make_collate
from .data import CleanDataCorpus, pair_code_ids, verify_alignment, _is_gzip_file
from .model import RevisionModelConfig, RevisionPairModel, canonical_spectra_loss
from .protocol import RunConfig, choose_threshold, fingerprint, metrics, run_identity, runtime_forecast
from .io_utils import replace_file

LOSS_WEIGHTS = {
    "spectral_weight": .30, "auc_ranking_weight": .10, "hard_negative_weight": .20,
    "reconstruction_weight": .05, "topology_reconstruction_weight": .05,
    "spectral_variance_weight": .05, "graph_weight": .01,
}
ARMS = {
    "full": {}, "mean_max": {"pooling": "mean_max"},
    "attention": {"pooling": "attention"}, "projected": {"adjacency_mode": "projected"},
    "projected_ast": {"adjacency_mode": "projected", "prior_relation_indices": (0,)},
    "projected_ddg": {"adjacency_mode": "projected", "prior_relation_indices": (2,)},
    "density_heat": {"readout_mode": "eigenvalue_only"}, "energy_only": {"pooling": "energy_only"},
    "classification_only": {},
    "mean_max_classification_only": {"pooling": "mean_max"},
    "attention_classification_only": {"pooling": "attention"},
    "auxiliary_half": {}, "auxiliary_double": {},
    "without_spectral_and_hard_negative": {},
    **{f"without_{name.removesuffix('_weight')}": {} for name in LOSS_WEIGHTS},
}


def arm_settings(arm: str) -> tuple[dict, dict]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}; choose {tuple(ARMS)}")
    losses = dict(LOSS_WEIGHTS)
    if arm.endswith("classification_only"):
        losses = dict.fromkeys(losses, 0.0)
    elif arm in {"auxiliary_half", "auxiliary_double"}:
        scale = .5 if arm == "auxiliary_half" else 2.0
        losses = {key: scale * value for key, value in losses.items()}
    elif arm == "without_spectral_and_hard_negative":
        losses.update(spectral_weight=0.0, hard_negative_weight=0.0)
    elif arm.startswith("without_"):
        losses[arm.removeprefix("without_") + "_weight"] = 0.0
    return dict(ARMS[arm]), losses


class SessionBudgetReached(RuntimeError):
    pass


class SessionClock:
    def __init__(self, config: RunConfig, started: float | None = None):
        self.started = time.monotonic() if started is None else started
        self.deadline = self.started + config.session_hours * 3600 - config.reserve_minutes * 60

    def check(self):
        if time.monotonic() >= self.deadline:
            raise SessionBudgetReached("Session training/evaluation budget exhausted; resume the saved progress")


def atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    replace_file(temp, path)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(value: dict):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in value["cuda"]])


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_checkpoint(path, device):
    # Only load progress produced by this project and deliberately reattached
    # by the user; optimizer/RNG state requires the full trusted checkpoint.
    return torch.load(path, map_location=device, weights_only=False)


def _batches(size: int, batch_size: int, seed: int, epoch: int, skip: int = 0):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(size, generator=generator).tolist()
    return [order[i:i + batch_size] for i in range(0, size, batch_size)][skip:]


def _loader(frame, graphs, config, batches=None):
    if hasattr(graphs, "pair_loader"):
        return graphs.pair_loader(frame, config, batches)
    kwargs = {"batch_sampler": batches} if batches is not None else {"batch_size": config.batch_size, "shuffle": False}
    return DataLoader(CanonicalPairDataset(frame, graphs), num_workers=0, collate_fn=make_collate(config.max_nodes),
                      generator=torch.Generator().manual_seed(config.seed + 9187),
                      pin_memory=torch.cuda.is_available(), **kwargs)


@torch.no_grad()
def predict(model, frame, graphs, config, device, clock=None):
    model.eval()
    probabilities = []
    for left, right, labels in _loader(frame, graphs, config):
        if clock is not None:
            clock.check()
        with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            logits, _ = model(left.to(device), right.to(device), None)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probabilities)


def load_pair_data(clean_data: Path, *, require_disjoint: bool = True):
    metadata_path = clean_data / "metadata.json"
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    alignment = manifest.get("ddg_ast_alignment")
    if alignment:
        verify_alignment(clean_data, alignment)
    if require_disjoint:
        if manifest.get("protocol") != "revision_problem_disjoint_v1":
            raise ValueError("Revision runs require the audited problem-disjoint bundle. Build it with revision.prepare first.")
        if any(n for values in manifest.get("overlap", {}).values() for n in values.values()):
            raise ValueError("Bundle overlap audit is nonzero")
        if not manifest.get("overlap"):
            raise ValueError("Bundle has no disjointness audit")
    else:
        if manifest.get("protocol") != "revision_reference_audited_v1":
            raise ValueError("Reference runs require an audited derived benchmark, not an unfiltered original release")
        if not manifest.get("overlap") or any(n for group in manifest["overlap"].values() for n in group.values()):
            raise ValueError("Reference bundle overlap audit is absent or nonzero")
    corpus = CleanDataCorpus(clean_data)
    frames = {name: corpus.pairs(name) for name in ("train", "valid", "test")}
    for name, frame in frames.items():
        if set(frame.label.unique()) != {0, 1}:
            raise ValueError(f"{name}: both labels are required")
    digest = hashlib.sha256()
    import gzip
    opener = gzip.open if _is_gzip_file(corpus.pairs_path) else open
    with opener(corpus.pairs_path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    data_hash = fingerprint({"manifest": manifest, "pairs_file_sha256": digest.hexdigest()})
    return corpus, frames, data_hash


def prepare_data(clean_data: Path, config: RunConfig, *, require_disjoint: bool = True):
    corpus, frames, data_hash = load_pair_data(clean_data, require_disjoint=require_disjoint)
    graphs = corpus.load_graphs(pair_code_ids(*frames.values()), max_nodes=config.max_nodes)
    for name, frame in frames.items():
        usable = frame.left_id.isin(graphs) & frame.right_id.isin(graphs)
        if not usable.all():
            raise ValueError(f"{name}: {int((~usable).sum())} pairs missing graph endpoints; rebuild common support before comparing arms")
    return frames, graphs, data_hash


def train_session(frames: dict, graphs: dict, config: RunConfig, model_config: RevisionModelConfig,
                  losses: dict, work_dir: Path, *, data_hash: str, resume: Path | None = None,
                  clock=None, model_factory=RevisionPairModel, experiment_label: str = "unspecified",
                  data_group_hash: str | None = None, loss_function=None) -> dict:
    """Train one arm/seed for one session. Reading test metrics is a separate step."""
    clock = clock or SessionClock(config)
    accounted_at = time.monotonic()
    work_dir.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config, asdict(model_config), losses, data_hash)
    # Code changes invalidate optimizer continuation; never silently mix models.
    from .versioning import training_implementation_hash
    source_hash = training_implementation_hash()
    identity = fingerprint({"run": identity, "implementation": source_hash, "arm": experiment_label})
    progress = work_dir / "progress.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = model_factory(model_config).to(device)
    loss_function = loss_function or getattr(model, "compute_loss", canonical_spectra_loss)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    state = {"identity": identity, "epoch": 1, "batch": 0, "history": [], "bad_epochs": 0,
             "best_score": None, "best_model": None, "best_threshold": None,
             "seen": 0, "loss_sum": 0.0, "optimizer_steps": 0, "complete": False,
             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "total_parameters": sum(p.numel() for p in model.parameters()),
             "training": asdict(config), "model_config": asdict(model_config), "losses": losses,
             "data_hash": data_hash, "implementation_hash": source_hash,
             "data_group_hash": data_group_hash or data_hash,
             "arm": experiment_label,
             "training_validation_wall_seconds": 0.0,
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
             "device": str(device), "torch_version": str(torch.__version__)}
    checkpoint_path = progress if progress.exists() else resume
    if checkpoint_path is not None:
        loaded = _load_checkpoint(checkpoint_path, device)
        if loaded["identity"] != identity:
            raise ValueError("Resume identity mismatch: data, seed, architecture, losses, protocol or implementation changed")
        state = loaded
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
    now = time.monotonic()
    state["session_setup_wall_seconds"] = state.get("session_setup_wall_seconds", 0.0) + max(
        0.0, now - getattr(clock, "started", accounted_at))
    accounted_at = now  # model loading/data preparation are setup, not training compute
    last_checkpoint = time.monotonic()

    def save(status):
        nonlocal last_checkpoint, accounted_at
        now = time.monotonic()
        state["training_validation_wall_seconds"] += now - accounted_at
        accounted_at = now
        state["model"] = _cpu_state(model)
        state["optimizer"] = optimizer.state_dict()
        state["scaler"] = scaler.state_dict()
        state["rng"] = rng_state()
        atomic_save(state, progress)
        summary = {k: state[k] for k in ("identity", "epoch", "batch", "optimizer_steps", "complete", "history", "trainable_parameters")}
        summary.update({key: state.get(key) for key in ("best_epoch", "best_valid", "stop_reason", "arm", "training",
                                                        "model_config", "losses", "data_hash", "data_group_hash",
                                                        "implementation_hash", "gpu", "training_validation_wall_seconds")})
        summary["session_setup_wall_seconds"] = state["session_setup_wall_seconds"]
        summary.update(status=status, checkpoint=str(progress), test_evaluated=False,
                       continuation="Attach this output as a Kaggle input and set RESUME_CHECKPOINT to its progress.pt")
        deadline = getattr(clock, "deadline", None)
        summary["runtime_forecast"] = runtime_forecast(
            config.max_epochs, len(state["history"]), state["training_validation_wall_seconds"],
            None if deadline is None else deadline - time.monotonic())
        (work_dir / "status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"status": status, "runtime_forecast": summary["runtime_forecast"]}), flush=True)
        last_checkpoint = time.monotonic()
        return summary

    train = frames["train"]
    positive = int(train.label.sum())
    positive_weight = min(10.0, (len(train) - positive) / max(1, positive))
    accumulation = config.effective_batch_size // config.batch_size
    total_batches = math.ceil(len(train) / config.batch_size)
    try:
        while state["epoch"] <= config.max_epochs and not state["complete"]:
            clock.check()
            model.train(); optimizer.zero_grad(set_to_none=True)
            batches = _batches(len(train), config.batch_size, config.seed, state["epoch"], state["batch"])
            loader = iter(_loader(train, graphs, config, batches))
            while state["batch"] < total_batches:
                clock.check()
                group_size = min(accumulation, total_batches - state["batch"])
                group_examples = min(config.effective_batch_size, len(train) - state["batch"] * config.batch_size)
                for _ in range(group_size):
                    left, right, labels = next(loader)
                    labels = labels.to(device)
                    with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                        logits, auxiliary = model(left.to(device), right.to(device), None)
                        loss, _ = loss_function(logits, auxiliary, labels,
                                                       positive_class_weight=positive_weight, **losses)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss; this run cannot be reported as complete")
                    scaler.scale(loss * len(labels) / group_examples).backward()
                    state["loss_sum"] += float(loss.detach().cpu()) * len(labels)
                    state["seen"] += len(labels)
                    state["batch"] += 1
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient norm")
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                state["optimizer_steps"] += 1
                if time.monotonic() - last_checkpoint >= config.checkpoint_minutes * 60:
                    save("training")
            # Persist the epoch before validation: validation can be retried on
            # resume without repeating updates if the session ends here.
            save("validation_pending")
            p = predict(model, frames["valid"], graphs, config, device, clock)
            threshold, result = choose_threshold(frames["valid"].label.to_numpy(), p, config.selection_metric)
            score = (result[config.selection_metric], result["F1"] if config.selection_metric == "Accuracy" else result["Accuracy"])
            if state["best_score"] is None or score > tuple(state["best_score"]):
                state.update(best_score=score, best_model=_cpu_state(model), best_threshold=threshold,
                             best_epoch=state["epoch"], best_valid=result, bad_epochs=0)
            else:
                state["bad_epochs"] += 1
            state["history"].append({"epoch": state["epoch"], "train_loss": state["loss_sum"] / max(1, state["seen"]),
                                     "valid": result, "optimizer_steps": state["optimizer_steps"]})
            print(json.dumps(state["history"][-1]), flush=True)
            converged = state["epoch"] >= config.min_epochs and state["bad_epochs"] >= config.patience
            state["epoch"] += 1; state["batch"] = 0; state["seen"] = 0; state["loss_sum"] = 0.0
            state["complete"] = converged or state["epoch"] > config.max_epochs
            state["stop_reason"] = "validation_patience" if converged else "maximum_epochs" if state["complete"] else None
            save("complete" if state["complete"] else "training")
    except SessionBudgetReached:
        return save("incomplete_session_budget")
    return save("complete")


def evaluate_final(checkpoint: Path, frames: dict, graphs: dict, config: RunConfig, output: Path,
                   *, expected_data_hash: str, clock=None, model_factory=RevisionPairModel,
                   model_config_factory=RevisionModelConfig) -> dict:
    """Explicit release of test results after all model/protocol decisions are fixed."""
    output.mkdir(parents=True, exist_ok=True)
    state = _load_checkpoint(checkpoint, "cpu")
    if not state["complete"] or state["best_model"] is None:
        raise ValueError("Training is incomplete; resume it before releasing test metrics")
    if state["data_hash"] != expected_data_hash:
        raise ValueError("Final-evaluation data differs from the training protocol")
    if state["training"] != asdict(config):
        # Session duration may change across continuation jobs; scientific
        # settings and identity remain checked during training continuation.
        old, new = dict(state["training"]), asdict(config)
        for key in ("session_hours", "reserve_minutes", "checkpoint_minutes"):
            old.pop(key, None); new.pop(key, None)
        if old != new:
            raise ValueError("Final-evaluation configuration mismatch")
    result_path = output / "final_test.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["identity"] != state["identity"]:
            raise ValueError("Existing test result belongs to a different experiment")
        return result
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory(model_config_factory(**state["model_config"])).to(device)
    model.load_state_dict(state["best_model"])
    started = time.monotonic()
    try:
        p = predict(model, frames["test"], graphs, config, device, clock)
    except SessionBudgetReached:
        return {"identity": state["identity"], "status": "final_test_pending_session_budget",
                "complete": True, "test_evaluated": False, "checkpoint": str(checkpoint)}
    result = {"identity": state["identity"], "best_epoch": state["best_epoch"],
              "arm": state["arm"], "seed": config.seed,
              "data_hash": state["data_hash"], "implementation_hash": state["implementation_hash"],
              "data_group_hash": state["data_group_hash"],
              "model_config": state["model_config"], "losses": state["losses"],
              "pair_counts": {name: len(frame) for name, frame in frames.items()},
              "stop_reason": state["stop_reason"], "test": metrics(frames["test"].label.to_numpy(), p, state["best_threshold"]),
              "inference_seconds": time.monotonic() - started,
              "trainable_parameters": state["trainable_parameters"], "protocol": state["training"]}
    result["training_validation_wall_seconds"] = state["training_validation_wall_seconds"]
    result["session_setup_wall_seconds"] = state.get("session_setup_wall_seconds", 0.0)
    result["gpu"] = state["gpu"]
    predictions = frames["test"].copy()
    predictions["probability"] = p
    predictions["threshold"] = state["best_threshold"]
    predictions.to_csv(output / "test_predictions.csv.gz", index=False, compression="gzip")
    by_language = []
    if "configuration_id" in predictions:
        for name, group in predictions.groupby("configuration_id"):
            by_language.append({"configuration_id": name, **metrics(group.label, group.probability, state["best_threshold"])})
    pd.DataFrame(by_language).to_csv(output / "test_by_language.csv", index=False)
    # Match the persisted JSON schema on the first call too (e.g. tuple fields
    # in model configuration become lists), so cached evaluations are identical.
    result = json.loads(json.dumps(result))
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
