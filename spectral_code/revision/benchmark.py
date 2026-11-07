"""Comparable measured inference and forward/backward costs on fixed support."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import canonical_spectra_loss
from .protocol import fingerprint
from .training import _loader, SessionBudgetReached


def benchmark_model(model, frame, packed, config, output, *, losses=None, clock=None, warmup=2, batches=None):
    if (batches is not None and batches < 1) or warmup < 0:
        raise ValueError("Positive measured batch count and nonnegative warmup required")
    device = next(model.parameters()).device
    # No optimizer updates: timing never modifies the learned checkpoint.
    loss_function = getattr(model, "compute_loss", canonical_spectra_loss)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    rows = frame.sample(frac=1, random_state=9173).head(256).reset_index(drop=True)
    results = {}
    was_training = model.training
    try:
        for phase in ("inference", "forward_backward"):
            model.train(phase == "forward_backward")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            times, counts = [], []
            warmup_loader = _loader(rows.head(warmup * config.batch_size), packed, config) if warmup else ()
            for left, right, _ in warmup_loader:
                if clock:
                    clock.check()
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                    model(left.to(device), right.to(device), None)
            for index, (left, right, labels) in enumerate(_loader(rows, packed, config)):
                if batches is not None and index >= batches:
                    break
                if clock:
                    clock.check()
                left, right, labels = left.to(device), right.to(device), labels.to(device)
                sync()
                started = time.perf_counter()
                with torch.set_grad_enabled(phase == "forward_backward"):
                    with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                        logits, auxiliary = model(left, right, None)
                        if phase == "forward_backward":
                            loss, _ = loss_function(logits, auxiliary, labels,
                                                     positive_class_weight=1.0, **(losses or {}))
                    if phase == "forward_backward":
                        loss.backward()
                        model.zero_grad(set_to_none=True)
                sync()
                elapsed = time.perf_counter() - started
                times.append(elapsed); counts.append(len(labels))
            if not times:
                raise ValueError("Not enough examples after warmup to measure timing")
            results[phase] = {"measured_batches": len(times), "pairs": sum(counts),
                              "seconds_per_pair": sum(times) / sum(counts),
                              "pairs_per_second": sum(counts) / sum(times),
                              "batch_seconds_median": float(np.median(times)),
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        results.update(status="complete", physical_batch=config.batch_size, effective_batch=config.effective_batch_size,
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                       torch_version=str(torch.__version__), warmup_batches=warmup,
                       support_hash=fingerprint(rows.iloc[:sum(counts)][["left_id", "right_id", "label"]].to_dict("records")),
                       scope="model compute after inputs reach device; excludes loading, transfers, optimizer step and checkpoint I/O")
    except SessionBudgetReached:
        results.update(status="incomplete_session_budget")
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def benchmark_checkpoint(checkpoint, frame, packed, config, output, *, model_factory,
                         model_config_factory, clock=None):
    from .training import _load_checkpoint
    if clock:
        try:
            clock.check()
        except SessionBudgetReached:
            result = {"status": "pending_session_budget", "continuation": "Reattach completed checkpoint and rerun timing"}
            Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
    state = _load_checkpoint(checkpoint, "cpu")
    if state["best_model"] is None:
        raise ValueError("A validation-selected checkpoint is required for timing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory(model_config_factory(**state["model_config"])).to(device)
    model.load_state_dict(state["best_model"])
    return benchmark_model(model, frame, packed, config, output, losses=state["losses"], clock=clock)
