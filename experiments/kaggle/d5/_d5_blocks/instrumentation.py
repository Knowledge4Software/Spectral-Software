
# === Section 6.5: fine-grained timing instrumentation ===
#
# Wraps the model's own modules so every stage is timed without editing the
# model. Architecture, data, optimizer, and seed are untouched, so the metrics
# this run reports match the RQ2 run for this benchmark; only measurement is
# added.
#
# GPU work is asynchronous: perf_counter() around a CUDA call measures the
# kernel launch, not the kernel. Device stages are therefore timed with CUDA
# events, which record on the stream itself. Host stages use perf_counter.

import time as _time
from contextlib import contextmanager as _ctx

import numpy as _d5_np
import pandas as _d5_pd
import torch as _d5_torch

D5_DATASET = "__LABEL__"
D5_CUDA = _d5_torch.cuda.is_available()

D5_BATCH_ROWS = []
D5_EPOCH_ROWS = []

# predict() runs for validation too; it reads the current epoch from
# here so its rows can be grouped. Timing inference per batch adds a
# synchronize per batch, so it can be switched off if that cost matters.
D5_EPOCH_NOW = 0
D5_TIME_INFERENCE = True

_d5_current = {}
_d5_events = []

# Stage names follow the methodology from the input inwards. s06_to_s08 is the
# whole spectral block; s07 is the eigensolver inside it, reported separately
# because its cost is the hardest to predict. s07 is therefore NOT added to
# the measured total, or it would be counted twice.
D5_STAGES = [
    "h01_data_wait",
    "h02_host_to_device",
    "s01_embed_input",
    "s02_relation_gnn",
    "s03_slot_attention",
    "s04_latent_adjacency",
    "s05_latent_refine",
    "s06_to_s08_spectral",
    "s07_eigendecomposition",
    "s09_pair_features",
    "s10_classifier",
    "s11_loss",
    "s12_backward",
    "s13_optimizer_step",
]
D5_NESTED = {"s07_eigendecomposition"}


@_ctx
def d5_stage(name):
    """Time one stage of the batch currently being processed."""
    if D5_CUDA:
        start = _d5_torch.cuda.Event(enable_timing=True)
        end = _d5_torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        _d5_events.append((name, start, end))
    else:
        began = _time.perf_counter()
        yield
        _d5_current[name] = _d5_current.get(name, 0.0) + (
            _time.perf_counter() - began) * 1000.0


def d5_flush(epoch, phase, index, pairs, wall_ms):
    """Resolve the batch's CUDA events and append one row."""
    if D5_CUDA:
        # A CUDA event carries no readable time until the device reaches it.
        _d5_torch.cuda.synchronize()
        for name, start, end in _d5_events:
            _d5_current[name] = _d5_current.get(name, 0.0) + start.elapsed_time(end)
        _d5_events.clear()
    row = {"Epoch": epoch, "Phase": phase, "Batch": index, "Pairs": pairs,
           "WallMs": round(wall_ms, 4)}
    for stage in D5_STAGES:
        row[stage] = round(_d5_current.get(stage, 0.0), 4)
    measured = sum(row[s] for s in D5_STAGES if s not in D5_NESTED)
    row["MeasuredMs"] = round(measured, 4)
    row["UnattributedMs"] = round(max(wall_ms - measured, 0.0), 4)
    D5_BATCH_ROWS.append(row)
    _d5_current.clear()


def d5_instrument(model):
    """Wrap each component's forward so it reports its own time."""
    encoder = model.encoder

    def wrap(module, name):
        original = module.forward

        def timed(*args, **kwargs):
            with d5_stage(name):
                return original(*args, **kwargs)

        module.forward = timed

    for layer in encoder.graph_layers:
        wrap(layer, "s02_relation_gnn")
    wrap(encoder.slot_attention, "s03_slot_attention")
    # refinement is a ModuleList: wrapping the list itself would never fire,
    # because only the individual layers are ever called.
    _refinement = getattr(encoder, "refinement", None)
    if _refinement is not None:
        if isinstance(_refinement, _d5_torch.nn.ModuleList):
            for layer in _refinement:
                wrap(layer, "s05_latent_refine")
        else:
            wrap(_refinement, "s05_latent_refine")
    wrap(encoder.spectral, "s06_to_s08_spectral")
    wrap(model.classifier, "s10_classifier")

    wrap(encoder.canonical_embedding, "s01_embed_input")
    if getattr(encoder, "lexical_embedding", None) is not None:
        wrap(encoder.lexical_embedding, "s01_embed_input")
    wrap(encoder.input_norm, "s01_embed_input")

    wrap(encoder.adjacency_query, "s04_latent_adjacency")
    wrap(encoder.adjacency_key, "s04_latent_adjacency")

    original_pair = model._spectral_pair_features

    def timed_pair(*args, **kwargs):
        with d5_stage("s09_pair_features"):
            return original_pair(*args, **kwargs)

    model._spectral_pair_features = timed_pair

    original_eigvalsh = _d5_torch.linalg.eigvalsh

    def timed_eigvalsh(matrix, *args, **kwargs):
        with d5_stage("s07_eigendecomposition"):
            return original_eigvalsh(matrix, *args, **kwargs)

    _d5_torch.linalg.eigvalsh = timed_eigvalsh
    print("instrumented " + str(len(D5_STAGES)) + " stages on " + D5_DATASET
          + "; CUDA events: " + str(D5_CUDA))
    return model


def d5_assert_instrumented(minimum=9):
    """Fail loudly if the forward stages never recorded anything.

    A previous run defined the timers but never attached them, so
    every in-model stage reported zero while the run still finished
    cleanly. That is the failure this guards against.
    """
    if not D5_BATCH_ROWS:
        raise RuntimeError("no batches were timed at all")
    live = [s for s in D5_STAGES
            if any(r.get(s, 0) > 0 for r in D5_BATCH_ROWS)]
    dead = [s for s in D5_STAGES if s not in live]
    print("stages recording data: " + str(len(live)) + "/"
          + str(len(D5_STAGES)))
    if dead:
        print("  silent: " + ", ".join(dead))
    if len(live) < minimum:
        raise RuntimeError(
            "only " + str(len(live)) + " stages recorded data; expected at "
            "least " + str(minimum) + ". d5_instrument(model) was probably "
            "never called, so the timings would be meaningless.")
