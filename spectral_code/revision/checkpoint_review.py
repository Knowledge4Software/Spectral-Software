"""Read-only checkpoint diagnostics and post-hoc validation F1 calibration.

This reporting module is deliberately outside TRAINING_SOURCES. It never creates
an optimizer, updates weights, evaluates test predictions, or rewrites a checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import gc
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .batching import pack_graphs
from .model import RevisionModelConfig, RevisionPairModel, canonical_spectra_loss
from .protocol import RunConfig, choose_threshold, fingerprint, metrics, run_identity
from .screening import INPUT_V2_HASH, VALIDATION_RECOVERY_HASH
from .training import (_load_checkpoint, _loader, load_pair_data, predict,
                       SessionBudgetReached, SessionClock)
from .versioning import training_implementation_hash

REVIEW_VERSION = "validation_checkpoint_review_v1"
CORE_ARMS = ("full", "mean_max", "attention", "classification_only",
             "mean_max_classification_only", "attention_classification_only")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def _finite_summary(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Diagnostic arrays must be nonempty and finite")
    return dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
                std=float(x.std()), q05=float(np.quantile(x, .05)), q95=float(np.quantile(x, .95)))


def pair_support(frame):
    columns = [c for c in ("pair_id", "left_id", "right_id", "label") if c in frame]
    if not {"left_id", "right_id", "label"} <= set(columns):
        raise ValueError("Pair support needs endpoints and labels")
    return fingerprint(frame[columns].astype(str).to_dict("records"))


def fixed_sample(frame, maximum, seed=9173):
    """Exact seeded class-stratified sample, independent of model predictions."""
    if maximum < 2 or set(frame.label.unique()) != {0, 1}:
        raise ValueError("Diagnostic samples require two classes and at least two pairs")
    if len(frame) <= maximum:
        return frame.reset_index(drop=True).copy()
    positive = max(1, min(maximum - 1, round(maximum * frame.label.mean())))
    counts = {1: min(positive, int(frame.label.eq(1).sum()))}
    counts[0] = min(maximum - counts[1], int(frame.label.eq(0).sum()))
    counts[1] = maximum - counts[0]
    result = pd.concat([frame[frame.label.eq(y)].sample(counts[y], random_state=seed)
                        for y in (0, 1)])
    # Retain the original frame order so every checkpoint sees the same batches.
    return result.sort_index().reset_index(drop=True)


def discover_checkpoints(root, entries, overrides=None):
    """Identify raw checkpoints or named result ZIPs by pinned checkpoint bytes.

    A repeated attachment of identical bytes is an alias, never an extra seed.
    Unknown/mismatched checkpoints are never deserialized.
    """
    root = Path(root)
    wanted = {e["checkpoint_sha256"]: e for e in entries}
    found = {}
    candidates = list(sorted(root.rglob("progress.pt")))
    for path in (overrides or {}).values():
        candidates.append(Path(path))
    for path in dict.fromkeys(candidates):
        digest = sha256(path)
        if digest in wanted:
            found.setdefault(digest, []).append(dict(path=str(path), member=None))
    names = {e["archive"]: e for e in entries}
    for path in sorted(root.rglob("*.zip")):
        entry = names.get(path.name)
        if entry is None:
            continue
        if sha256(path) != entry["archive_sha256"]:
            raise ValueError(f"Result ZIP checksum differs from the audited upload: {path}")
        with zipfile.ZipFile(path) as archive:
            member = entry["checkpoint_member"]
            digest = hashlib.sha256(archive.read(member)).hexdigest()
        if digest != entry["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint member checksum differs: {path}")
        found.setdefault(digest, []).append(dict(path=str(path), member=member))
    return {e["job_id"]: found[e["checkpoint_sha256"]] for e in entries
            if e["checkpoint_sha256"] in found}


def _read_state(location):
    if location["member"] is None:
        return _load_checkpoint(Path(location["path"]), "cpu")
    # Only exact, locally audited bytes reach this project checkpoint loader.
    with zipfile.ZipFile(location["path"]) as archive:
        return torch.load(io.BytesIO(archive.read(location["member"])), map_location="cpu", weights_only=False)


def verify_state(state, entry):
    config = RunConfig(**state["training"])
    implementation = state["implementation_hash"]
    if training_implementation_hash() != VALIDATION_RECOVERY_HASH:
        raise ValueError("Analysis snapshot no longer matches the approved numerical runtime")
    if implementation not in {INPUT_V2_HASH, VALIDATION_RECOVERY_HASH}:
        raise ValueError("Only the audited input-v2 and numerical-only recovery snapshots are allowed")
    identity = fingerprint(dict(run=run_identity(config, state["model_config"], state["losses"], state["data_hash"]),
                                implementation=implementation, arm=state["arm"]))
    if identity != state["identity"] or identity != entry["identity"]:
        raise ValueError("Checkpoint identity differs from the pinned analysis manifest")
    checks = dict(dataset=config.dataset == entry["dataset"], seed=config.seed == entry["seed"],
                  arm=state["arm"] == entry["arm"], implementation=implementation == entry["implementation_hash"],
                  data=state["data_hash"] == entry["data_hash"], complete=state["complete"] is True,
                  best_weights=state.get("best_model") is not None,
                  config=fingerprint(state["model_config"]) == entry["model_config_hash"],
                  losses=fingerprint(state["losses"]) == entry["losses_hash"],
                  training=fingerprint(state["training"]) == entry["training_config_hash"])
    if not all(checks.values()):
        raise ValueError(f"Checkpoint verification failed: {[k for k, v in checks.items() if not v]}")
    return config


def _environment():
    return dict(python=platform.python_version(), platform=platform.platform(), torch=str(torch.__version__),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                cudnn=torch.backends.cudnn.version())


def _predict_report(model, frame, graphs, config, clock):
    events = []
    device = next(model.parameters()).device
    probabilities = predict(model, frame, graphs, config, device, clock, on_event=events.append)
    if len(probabilities) != len(frame) or not np.isfinite(probabilities).all():
        raise ValueError("Every requested pair must have a finite probability")
    return probabilities, events


@torch.no_grad()
def descriptor_probe(model, frame, graphs, config, clock, output, maximum_codes=256):
    ids = sorted(set(frame.left_id.astype(str)) | set(frame.right_id.astype(str)))
    if len(ids) > maximum_codes:
        ids = sorted(np.random.default_rng(9173).choice(ids, maximum_codes, replace=False).tolist())
    model.eval()
    model.encoder.capture_diagnostics = True
    descriptors, eigenvalues, rows = [], [], []
    try:
        for start in tqdm(range(0, len(ids), min(config.batch_size, 16)), desc="Descriptor/slot probe", unit="batch"):
            clock.check()
            keys = ids[start:start + min(config.batch_size, 16)]
            batch = pack_graphs([graphs[k] for k in keys], config.max_nodes).to(next(model.parameters()).device)
            # Diagnostic descriptors use FP32 consistently across all seeds.
            _, descriptor, auxiliary = model.encoder(batch, None)
            if not torch.isfinite(descriptor).all():
                raise ValueError("Non-finite FP32 diagnostic descriptors")
            assignment = auxiliary["assignment"].float()
            mass = assignment.sum(1)
            occupancy = mass / mass.sum(1, keepdim=True).clamp_min(1e-12)
            entropy = -(occupancy * occupancy.clamp_min(1e-12).log()).sum(1)
            entropy /= np.log(occupancy.shape[1])
            adjacency = auxiliary["adjacency"].float()
            density = adjacency.sum((1, 2)) / (adjacency.shape[1] * (adjacency.shape[1] - 1))
            descriptors.append(descriptor.float().cpu().numpy())
            eigenvalues.append(auxiliary["eigenvalues"].float().cpu().numpy())
            for i, key in enumerate(keys):
                graph = graphs[key]
                rows.append(dict(code_id=key, language=graph.language, retained_nodes=graph.node_count,
                                 unique_canonical_types=int(len(np.unique(graph.canonical_ids))),
                                 nonzero_lexical_fraction=float((graph.lexical_ids != 0).mean()),
                                 ast_edges=len(graph.edges[0]), ddg_edges=len(graph.edges[2]),
                                 latent_density=float(density[i]), slot_occupancy_entropy=float(entropy[i]),
                                 adaptive_temperature=float(auxiliary["adaptive_temperature"][i])))
    finally:
        model.encoder.capture_diagnostics = False
    values = np.concatenate(descriptors)
    blocks, offset = {}, 0
    for name, size in zip(("density", "heat", "spectral_energy"), model.block_sizes):
        block = np.log1p(np.maximum(values[:, offset:offset + size], 0))
        shape = block / np.linalg.norm(block, axis=1, keepdims=True).clip(1e-12)
        blocks[name] = dict(log_magnitude=_finite_summary(np.linalg.norm(block, axis=1)),
                            shape_std=_finite_summary(shape.std(0)))
        offset += size
    values64 = values.astype(np.float64)
    normalized = values64 / np.linalg.norm(values64, axis=1, keepdims=True).clip(1e-12)
    centered = normalized - normalized.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    mass = singular ** 2
    mass /= max(float(mass.sum()), 1e-30)
    effective_rank = float(np.exp(-(mass * np.log(mass.clip(1e-30))).sum())) if singular.max() > 1e-12 else 0.0
    frame_out = pd.DataFrame(rows)
    frame_out.to_csv(output / "code_diagnostics.csv", index=False)
    np.savez_compressed(output / "descriptors.npz", code_ids=np.asarray(ids), descriptors=values,
                        eigenvalues=np.concatenate(eigenvalues))
    return dict(codes=len(ids), code_support_hash=fingerprint(ids), precision="fp32",
                normalized_descriptor_std=_finite_summary(normalized.std(0)),
                near_constant_dimensions=int((normalized.std(0) < 1e-6).sum()),
                dimensions=values.shape[1], centered_effective_rank=effective_rank,
                comparison_blocks=blocks,
                mean_pair_cosine=float((np.square(normalized.sum(0)).sum() - np.square(normalized).sum())
                                       / max(1, len(ids) * (len(ids) - 1))),
                latent_density=_finite_summary(frame_out.latent_density),
                slot_occupancy_entropy=_finite_summary(frame_out.slot_occupancy_entropy),
                limitation="Aggregate collapse indicators on a fixed code sample; slot identities are not aligned across checkpoints")


def gradient_probe(model, train_frame, graphs, config, losses, clock, positive_weight):
    """One fixed FP32 eval-mode batch; gradients only, no optimizer or updates."""
    clock.check()
    frame = fixed_sample(train_frame, min(16, len(train_frame)))
    batch_config = replace(config, batch_size=len(frame), effective_batch_size=len(frame), amp=False)
    left, right, labels = next(iter(_loader(frame, graphs, batch_config)))
    device = next(model.parameters()).device
    model.eval()
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    logits, auxiliary = model(left.to(device), right.to(device), None)
    total, parts = canonical_spectra_loss(logits, auxiliary, labels.to(device),
                                          positive_class_weight=positive_weight, **losses)
    if not torch.isfinite(total) or any(not torch.isfinite(value).all() for value in parts.values()):
        raise FloatingPointError("Non-finite FP32 diagnostic loss/components")
    classification = F.binary_cross_entropy_with_logits(logits, labels.to(device),
                            pos_weight=logits.new_tensor(positive_weight))
    cls_grads = torch.autograd.grad(classification, [p for _, p in parameters], retain_graph=True, allow_unused=True)
    total_grads = torch.autograd.grad(total, [p for _, p in parameters], allow_unused=True)
    groups = {}
    for (name, _), c, t in zip(parameters, cls_grads, total_grads):
        group = ("classifier" if name.startswith("classifier.") else
                 "graph_encoder" if name.startswith("encoder.graph_layers.") else
                 "spectral_extractor" if name.startswith("encoder.spectral.") else "other_encoder")
        g = groups.setdefault(group, dict(classification_squared=0.0, total_squared=0.0, dot=0.0,
                                          parameters=0, disconnected_classification=0))
        if c is not None and not torch.isfinite(c).all() or t is not None and not torch.isfinite(t).all():
            raise ValueError("Non-finite FP32 diagnostic gradients")
        g["parameters"] += 1
        g["disconnected_classification"] += c is None
        g["classification_squared"] += float(c.double().square().sum()) if c is not None else 0.0
        g["total_squared"] += float(t.double().square().sum()) if t is not None else 0.0
        g["dot"] += float((c.double() * t.double()).sum()) if c is not None and t is not None else 0.0
    for g in groups.values():
        c, t = g.pop("classification_squared") ** .5, g.pop("total_squared") ** .5
        g.update(classification_norm=c, total_norm=t, cosine=g.pop("dot") / (c * t) if c * t > 0 else None)
    weights = dict(classification=1.0, spectral_contrastive=losses["spectral_weight"],
                   auc_ranking=losses["auc_ranking_weight"], embedding_contrastive=0.0,
                   hard_negative=losses["hard_negative_weight"], reconstruction=losses["reconstruction_weight"],
                   topology_reconstruction=losses["topology_reconstruction_weight"],
                   spectral_variance=losses["spectral_variance_weight"], graph=losses["graph_weight"])
    return dict(pairs=len(frame), support_hash=pair_support(frame), mode="eval_fp32_no_optimizer_no_updates",
                total_loss=float(total.detach()), raw_loss_components={k: float(v) for k, v in parts.items()},
                weighted_loss_components={k: float(v) * weights[k] for k, v in parts.items()},
                loss_weights=losses, gradient_groups=groups,
                limitation="One diagnostic batch at saved weights, not a reconstruction of training gradients")


@torch.no_grad()
def precision_probe(model, frame, graphs, config, clock):
    frame = fixed_sample(frame, min(128, len(frame)))
    fp32, _ = _predict_report(model, frame, graphs, replace(config, amp=False), clock)
    recorded, events = _predict_report(model, frame, graphs, config, clock)
    return dict(pairs=len(frame), support_hash=pair_support(frame),
                max_absolute_probability_difference=float(np.abs(fp32 - recorded).max()),
                fp32_AUROC=metrics(frame.label, fp32, .5)["AUROC"],
                recorded_precision_AUROC=metrics(frame.label, recorded, .5)["AUROC"], events=events)


def _artifact_hashes(folder):
    return {p.name: sha256(p) for p in folder.iterdir()
            if p.is_file() and p.name not in {"analysis_result.json", "analysis_error.json"}}


def _diagnostic_call(name, function, errors):
    try:
        return function()
    except SessionBudgetReached:
        raise
    except (ValueError, RuntimeError, FloatingPointError) as error:
        record = dict(probe=name, type=type(error).__name__, message=str(error))
        errors.append(record)
        print(json.dumps(dict(status="diagnostic_probe_failed", **record)), flush=True)
        return dict(status="failed", error=record)


def _reuse(folder, identity, resume_root=None):
    candidates = [folder]
    if resume_root:
        candidates += sorted(Path(resume_root).rglob(folder.name))
    for candidate in candidates:
        path = candidate / "analysis_result.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("analysis_identity") != identity:
            continue
        for name, digest in result["artifacts"].items():
            if Path(name).name != name or sha256(candidate / name) != digest:
                raise ValueError("Cached analysis artifacts differ from their saved checksums")
        folder.mkdir(parents=True, exist_ok=True)
        if candidate.resolve() != folder.resolve():
            for name in [*result["artifacts"], "analysis_result.json"]:
                shutil.copyfile(candidate / name, folder / name)
        return result
    return None


def small_report_zip(output):
    output = Path(output)
    path = output.with_name(output.name + "_small_reports.zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file.suffix in {".json", ".csv", ".gz", ".npz", ".md"}:
                archive.write(file, file.relative_to(output))
    return str(path)


def run_checkpoint_review(entries, clean_data, output, *, input_root=Path("/kaggle/input"),
                          diagnostics=False, session_hours=10.0, started=None,
                          resume_reports=None, checkpoint_paths=None, train_pairs=2048,
                          descriptor_codes=256, require_gpu=True):
    """Evaluate a pinned batch; completed per-checkpoint reports can be resumed."""
    if not entries or len({e["job_id"] for e in entries}) != len(entries):
        raise ValueError("A nonempty unique checkpoint roster is required")
    if train_pairs < 2 or descriptor_codes < 2:
        raise ValueError("Diagnostic pair/code sample sizes must be at least two")
    datasets = {e["dataset"] for e in entries}
    if len(datasets) != 1:
        raise ValueError("Use one matching dataset per notebook")
    dataset = next(iter(datasets))
    if dataset not in {"atcoder", "codenet"}:
        raise ValueError("This analysis batch is defined for AtCoder or CodeNet")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clock = SessionClock(RunConfig(session_hours=session_hours), started=started)
    roster = discover_checkpoints(input_root, entries, checkpoint_paths)
    missing = [e["job_id"] for e in entries if e["job_id"] not in roster]
    summary = dict(version=REVIEW_VERSION, dataset=dataset, requested_jobs=[e["job_id"] for e in entries],
                   missing_inputs=missing, completed=[], errors=[], test_evaluated=False, training_updates=0)
    _json(output / "checkpoint_discovery.json", dict(found=roster, missing=missing, expected=entries))
    if missing:
        summary["status"] = "missing_inputs"
        _json(output / "analysis_batch.json", summary)
        summary["download"] = small_report_zip(output)
        return summary
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU for inference; no training is performed")
    clock.check()
    corpus, frames, data_hash = load_pair_data(Path(clean_data), require_disjoint=dataset == "codenet")
    if any(e["data_hash"] != data_hash for e in entries):
        raise ValueError("Attached dataset differs from the audited checkpoint support")
    valid = frames["valid"].reset_index(drop=True)
    train = fixed_sample(frames["train"], train_pairs) if diagnostics else None
    wanted_ids = set(valid.left_id.astype(str)) | set(valid.right_id.astype(str))
    if train is not None:
        wanted_ids |= set(train.left_id.astype(str)) | set(train.right_id.astype(str))
    graphs = corpus.load_graphs(wanted_ids, max_nodes=256)
    if wanted_ids - set(graphs):
        raise ValueError("Requested pair endpoints are missing graphs; no pairs may be dropped")
    views = ("best_model", "model") if diagnostics else ("best_model",)
    settings = dict(version=REVIEW_VERSION, diagnostics=diagnostics, train_pairs=train_pairs,
                    descriptor_codes=descriptor_codes, valid_support=pair_support(valid),
                    train_support=pair_support(train) if train is not None else None,
                    analysis_source_sha256=sha256(Path(__file__)), runtime=training_implementation_hash())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        for entry in tqdm(entries, desc="Audited checkpoints", unit="checkpoint"):
            clock.check()
            state = _read_state(roster[entry["job_id"]][0])
            config = verify_state(state, entry)
            if config.max_nodes != 256:
                raise ValueError("This analysis plan fixes the recorded 256-node support")
            for view in views:
                clock.check()
                identity = fingerprint(dict(entry=entry, view=view, settings=settings))
                folder = output / f"{entry['job_id']}__{view}__{identity[:12]}"
                cached = _reuse(folder, identity, resume_reports)
                if cached:
                    summary["completed"].append(cached)
                    summary["errors"] += [dict(job_id=entry["job_id"], view=view, **error)
                                          for error in cached.get("diagnostic_errors", [])]
                    continue
                folder.mkdir(exist_ok=True)
                model = RevisionPairModel(RevisionModelConfig(**state["model_config"])).to(device).eval()
                model.load_state_dict(state[view], strict=True)
                start = time.monotonic()
                p, events = _predict_report(model, valid, graphs, config, clock)
                threshold, f1_metrics = choose_threshold(valid.label, p, "F1")
                old = metrics(valid.label, p, state["best_threshold"])
                prediction_columns = [c for c in ("pair_id", "left_id", "right_id", "label", "left_problem_id", "right_problem_id") if c in valid]
                predictions = valid[prediction_columns].copy()
                predictions["probability"] = p.astype(np.float64)
                predictions.to_csv(folder / "validation_predictions.csv.gz", index=False, compression="gzip")
                result = dict(status="complete", analysis_identity=identity, job_id=entry["job_id"],
                    dataset=dataset, arm=entry["arm"], seed=config.seed, weight_view=view,
                    checkpoint_sha256=entry["checkpoint_sha256"], checkpoint_identity=state["identity"],
                    data_hash=data_hash, implementation_hash=state["implementation_hash"],
                    original_selection_metric=config.selection_metric, selected_epoch=state["best_epoch"] if view == "best_model" else len(state["history"]),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    recorded_best_valid=state["best_valid"], validation_support_hash=pair_support(valid),
                    recorded_threshold_metrics=old, validation_f1_calibrated=f1_metrics,
                    threshold_calibration="posthoc_validation_F1_at_existing_weights",
                    f1_best_epoch_recovered=False, all_positive_F1=float(2 * valid.label.mean() / (1 + valid.label.mean())),
                    probability_summary=_finite_summary(p), recovery_events=events,
                    recorded_threshold_F1_drift=float(old["F1"] - state["best_valid"]["F1"]) if view == "best_model" else None,
                    environment=_environment(), settings=settings, test_evaluated=False, training_updates=0,
                    limitations=["Validation threshold fitting is optimistic; this is not held-out test performance.",
                                 "Existing epochs were selected by the recorded protocol, not retroactively by F1."])
                if diagnostics:
                    train_p, train_events = _predict_report(model, train, graphs, config, clock)
                    result["train_sample"] = dict(pairs=len(train), support_hash=pair_support(train),
                        at_validation_F1_threshold=metrics(train.label, train_p, threshold),
                        at_recorded_threshold=metrics(train.label, train_p, state["best_threshold"]),
                        probability_summary=_finite_summary(train_p), recovery_events=train_events)
                    train_out = train[[c for c in prediction_columns if c in train]].copy()
                    train_out["probability"] = train_p.astype(np.float64)
                    train_out.to_csv(folder / "train_sample_predictions.csv.gz", index=False, compression="gzip")
                    diagnostic_errors = []
                    result["descriptors"] = _diagnostic_call("descriptors", lambda:
                        descriptor_probe(model, valid, graphs, config, clock, folder, descriptor_codes), diagnostic_errors)
                    positive = int(frames["train"].label.sum())
                    result["gradient_probe"] = _diagnostic_call("gradients", lambda:
                        gradient_probe(model, train, graphs, config, state["losses"], clock,
                            min(10.0, (len(frames["train"]) - positive) / max(1, positive))), diagnostic_errors)
                    result["precision_probe"] = _diagnostic_call("precision", lambda:
                        precision_probe(model, valid, graphs, config, clock), diagnostic_errors)
                    result["diagnostic_errors"] = diagnostic_errors
                    summary["errors"] += [dict(job_id=entry["job_id"], view=view, **error) for error in diagnostic_errors]
                result["wall_seconds"] = time.monotonic() - start
                result["artifacts"] = _artifact_hashes(folder)
                _json(folder / "analysis_result.json", result)
                summary["completed"].append(result)
                _json(output / "analysis_batch.json", {**summary, "status": "in_progress"})
                remaining = len(entries) * len(views) - len(summary["completed"])
                print(json.dumps(dict(job=entry["job_id"], view=view, F1=f1_metrics["F1"], AUROC=f1_metrics["AUROC"],
                    estimated_remaining_minutes=float(np.mean([x["wall_seconds"] for x in summary["completed"]]))
                        * remaining / 60, estimate_excludes_already_completed_input_setup=True)), flush=True)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del state
        summary["status"] = "complete_with_diagnostic_errors" if summary["errors"] else "complete"
    except SessionBudgetReached:
        summary["status"] = "incomplete_session_budget"
    except Exception as error:
        summary["status"] = "analysis_error"
        summary["errors"].append(dict(type=type(error).__name__, message=str(error)))
        _json(output / "analysis_error.json", summary["errors"])
        import traceback
        traceback.print_exc()
    _json(output / "analysis_batch.json", summary)
    summary["download"] = small_report_zip(output)
    return summary


def collect_checkpoint_reviews(root, output, entries):
    """Combine matching reports, requiring all 18 core conditions for summaries."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Clear only tables produced by this collector; a partial recollection must
    # not leave a stale complete-results table from an earlier attachment set.
    for name in ("core_f1_by_seed.csv", "core_f1_summary.csv", "core_f1_paired_seed_deltas.csv", "atcoder_diagnostics_summary.csv"):
        (output / name).unlink(missing_ok=True)
    expected = {e["job_id"]: e for e in entries}
    found, aliases = {}, {}
    for path in sorted(Path(root).rglob("analysis_result.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        entry = expected.get(r.get("job_id"))
        if entry is None:
            continue
        if r.get("checkpoint_sha256") != entry["checkpoint_sha256"] or r.get("status") != "complete":
            raise ValueError("Analysis report does not match the audited checkpoint roster")
        if any(r.get(k) != entry[k] for k in ("dataset", "arm", "seed")):
            raise ValueError("Dataset, readout or seed differs from the pinned checkpoint")
        allowed_views = {"best_model", "model"} if entry["dataset"] == "atcoder" else {"best_model"}
        if r.get("weight_view") not in allowed_views:
            raise ValueError("Unexpected weight view in the fixed analysis plan")
        if r.get("test_evaluated") is not False or r.get("training_updates") != 0:
            raise ValueError("Only read-only validation reports belong in this batch")
        if (r.get("threshold_calibration") != "posthoc_validation_F1_at_existing_weights"
                or r.get("f1_best_epoch_recovered") is not False
                or r.get("settings", {}).get("analysis_source_sha256") != sha256(Path(__file__))):
            raise ValueError("Different analysis protocol/source; do not combine reports")
        for name, digest in r["artifacts"].items():
            if Path(name).name != name or sha256(path.parent / name) != digest:
                raise ValueError("Analysis prediction/diagnostic artifacts failed checksum verification")
        predictions = pd.read_csv(path.parent / "validation_predictions.csv.gz",
                                  dtype={"pair_id": str, "left_id": str, "right_id": str}, float_precision="round_trip")
        if pair_support(predictions) != r["validation_support_hash"]:
            raise ValueError("Exported predictions differ from their declared ordered support")
        _, reproduced = choose_threshold(predictions.label, predictions.probability, "F1")
        if not np.isclose(reproduced["F1"], r["validation_f1_calibrated"]["F1"], atol=1e-12):
            raise ValueError("F1 calibration does not reproduce from exported predictions")
        key = (r["job_id"], r["weight_view"])
        if key in found and fingerprint(found[key]) != fingerprint(r):
            raise ValueError(f"Divergent analysis attempts: {key}; retain both and resolve provenance")
        found[key] = r
        aliases.setdefault(str(key), []).append(str(path))
    core = [r for (_, view), r in found.items() if r["dataset"] == "codenet" and view == "best_model"]
    expected_core = {e["job_id"] for e in entries if e["dataset"] == "codenet"}
    missing = sorted(expected_core - {r["job_id"] for r in core})
    if core and len({r["validation_support_hash"] for r in core}) != 1:
        raise ValueError("Core models have different ordered validation support")
    rows = [dict(job_id=r["job_id"], arm=r["arm"], seed=r["seed"],
                 trainable_parameters=r.get("trainable_parameters"),
                 F1=r["validation_f1_calibrated"]["F1"],
                 P=r["validation_f1_calibrated"]["P"], R=r["validation_f1_calibrated"]["R"],
                 AUROC=r["validation_f1_calibrated"]["AUROC"],
                 original_threshold_F1=r["recorded_threshold_metrics"]["F1"],
                 threshold=r["validation_f1_calibrated"]["Threshold"],
                 original_epoch_selection=r["original_selection_metric"]) for r in core]
    if rows:
        pd.DataFrame(rows).sort_values(["arm", "seed"]).to_csv(output / "core_f1_by_seed.csv", index=False)
    complete = len(expected_core) == 18 and not missing
    if complete:
        frame = pd.DataFrame(rows)
        if set(frame.arm) != set(CORE_ARMS) or any(set(g.seed) != {42, 123, 231} for _, g in frame.groupby("arm")):
            raise ValueError("Core summaries require the prespecified six conditions x three seeds")
        frame.groupby("arm").agg(F1_mean=("F1", "mean"), F1_sample_sd=("F1", "std"),
                                  seeds=("seed", "count")).to_csv(output / "core_f1_summary.csv")
        deltas = []
        for a, b in (("full", "mean_max"), ("full", "attention"),
                     ("classification_only", "mean_max_classification_only"),
                     ("classification_only", "attention_classification_only"),
                     ("full", "classification_only"), ("mean_max", "mean_max_classification_only"),
                     ("attention", "attention_classification_only")):
            for seed in (42, 123, 231):
                get = lambda arm: float(frame[(frame.arm == arm) & (frame.seed == seed)].F1.iloc[0])
                deltas.append(dict(condition_a=a, condition_b=b, seed=seed, F1_a_minus_b=get(a) - get(b)))
        pd.DataFrame(deltas).to_csv(output / "core_f1_paired_seed_deltas.csv", index=False)
    atcoder = [dict(job_id=r["job_id"], seed=r["seed"], weight_view=r["weight_view"],
                    valid_F1=r["validation_f1_calibrated"]["F1"],
                    valid_AUROC=r["validation_f1_calibrated"]["AUROC"],
                    train_F1=r["train_sample"]["at_validation_F1_threshold"]["F1"],
                    train_AUROC=r["train_sample"]["at_validation_F1_threshold"]["AUROC"],
                    descriptor_effective_rank=r["descriptors"].get("centered_effective_rank"),
                    descriptor_mean_cosine=r["descriptors"].get("mean_pair_cosine"),
                    slot_entropy=r["descriptors"].get("slot_occupancy_entropy", {}).get("mean"))
               for r in found.values() if r["dataset"] == "atcoder"]
    if atcoder:
        pd.DataFrame(atcoder).sort_values(["seed", "weight_view"]).to_csv(output / "atcoder_diagnostics_summary.csv", index=False)
    diagnostic_errors = [dict(job_id=r["job_id"], view=r["weight_view"], **error)
                         for r in found.values() for error in r.get("diagnostic_errors", [])]
    result = dict(status=("complete_with_diagnostic_errors" if diagnostic_errors else "complete")
                  if complete and len(atcoder) == 6 else "incomplete_reports",
                  core_complete=complete, missing_core=missing, atcoder_views=len(atcoder),
                  diagnostic_errors=diagnostic_errors,
                  report_aliases=aliases, test_evaluated=False,
                  limitations=["Post-hoc F1 thresholds at Accuracy-selected checkpoints; no F1-best epoch recovery.",
                               "Validation results, not final tests; sample SD is not a confidence interval.",
                               "BCB seed231 duplicate-attempt provenance remains unresolved; no representative chosen."])
    _json(output / "collection_status.json", result)
    result["download"] = small_report_zip(output)
    return result
