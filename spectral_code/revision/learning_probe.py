"""Versioned tiny-training diagnostic; never replaces main experiment results."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import log_loss
from tqdm.auto import tqdm

from .checkpoint_review import fixed_sample, pair_support, small_report_zip, sha256
from .model import RevisionModelConfig, RevisionPairModel, canonical_spectra_loss
from .numerics import optimizer_group
from .protocol import RunConfig, metrics
from .screening import VALIDATION_RECOVERY_HASH
from .training import _loader, arm_settings, load_pair_data, predict, SessionBudgetReached
from .versioning import training_implementation_hash

PROBE_VERSION = "atcoder_tiny_learning_probe_v1"


class ProbeClock:
    def __init__(self, seconds, started=None):
        if seconds <= 0:
            raise ValueError("Budget must be positive")
        self.started = time.monotonic() if started is None else started
        self.deadline = self.started + seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise SessionBudgetReached("Tiny-probe wall budget reached")


def initial_weights_hash(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_probe(root, output, *, budget_seconds=540, steps=60, started=None):
    if training_implementation_hash() != VALIDATION_RECOVERY_HASH:
        raise ValueError("Frozen scientific implementation changed")
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU")
    if steps < 1:
        raise ValueError("steps must be positive")
    clock = ProbeClock(budget_seconds, started)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(version=PROBE_VERSION, status="preparing", cases=[], errors=[], test_evaluated=False,
        purpose="Memorization/optimization diagnostic, not a new main model or benchmark result",
        plan=dict(seeds=[123, 231], objectives=["full", "classification_only"], steps=steps,
                  train_pairs=64, valid_pairs=256, sample_seed=9173, physical_batch=32, effective_batch=128,
                  initialization="fresh; identical weights within each seed", learning_rate=1e-4,
                  criterion="train AUROC >= 0.95 AND train BCE < 0.2 within the fixed update budget"),
        budget_seconds=budget_seconds, source_sha256=sha256(Path(__file__)),
        implementation_hash=training_implementation_hash(),
        environment=dict(torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
                         cuda=torch.version.cuda, cpu_threads=torch.get_num_threads()))

    def save(status):
        summary.update(status=status, wall_seconds=time.monotonic() - clock.started)
        (output / "probe_status.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save("preparing")
    try:
        corpus, frames, data_hash = load_pair_data(Path(root), require_disjoint=False)
        if data_hash != "787cd8526f68728f416d5109bb3a809f6c30b964cb915dad31c889600b3e0247":
            raise ValueError("Use the existing audited AtCoder input-v2 bundle")
        train, valid = fixed_sample(frames["train"], 64), fixed_sample(frames["valid"], 256)
        ids = set(train.left_id) | set(train.right_id) | set(valid.left_id) | set(valid.right_id)
        graphs = corpus.load_graphs(ids, max_nodes=256)
        if ids - set(graphs):
            raise ValueError("Missing graph endpoints; no pairs may be dropped")
        summary.update(data_hash=data_hash, train_support=pair_support(train), valid_support=pair_support(valid))
        train.to_csv(output / "fixed_train_pairs.csv", index=False)
        valid.to_csv(output / "fixed_validation_pairs.csv", index=False)
        init_by_seed = {}
        for seed in (123, 231):
            for arm in ("full", "classification_only"):
                clock.check()
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                config = RunConfig(dataset="atcoder", seed=seed, batch_size=32, effective_batch_size=128)
                changes, losses = arm_settings(arm)
                model_config = RevisionModelConfig(**changes)
                model = RevisionPairModel(model_config).cuda()
                initial = initial_weights_hash(model)
                if seed in init_by_seed and initial != init_by_seed[seed]:
                    raise ValueError("Objective siblings must start from identical weights")
                init_by_seed[seed] = initial
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                scaler = torch.amp.GradScaler("cuda", enabled=True)
                case = dict(seed=seed, arm=arm, initialization_hash=initial, losses=losses,
                            config=asdict(config), model_config=asdict(model_config), history=[], numerical_events=[],
                            status="in_progress", updates=0)
                summary["cases"].append(case)
                folder = output / f"{arm}_seed{seed}"
                folder.mkdir(exist_ok=True)

                def evaluate(step):
                    model.eval()
                    values = {}
                    for split, frame in (("train", train), ("valid", valid)):
                        probabilities = predict(model, frame, graphs, config, torch.device("cuda"), clock,
                                                on_event=case["numerical_events"].append)
                        values[split] = dict(**metrics(frame.label, probabilities, .5),
                                             BCE=float(log_loss(frame.label, probabilities.astype(float), labels=[0, 1])))
                        if step == steps:
                            f = frame.copy(); f["probability"] = probabilities.astype(float)
                            f.to_csv(folder / f"{split}_predictions.csv.gz", index=False, compression="gzip")
                    case["history"].append(dict(step=step, **values))
                    print(json.dumps(dict(seed=seed, arm=arm, step=step, train_AUROC=values["train"]["AUROC"],
                        train_BCE=values["train"]["BCE"], valid_AUROC=values["valid"]["AUROC"])), flush=True)
                    save("running")

                evaluate(0)
                case_started = time.monotonic()
                for step in tqdm(range(1, steps + 1), desc=f"Tiny {arm}/{seed}", unit="update", mininterval=3):
                    clock.check()
                    generator = torch.Generator().manual_seed(seed * 10000 + step)
                    order = torch.cat([torch.randperm(len(train), generator=generator) for _ in range(2)]).tolist()
                    batches = [order[i:i+32] for i in range(0, 128, 32)]
                    group = list(_loader(train, graphs, config, batches))
                    model.train()
                    optimizer_group(model, optimizer, scaler, group, config=config, device=torch.device("cuda"),
                                    losses=losses, loss_function=canonical_spectra_loss, positive_class_weight=1.0,
                                    clock=clock, on_event=case["numerical_events"].append)
                    case["updates"] = step
                    if step % 15 == 0 or step == steps:
                        evaluate(step)
                    if step == 5:
                        remaining_cases = 4 - len(summary["cases"])
                        elapsed = time.monotonic() - clock.started
                        # Input cost is already paid. Estimate from five measured updates plus evaluation overhead.
                        estimated_left = (time.monotonic() - case_started) / 5 * ((steps - 5) + remaining_cases * steps) * 1.2
                        if elapsed + estimated_left > budget_seconds:
                            raise SessionBudgetReached("Measured forecast exceeds local wall budget; use the Kaggle notebook")
                case["status"] = "complete"
                last = case["history"][-1]["train"]
                case["memorized_within_budget"] = last["AUROC"] >= .95 and last["BCE"] < .2
                save("running")
                del model, optimizer, scaler
                gc.collect(); torch.cuda.empty_cache()
        save("complete")
    except SessionBudgetReached as error:
        summary["errors"].append(str(error)); save("incomplete_budget")
    except Exception as error:
        summary["errors"].append(dict(type=type(error).__name__, message=str(error))); save("probe_error")
        import traceback
        traceback.print_exc()
    summary["download"] = small_report_zip(output)
    return summary
