"""A modern frozen Qwen3 embedding baseline with a train-only linear probe.

Model loading/pooling follows the publisher's model card:
https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
This is explicitly a frozen representation baseline, not ZC3 or a fine-tuned LLM.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .protocol import choose_threshold, fingerprint, metrics
from .io_utils import replace_file
from .training import SessionBudgetReached


def _json_save(value, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    replace_file(temporary, path)


def extract_embeddings(codes: dict[str, str], ids: list[str], output: Path, *, clock,
                       resume: Path | None = None, max_length: int = 1024, batch_size: int = 8,
                       model_id: str = "Qwen/Qwen3-Embedding-0.6B", revision: str | None = None):
    output.mkdir(parents=True, exist_ok=True)
    code_hashes = [(key, hashlib.sha256(codes[key].encode()).hexdigest()) for key in ids]
    identity = fingerprint({"codes": code_hashes, "model": model_id, "requested_revision": revision,
                            "max_length": max_length, "pooling": "last_token", "prompt": "none",
                            "source_version": 1,
                            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    manifest_path = output / "embeddings.json"
    matrix_path = output / "embeddings.npy"
    if resume and not manifest_path.exists():
        for name in ("embeddings.json", "embeddings.npy", "embedding_ids.json"):
            shutil.copyfile(resume / name, output / name)
    state = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if state is not None:
        if state["identity"] != identity:
            raise ValueError("Embedding resume mismatch: code pool/model/token limit changed")
        if state["complete"]:
            return state
    clock.check()
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = state["resolved_revision"] if state else revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved, padding_side="left", trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id, revision=resolved, trust_remote_code=False,
                                      torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    model.requires_grad_(False)
    dimension = model.config.hidden_size
    if state is None:
        state = {"identity": identity, "model": model_id, "resolved_revision": model.config._commit_hash,
                 "next_index": 0, "complete": False, "dimension": dimension,
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "max_length": max_length, "batch_size": batch_size, "truncated_codes": 0,
                 "encoding_seconds": 0.0, "codes": len(ids), "encoder_training": "frozen",
                 "pooling": "last token; left padding; L2 normalized; no task prompt"}
        np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(len(ids), dimension)).flush()
        (output / "embedding_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        _json_save(state, manifest_path)
    matrix = np.load(matrix_path, mmap_mode="r+")
    try:
        while state["next_index"] < len(ids):
            clock.check()
            start = state["next_index"]
            stop = min(len(ids), start + state["batch_size"])
            texts = [codes[key] for key in ids[start:stop]]
            started = time.monotonic()
            lengths = [len(row) for row in tokenizer(texts, truncation=False)["input_ids"]]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            try:
                with torch.inference_mode():
                    hidden = model(**tokens).last_hidden_state
                    vectors = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                del tokens
                torch.cuda.empty_cache()
                if state["batch_size"] <= 1:
                    raise
                state["batch_size"] = max(1, state["batch_size"] // 2)
                _json_save(state, manifest_path)
                continue
            if not np.isfinite(vectors).all():
                raise FloatingPointError("Non-finite pretrained embeddings")
            matrix[start:stop] = vectors
            matrix.flush()
            state["next_index"] = stop
            state["truncated_codes"] += sum(n > max_length for n in lengths)
            state["encoding_seconds"] += time.monotonic() - started
            state["complete"] = stop == len(ids)
            _json_save(state, manifest_path)
            print(f"Embedded {stop}/{len(ids)}", flush=True)
    except SessionBudgetReached:
        _json_save(state, manifest_path)
    return state


def pair_vectors(matrix, left, right):
    a, b = np.asarray(matrix[left]), np.asarray(matrix[right])
    return np.concatenate((np.abs(a - b), a * b, (a * b).sum(-1, keepdims=True),
                           np.linalg.norm(a - b, axis=-1, keepdims=True)), axis=-1)


def fit_probe(frames: dict, embeddings: Path, output: Path, *, clock, seed: int = 42,
              primary: str = "Accuracy", resume: Path | None = None, final_test: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((embeddings / "embeddings.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Complete embedding extraction before fitting the supervised probe")
    ids = json.loads((embeddings / "embedding_ids.json").read_text())
    index = {key: i for i, key in enumerate(ids)}
    arrays = {name: (f.left_id.map(index).to_numpy(), f.right_id.map(index).to_numpy(), f.label.to_numpy())
              for name, f in frames.items()}
    if any(pd.isna(a).any() or pd.isna(b).any() for a, b, _ in arrays.values()):
        raise ValueError("An evaluation endpoint has no pretrained embedding")
    matrix = np.load(embeddings / "embeddings.npy", mmap_mode="r")
    identity = fingerprint({"embedding": manifest["identity"], "seed": seed, "metric": primary,
                            "pairs": {k: fingerprint(v.to_csv(index=False)) for k, v in frames.items()},
                            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "learner": "train-scaled SGD logistic alpha1e-4, 100 epochs, patience10, min10"})
    path = output / "probe.pkl"
    if resume and not path.exists():
        shutil.copyfile(resume / "probe.pkl", path)
    if path.exists():
        with path.open("rb") as stream:
            state = pickle.load(stream)  # trusted checkpoint from this experiment only
        if state["identity"] != identity:
            raise ValueError("Linear probe resume identity mismatch")
    else:
        state = {"identity": identity, "stage": "scale", "offset": 0, "epoch": 1,
                 "scaler": StandardScaler(), "model": SGDClassifier(loss="log_loss", alpha=1e-4,
                                                                       random_state=seed, shuffle=False),
                 "history": [], "best": None, "bad_epochs": 0, "complete": False}

    def save():
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(state, stream)
        replace_file(temporary, path)

    def scores(name):
        a, b, _ = arrays[name]
        return np.concatenate([expit(state["model"].decision_function(state["scaler"].transform(pair_vectors(matrix, a[i:i+1024], b[i:i+1024]))))
                               for i in range(0, len(a), 1024)])

    a, b, y = arrays["train"]
    try:
        while state["stage"] == "scale":
            clock.check()
            i = state["offset"]
            state["scaler"].partial_fit(pair_vectors(matrix, a[i:i+1024], b[i:i+1024]))
            state["offset"] += 1024
            if state["offset"] >= len(a):
                state.update(stage="train", offset=0)
            save()
        while not state["complete"]:
            order = np.random.default_rng(seed + state["epoch"]).permutation(len(a))
            for i in range(state["offset"], len(a), 1024):
                clock.check()
                rows = order[i:i+1024]
                features = state["scaler"].transform(pair_vectors(matrix, a[rows], b[rows]))
                state["model"].partial_fit(features, y[rows], classes=np.array([0, 1]))
                state["offset"] = i + 1024
                save()
            clock.check()
            threshold, valid = choose_threshold(arrays["valid"][2], scores("valid"), primary)
            state["history"].append({"epoch": state["epoch"], "valid": valid})
            if state["best"] is None or valid[primary] > state["best"]["valid"][primary]:
                state["best"] = {"valid": valid, "threshold": threshold,
                                 "model": pickle.dumps(state["model"]), "epoch": state["epoch"]}
                state["bad_epochs"] = 0
            else:
                state["bad_epochs"] += 1
            state["epoch"] += 1; state["offset"] = 0
            state["complete"] = state["epoch"] > 100 or (state["epoch"] > 10 and state["bad_epochs"] >= 10)
            save()
    except SessionBudgetReached:
        save()
    summary = {"identity": identity, "complete": state["complete"], "history": state["history"],
               "encoder_training": "frozen", "head": "train-only standardized logistic SGD"}
    if final_test and state["complete"]:
        result_file = output / "final_test.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        clock.check()
        state["model"] = pickle.loads(state["best"]["model"])
        p = scores("test")
        cosine_valid = np.sum(matrix[arrays["valid"][0]] * matrix[arrays["valid"][1]], axis=-1)
        cosine_test = np.sum(matrix[arrays["test"][0]] * matrix[arrays["test"][1]], axis=-1)
        threshold, _ = choose_threshold(arrays["valid"][2], cosine_valid, primary)
        summary.update(seed=seed, model=manifest["model"], resolved_revision=manifest["resolved_revision"],
                       probe_test=metrics(arrays["test"][2], p, state["best"]["threshold"]),
                       cosine_test=metrics(arrays["test"][2], cosine_test, threshold))
        predictions = frames["test"].copy()
        predictions["probability"] = p
        predictions.to_csv(output / "test_predictions.csv.gz", index=False, compression="gzip")
        _json_save(summary, result_file)
    _json_save(summary, output / "probe_status.json")
    return summary
