"""Train and evaluate one RQ3 configuration for one model family.

Every model is trained with the same schedule, splits, and validation-based
checkpoint selection, so accuracies across families stay comparable. Results
append to a CSV after each configuration, which makes a run resumable across
Kaggle sessions: rerunning skips whatever is already recorded.

The heavy model code lives in the existing baseline runtimes rather than being
duplicated here:
  * SPECTRA-Siam  -> the RQ2 notebook runtime
  * ASTNN/RtvNN/DeepSim -> spectral_code.faithful_graph_baselines
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.kaggle.rq3.matrix import Configuration, all_configurations

RESULT_COLUMNS = (
    "Model", "Table", "Configuration", "Target", "Path", "Reinforcement",
    "TrainBuckets", "TestBucket", "Accuracy", "P", "R", "F1",
    "TrainPairs", "ValidPairs", "TestPairs", "Threshold",
    "RuntimeMinutes", "Seed", "CompletedUTC",
)


def load_pairs(clean_data: Path) -> pd.DataFrame:
    """Read the CodeNet pairs table with its language-bucket column."""
    frame = pd.read_csv(
        clean_data / "pairs.csv.gz",
        dtype={"left_id": str, "right_id": str, "label": np.int8},
    )
    required = {"split", "left_id", "right_id", "label", "configuration_id"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{clean_data/'pairs.csv.gz'} is missing columns: {sorted(missing)}")
    return frame


def split_frames(
    pairs: pd.DataFrame, configuration: Configuration, *, max_train: int | None = None, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Select the train/valid/test rows this configuration is allowed to use.

    Train and validation come from the configuration's training buckets only.
    Test comes from the target bucket, which training never sees, so a bridge
    result cannot be inflated by direct supervision on the target.
    """
    train_mask = pairs.configuration_id.isin(configuration.train_buckets)
    test_mask = pairs.configuration_id.eq(configuration.test_bucket)
    frames = {
        "train": pairs[train_mask & pairs.split.eq("train")].reset_index(drop=True),
        "valid": pairs[train_mask & pairs.split.eq("valid")].reset_index(drop=True),
        "test": pairs[test_mask & pairs.split.eq("test")].reset_index(drop=True),
    }
    if max_train is not None and len(frames["train"]) > max_train:
        frames["train"] = frames["train"].sample(max_train, random_state=seed).reset_index(drop=True)
    for name, frame in frames.items():
        if frame.empty:
            raise RuntimeError(f"{configuration.key}: {name} split is empty")
        if set(frame.label.astype(int).unique()) != {0, 1}:
            raise RuntimeError(f"{configuration.key}: {name} split is single-class")
    return frames


def completed_keys(result_path: Path, model: str) -> set[str]:
    """Configuration keys already recorded for this model."""
    if not result_path.is_file():
        return set()
    frame = pd.read_csv(result_path)
    if frame.empty or "Model" not in frame:
        return set()
    frame = frame[frame.Model == model]
    return {f"{row.Table}|{row.Configuration}" for row in frame.itertuples()}


def append_result(result_path: Path, record: dict) -> None:
    """Append one finished configuration, creating the file on first write."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{column: record.get(column) for column in RESULT_COLUMNS}])
    header = not result_path.is_file()
    frame.to_csv(result_path, mode="a", header=header, index=False)


def build_record(
    model: str, configuration: Configuration, metrics: dict, sizes: dict, seconds: float, seed: int
) -> dict:
    return {
        "Model": model,
        "Table": configuration.table,
        "Configuration": configuration.name,
        "Target": configuration.target,
        "Path": "->".join(configuration.path),
        "Reinforcement": configuration.reinforcement,
        "TrainBuckets": "+".join(configuration.train_buckets),
        "TestBucket": configuration.test_bucket,
        "Accuracy": metrics["Acc"],
        "P": metrics["P"],
        "R": metrics["R"],
        "F1": metrics["F1"],
        "TrainPairs": sizes["train"],
        "ValidPairs": sizes["valid"],
        "TestPairs": sizes["test"],
        "Threshold": metrics.get("Threshold"),
        "RuntimeMinutes": seconds / 60.0,
        "Seed": seed,
        "CompletedUTC": pd.Timestamp.utcnow().isoformat(),
    }


def run(
    model: str,
    train_one,
    clean_data: Path,
    result_path: Path,
    *,
    tables: tuple[str, ...] | None = None,
    max_train: int | None = None,
    seed: int = 42,
    resume: bool = True,
) -> pd.DataFrame:
    """Run every configuration for one model, skipping finished ones.

    ``train_one(frames, configuration) -> metrics`` is supplied by the caller
    so this module stays independent of any particular model implementation.
    """
    pairs = load_pairs(clean_data)
    configurations = [
        configuration for configuration in all_configurations()
        if tables is None or configuration.table in tables
    ]
    done = completed_keys(result_path, model) if resume else set()
    pending = [configuration for configuration in configurations if configuration.key not in done]
    print(f"[{model}] {len(configurations)} configurations, {len(done)} already done, {len(pending)} to run")

    from tqdm.auto import tqdm

    progress = tqdm(pending, desc=f"RQ3 {model}", unit="cfg")
    for configuration in progress:
        progress.set_description(f"RQ3 {model}: {configuration.table} {configuration.name}")
        frames = split_frames(pairs, configuration, max_train=max_train, seed=seed)
        started = time.perf_counter()
        metrics = train_one(frames, configuration)
        seconds = time.perf_counter() - started
        sizes = {name: len(frame) for name, frame in frames.items()}
        append_result(result_path, build_record(model, configuration, metrics, sizes, seconds, seed))
        tqdm.write(
            f"  {configuration.table:14s} {configuration.name:24s} "
            f"acc={metrics['Acc']:.4f} f1={metrics['F1']:.4f} ({seconds/60:.1f} min)"
        )
        gc.collect()
    progress.close()
    return pd.read_csv(result_path) if result_path.is_file() else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tables", nargs="*", default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()
