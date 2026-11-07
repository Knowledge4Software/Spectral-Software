"""Matched data/step transfer controls; target-language participation is explicit."""
from __future__ import annotations

import pandas as pd

from .protocol import fingerprint, matched_training_pairs
from .transfer_matrix import all_configurations, bucket


# Prespecified Java -> Python case study; S denotes C#, C denotes C++.
# All five conditions share the SAME Python-Python test rows. Only the three
# bridge conditions match target-containing training/validation label budgets.
FOCUSED_QUOTAS = {
    "focused|source_only": ({"J-J": 6000}, {"J-J": 1200}),
    "focused|direct": ({"J-J": 4500, "J-P": 1500}, {"J-J": 900, "J-P": 300}),
    "focused|two_hop": ({"J-J": 2250, "J-C": 2250, "C-P": 1500},
                        {"J-J": 450, "J-C": 450, "C-P": 300}),
    "focused|three_hop": ({"J-J": 1500, "J-C": 1500, "C-S": 1500, "S-P": 1500},
                          {"J-J": 300, "J-C": 300, "C-S": 300, "S-P": 300}),
    "focused|target_reference": ({"P-P": 6000}, {"P-P": 1200}),
    "focused|two_hop_reinforced": ({"J-J": 1500, "J-C": 1500, "C-C": 1500, "C-P": 1500},
                                  {"J-J": 300, "J-C": 300, "C-C": 300, "C-P": 300}),
}


def _bucket_codes(bucket):
    from .transfer_matrix import bucket as dataset_bucket
    return dataset_bucket(*bucket.split("-")) if "-" in bucket else bucket


def sample_quotas(frame, quotas, seed):
    """Exactly the stated quota, balanced within each language-pair bucket."""
    parts = []
    for bucket, count in sorted(quotas.items()):
        if count <= 0 or count % 2:
            raise ValueError("Every per-bucket quota must be positive and even")
        code = _bucket_codes(bucket)
        for label in (0, 1):
            group = frame[frame.configuration_id.eq(code) & frame.label.eq(label)]
            if len(group) < count // 2:
                raise ValueError(f"Insufficient {bucket}/{label}: need {count // 2}, have {len(group)}")
            parts.append(group.sample(count // 2, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _selected_hash(frames):
    # Bind identity to actual sampled support, not merely the quota description.
    return fingerprint({split: frame[["pair_id", "left_id", "right_id", "label"]].to_dict("records")
                        for split, frame in frames.items()})


def _focused_frames(frames, key, budget, seed):
    if budget != 6000:
        raise ValueError("The prespecified focused design requires exactly 6000 training pairs")
    train_quotas, valid_quotas = FOCUSED_QUOTAS[key]
    selected = {"train": sample_quotas(frames["train"], train_quotas, seed),
                "valid": sample_quotas(frames["valid"], valid_quotas, seed + 1),
                "test": frames["test"][frames["test"].configuration_id.eq(bucket("P", "P"))].reset_index(drop=True)}
    if set(selected["test"].label.unique()) != {0, 1}:
        raise ValueError("Focused transfer test requires both labels")
    exposure = lambda quotas: sum(n for bucket, n in quotas.items() if "P" in bucket.split("-"))
    metadata = {"key": key, "train_buckets": [_bucket_codes(b) for b in train_quotas], "test_bucket": bucket("P", "P"),
                "train_quotas": train_quotas, "valid_quotas": valid_quotas,
                "train_pairs": 6000, "valid_pairs": 1200,
                "target_containing_train_pairs": exposure(train_quotas),
                "target_containing_valid_pairs": exposure(valid_quotas),
                "target_language_supervision": "none in task training/validation" if "source_only" in key
                                                else "target-containing task labels",
                "interpretation": "Java-to-Python case study; pretraining exposure is not excluded; "
                                  "bridge language composition changes with path length",
                "source_problem_disjoint": True, "selected_pairs_hash": _selected_hash(selected)}
    return selected, metadata


def select_transfer_frames(frames: dict, key: str, budget: int, seed: int) -> tuple[dict, dict]:
    if key in FOCUSED_QUOTAS:
        return _focused_frames(frames, key, budget, seed)
    configurations = {c.key: c for c in all_configurations()}
    if key not in configurations:
        raise ValueError(f"Unknown transfer key {key}; inspect transfer_plan.csv")
    configuration = configurations[key]
    if configuration.table != "table7_within" and configuration.test_bucket in configuration.train_buckets:
        raise ValueError("Target pair configuration must be held out")
    train = frames["train"][frames["train"].configuration_id.isin(configuration.train_buckets)]
    train = matched_training_pairs(train, configuration.train_buckets, budget, seed)
    valid = frames["valid"][frames["valid"].configuration_id.isin(configuration.train_buckets)].copy()
    test = frames["test"][frames["test"].configuration_id.eq(configuration.test_bucket)].copy()
    if any(frame.empty or set(frame.label.unique()) != {0, 1} for frame in (train, valid, test)):
        raise ValueError("Every transfer partition must have both labels")
    metadata = {"key": key, "train_buckets": configuration.train_buckets,
                "test_bucket": configuration.test_bucket, "train_pairs": len(train),
                "target_language_supervision": "bridge cross-language labels" if configuration.path else
                                               "same-language endpoint labels",
                "interpretation": "held-out pair configuration; target language is not unseen",
                "source_problem_disjoint": True}
    for column in ("left_language", "right_language"):
        if column in train:
            metadata[f"{column}_counts"] = train[column].value_counts().to_dict()
    selected = {"train": train.reset_index(drop=True), "valid": valid.reset_index(drop=True),
                "test": test.reset_index(drop=True)}
    metadata["selected_pairs_hash"] = _selected_hash(selected)
    return selected, metadata


def write_focused_plan(path):
    rows = [{"key": key, "train_quotas": str(train), "valid_quotas": str(valid),
             "test_bucket": "P-P", "train_pairs": sum(train.values()), "valid_pairs": sum(valid.values()),
             "target_containing_train_pairs": sum(n for b, n in train.items() if "P" in b.split("-")),
             "target_containing_valid_pairs": sum(n for b, n in valid.items() if "P" in b.split("-")),
             "epochs": 30, "optimizer_steps": 1410,
             "optional": key.endswith("reinforced")}
            for key, (train, valid) in FOCUSED_QUOTAS.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_plan(path):
    rows = [{"key": c.key, "target": c.target, "path": "->".join(c.path),
             "train_buckets": ",".join(c.train_buckets), "test_bucket": c.test_bucket,
             "reinforcement": c.reinforcement, "matched_train_pairs": 6000,
             "epochs": 30, "optimizer_steps_at_batch32_effective128": 1410}
            for c in all_configurations()]
    pd.DataFrame(rows).to_csv(path, index=False)
