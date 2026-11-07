"""Aggregate completed revision runs only, with seed uncertainty and provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .protocol import fingerprint


def aggregate(root: Path, output: Path, minimum_seeds: int = 3) -> pd.DataFrame:
    rows, identities = [], set()
    for path in sorted(root.rglob("final_test.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "test" not in result or "implementation_hash" not in result:
            continue  # modern encoder probes have a separate, explicit schema
        if result["identity"] in identities:
            raise ValueError(f"Duplicate result identity; choose one authoritative archive: {path}")
        identities.add(result["identity"])
        training = dict(result.get("protocol", {}))
        for key in ("seed", "session_hours", "reserve_minutes", "checkpoint_minutes"):
            training.pop(key, None)
        rows.append({"arm": result["arm"], "seed": result["seed"], "data_hash": result["data_hash"],
                     "data_group_hash": result.get("data_group_hash", result["data_hash"]),
                     "implementation_hash": result["implementation_hash"], "source": str(path),
                     "model_configuration_hash": fingerprint(result.get("model_config", {})),
                     "training_schedule_hash": fingerprint({"training": training, "losses": result.get("losses", {})}),
                     **{key: result["test"][key] for key in ("Accuracy", "F1", "AUPRC_AP", "AUROC", "MCC")}})
    if not rows:
        raise ValueError("No completed revision test results found; training-only progress is not a paper result")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["arm", "seed", "data_hash", "implementation_hash"]).any():
        raise ValueError("Duplicate arm/seed results")
    if frame.implementation_hash.nunique() != 1:
        raise ValueError("Do not combine different implementation snapshots")
    for _, group in frame.groupby(["arm", "data_group_hash"]):
        if group.model_configuration_hash.nunique() != 1 or group.training_schedule_hash.nunique() != 1:
            raise ValueError("Model revision, hyperparameters or training schedule changed across seeds; freeze choices before combining")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "revision_seed_results.csv", index=False)
    summaries = []
    for (arm, data_hash), group in frame.groupby(["arm", "data_group_hash"]):
        n = group.seed.nunique()
        if n < minimum_seeds:
            continue
        row = {"arm": arm, "data_hash": data_hash, "seeds": n}
        for metric in ("Accuracy", "F1", "AUPRC_AP", "AUROC", "MCC"):
            values = group[metric].astype(float).to_numpy()
            mean, std = float(values.mean()), float(values.std(ddof=1)) if n > 1 else float("nan")
            half = float(t.ppf(.975, n - 1) * std / np.sqrt(n)) if n > 1 else float("nan")
            row.update({f"{metric}_mean": mean, f"{metric}_std": std,
                        f"{metric}_ci95_low": mean - half, f"{metric}_ci95_high": mean + half})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "revision_summary.csv", index=False)
    (output / "README.md").write_text(
        "Seed intervals are Student-t intervals for the mean across training seeds. "
        "They do not represent independent problem samples or independent transfer paths. "
        "Incomplete seed groups are retained in revision_seed_results.csv and excluded from the summary. "
        "No degradation statistic or cross-path significance claim is inferred automatically.\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    args = parser.parse_args()
    print(aggregate(args.root, args.output, args.minimum_seeds).to_string(index=False))
