"""Paired multi-seed comparisons with shared-problem resampling across paths.

This is an exploratory multi-membership/pigeonhole bootstrap, not a claim of
independent transfer paths or a distribution-free significance test. One set
of problem weights is reused across both methods, all seeds and all paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import fingerprint
from .screening import comparison_implementation_hash
from tqdm.auto import tqdm
from .io_utils import replace_file


def _family(label):
    if label.startswith("transfer/"):
        parts = label.split("/", 2)
        return parts[1] if len(parts) == 3 else "full"  # legacy transfer/{key}
    if label.startswith("baseline/"):
        return "/".join(label.split("/")[:2])
    return label


def read_predictions(root, methods=None):
    runs = {}
    for path in sorted(Path(root).rglob("final_test.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "test" not in result or not result.get("test", {}).get("Pairs"):
            continue
        if methods is not None and _family(result["arm"]) not in methods:
            continue
        transfer_file = path.parent / "transfer_metadata.json"
        condition = json.loads(transfer_file.read_text())["key"] if transfer_file.exists() else "main"
        key = (_family(result["arm"]), condition, int(result["seed"]))
        if key in runs:
            raise ValueError(f"Duplicate method/condition/seed: {key}")
        frame = pd.read_csv(path.parent / "test_predictions.csv.gz",
                            dtype={"pair_id": str, "left_id": str, "right_id": str,
                                   "left_problem_id": str, "right_problem_id": str})
        required = {"pair_id", "left_problem_id", "right_problem_id", "label", "probability"}
        if not required.issubset(frame) or frame[list(required)].isna().any().any():
            raise ValueError(f"Known problem IDs and pair IDs are required for clustered analysis: {path}")
        if frame.pair_id.duplicated().any():
            raise ValueError("Duplicate pair IDs within an evaluation")
        frame = frame.sort_values("pair_id").reset_index(drop=True)
        frame["prediction"] = frame.probability.ge(result["test"]["Threshold"]).astype(int)
        runs[key] = (result, frame)
    return runs


def _score(y, prediction, weights, metric):
    if weights.sum() <= 0:
        return np.nan
    if metric == "Accuracy":
        return np.average(y == prediction, weights=weights)
    tp = np.sum(weights * ((y == 1) & (prediction == 1)))
    fp = np.sum(weights * ((y == 0) & (prediction == 1)))
    fn = np.sum(weights * ((y == 1) & (prediction == 0)))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def paired_comparison(root, method_a, method_b, output, *, repetitions=2000, seed=42, metric="Accuracy", clock=None):
    if metric not in {"Accuracy", "F1"} or repetitions < 2:
        raise ValueError("Choose Accuracy/F1 and at least two bootstrap replicates")
    runs = read_predictions(root, {method_a, method_b})
    left = {(condition, run_seed): value for (method, condition, run_seed), value in runs.items() if method == method_a}
    right = {(condition, run_seed): value for (method, condition, run_seed), value in runs.items() if method == method_b}
    if not left or set(left) != set(right):
        raise ValueError("Both methods must have exactly the same completed conditions and seeds; no silent intersections")
    conditions = sorted({key[0] for key in left})
    seeds = sorted({key[1] for key in left})
    if len(seeds) < 3 or set(left) != {(condition, run_seed) for condition in conditions for run_seed in seeds}:
        raise ValueError("At least three shared seeds, complete for every condition, are required")
    aligned, problems, versions, supports, protocols = {}, set(), set(), set(), {}
    rows = []
    for key in sorted(left):
        a, fa = left[key]
        b, fb = right[key]
        versions.update((comparison_implementation_hash(a), comparison_implementation_hash(b)))
        supports.update((a.get("data_group_hash", a["data_hash"]), b.get("data_group_hash", b["data_hash"])))
        for result in (a, b):
            training = dict(result.get("protocol", {}))
            for field in ("seed", "session_hours", "reserve_minutes", "checkpoint_minutes"):
                training.pop(field, None)
            method_condition = (_family(result["arm"]), key[0])
            protocols.setdefault(method_condition, set()).add(fingerprint({
                "model": result.get("model_config", {}), "training": training, "losses": result.get("losses", {})}))
        columns = ["pair_id", "left_problem_id", "right_problem_id", "label"]
        if not fa[columns].equals(fb[columns]):
            raise ValueError(f"Different test support/labels for {key}; freeze shared support before comparing")
        problems.update(fa.left_problem_id); problems.update(fa.right_problem_id)
        aligned[key] = (fa, fb)
        y = fa.label.to_numpy()
        weights = np.ones(len(y))
        av = _score(y, fa.prediction.to_numpy(), weights, metric)
        bv = _score(y, fb.prediction.to_numpy(), weights, metric)
        rows.append({"condition": key[0], "seed": key[1], "method_a": av, "method_b": bv, "difference_a_minus_b": av - bv})
    if len(versions) != 1 or len(supports) != 1:
        raise ValueError("Mixed implementation snapshots or dataset supports")
    if any(len(settings) != 1 for settings in protocols.values()):
        raise ValueError("Model revisions/hyperparameters/schedules differ across seeds")
    problem_index = {key: i for i, key in enumerate(sorted(problems))}
    arrays = {}
    for key, (fa, fb) in aligned.items():
        arrays[key] = (fa.label.to_numpy(), fa.prediction.to_numpy(), fb.prediction.to_numpy(),
                       fa.left_problem_id.map(problem_index).to_numpy(), fa.right_problem_id.map(problem_index).to_numpy())
    rng = np.random.default_rng(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    progress_file = output / "bootstrap_progress.json"
    identity = fingerprint({"a": method_a, "b": method_b, "metric": metric, "repetitions": repetitions,
                            "seed": seed, "inputs": {str(key): tuple(array.tolist() for array in values) for key, values in arrays.items()},
                            "implementation": Path(__file__).read_text(encoding="utf-8")})
    progress = {"identity": identity, "next_repetition": 0, "differences": [], "rng": rng.bit_generator.state}
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["identity"] != identity:
            raise ValueError("Bootstrap continuation identity mismatch")
        rng.bit_generator.state = progress["rng"]
    differences = progress["differences"]
    def save_progress():
        progress["rng"] = rng.bit_generator.state
        temporary = progress_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress), encoding="utf-8")
        replace_file(temporary, progress_file)
    for iteration in tqdm(range(progress["next_repetition"], repetitions),
                          initial=progress["next_repetition"], total=repetitions,
                          desc="Paired bootstrap", unit="replicate", mininterval=2):
        if clock is not None:
            from .training import SessionBudgetReached
            try:
                clock.check()
            except SessionBudgetReached:
                save_progress()
                return {"status": "incomplete_session_budget", "completed_repetitions": iteration,
                        "continuation": "Reattach the bootstrap output and copy bootstrap_progress.json to this output directory"}
        # Resample seed identities as complete trajectories, not per-path seeds.
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        counts = rng.multinomial(len(problems), np.full(len(problems), 1 / len(problems)))
        path_differences = []
        for condition in conditions:
            values = []
            for run_seed in sampled_seeds:
                y, a, b, p, q = arrays[(condition, run_seed)]
                # Same-problem pairs belong to one cluster, not two copies.
                weights = counts[p] * np.where(p == q, 1, counts[q])
                values.append(_score(y, a, weights, metric) - _score(y, b, weights, metric))
            path_differences.append(np.mean(values))
        difference = np.mean(path_differences)
        if np.isfinite(difference):
            differences.append(float(difference))
        progress["next_repetition"] = iteration + 1
        if (iteration + 1) % 50 == 0:
            save_progress()
    save_progress()
    if len(differences) < repetitions * .9:
        raise ValueError("Too many empty resamples; more independent problems are needed")
    frame = pd.DataFrame(rows)
    estimate = float(frame.groupby("condition").difference_a_minus_b.mean().mean())
    result = {"method_a": method_a, "method_b": method_b, "metric": metric,
              "difference_a_minus_b": estimate,
              "ci95_percentile": np.quantile(differences, [.025, .975]).tolist(),
              "seeds": seeds, "conditions": len(conditions), "problems": len(problems),
              "repetitions": repetitions, "valid_repetitions": len(differences),
              "aggregation": "equal weight per condition, then per training seed",
              "resampling": "seed trajectories and shared problem weights; same weights across methods and paths",
              "limitation": "Exploratory multi-membership bootstrap; does not establish language-population generalization or independent transfer paths. No p-value reported."}
    frame.to_csv(output / "paired_condition_seed_scores.csv", index=False)
    pd.DataFrame({"difference_a_minus_b": differences}).to_csv(output / "bootstrap_differences.csv", index=False)
    (output / "paired_cluster_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--method-a", default="full")
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--metric", choices=("Accuracy", "F1"), default="Accuracy")
    args = parser.parse_args()
    print(json.dumps(paired_comparison(args.root, args.method_a, args.method_b, args.output,
                                       repetitions=args.repetitions, metric=args.metric), indent=2))
