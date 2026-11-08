"""Collect validation-only progress; never open test results or checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .protocol import fingerprint

# Explicit compatibility exception for the audited input-only fix. External
# baselines never call canonical_graph_from_record/lexical_source. Preserve
# their original hashes in exports; old SPECTRA outputs remain incompatible.
INPUT_V1_HASH = "793a3d2debc4cac7c045fb34df66f4f12f0694538e3da7a046c1b0623cee2de4"
INPUT_V2_HASH = "8ad938f286142084e2c89ef8439bf9310db61e30019f3d61d02b90426df52e26"
EXTERNAL_INPUT_V1_ARMS = frozenset({
    "baseline/" + method for method in
    ("astnn", "rtvnn", "cdlh", "deepsim", "fa_ast_ggnn", "fa_ast_gmn", "codebert", "c4_adapted", "qwen3_lora")
})


def comparison_implementation_hash(state):
    implementation = state["implementation_hash"]
    if implementation == INPUT_V1_HASH and state["arm"] in EXTERNAL_INPUT_V1_ARMS:
        return INPUT_V2_HASH
    return implementation


def collect_validation(root, output, queue=None):
    snapshots = {}
    for path in sorted(Path(root).rglob("status.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if not {"identity", "training", "implementation_hash", "arm"} <= state.keys():
            continue
        identity = state["identity"]
        state["sources"] = [str(path)]
        previous = snapshots.get(identity)
        if previous:
            a, b = previous.get("history", []), state.get("history", [])
            n = min(len(a), len(b))
            if fingerprint(a[:n]) != fingerprint(b[:n]):
                raise ValueError(f"Divergent histories for the same identity: {identity}")
            rank = lambda s: (bool(s.get("complete")), len(s.get("history", [])), s.get("optimizer_steps", 0))
            chosen = state if rank(state) > rank(previous) else previous
            sources = previous["sources"] + state["sources"]
            state = {**chosen, "sources": sources}
        snapshots[identity] = state
    planned = pd.read_csv(queue) if queue is not None else None
    planned_keys = (set(zip(planned.dataset, planned.arm, planned.seed.astype(int)))
                    if planned is not None else None)
    rows = []
    for state in snapshots.values():
        config = state["training"]
        key = (config["dataset"], state["arm"], int(config["seed"]))
        if planned_keys is not None and key not in planned_keys:
            continue
        valid = state.get("best_valid") or {}
        primary = config["selection_metric"]
        rows.append({"dataset": key[0], "arm": key[1], "seed": key[2],
                     "identity": state["identity"], "data_hash": state["data_hash"],
                     "implementation_hash": state["implementation_hash"],
                     "comparison_implementation_hash": comparison_implementation_hash(state),
                     "complete": bool(state.get("complete")), "status": state["status"],
                     "stop_reason": state.get("stop_reason"), "primary_metric": primary,
                     "validation_score": valid.get(primary), "best_epoch": state.get("best_epoch"),
                     "validated_epochs": len(state.get("history", [])),
                     "optimizer_steps": state.get("optimizer_steps"), "gpu": state.get("gpu"),
                     "sources": json.dumps(state["sources"]),
                     **{f"valid_{m}": valid.get(m) for m in
                        ("Accuracy", "F1", "AUROC", "AUPRC_AP", "MCC", "Threshold")}})
    frame = pd.DataFrame(rows)
    keys = ["dataset", "arm", "seed"]
    if not frame.empty and frame.duplicated(keys).any():
        raise ValueError("Multiple configurations for the same dataset/arm/seed; select authoritative candidate outputs first")
    if not frame.empty:
        for _, group in frame.groupby("dataset"):
            if group.comparison_implementation_hash.nunique() != 1 or group.data_hash.nunique() != 1:
                raise ValueError("Screen each dataset on one common support and training implementation")
        # Incomplete jobs are visible in the ledger but cannot win screening.
        frame["validation_rank"] = float("nan")
        complete = frame.complete & frame.validation_score.notna()
        frame.loc[complete, "validation_rank"] = frame[complete].groupby(["dataset", "seed"])[
            "validation_score"].rank(method="min", ascending=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "validation_screening.csv", index=False)
    if planned is not None:
        ledger = (planned.merge(frame, how="left", on=keys, validate="one_to_one") if not frame.empty
                  else planned.assign(status="not_started", complete=False))
        ledger["status"] = ledger.status.fillna("not_started")
        ledger.to_csv(output / "run_ledger.csv", index=False)
    (output / "README.md").write_text(
        "Validation only. No final_test.json, predictions or progress.pt files were opened. "
        "Only completed runs receive a rank. Resumed snapshots are deduplicated by identity; "
        "conflicting configurations require an explicit candidate choice. Single-seed ranks "
        "are screening evidence, not statistical superiority claims. The documented input-v2 "
        "exception reuses input-v1 external baselines only: their preparation is unchanged. "
        "Original implementation hashes remain in the CSV. Old SPECTRA inputs are incompatible.\n", encoding="utf-8")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue", type=Path)
    args = parser.parse_args()
    print(collect_validation(args.root, args.output, args.queue).to_string(index=False))
