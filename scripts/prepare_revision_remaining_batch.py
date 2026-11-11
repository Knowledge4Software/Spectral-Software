"""Prepare independent, already planned controls/repeats with the validation fix."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from scripts.prepare_revision_stage1 import DATASETS, ROOT
from spectral_code.revision.numerics import VALIDATION_RECOVERY_SOURCE, can_upgrade
from spectral_code.revision.protocol import fingerprint
from spectral_code.revision.screening import VALIDATION_RECOVERY_HASH
from spectral_code.revision.versioning import TRAINING_SOURCES, training_implementation_hash

CONTROLS = (
    ("projected", "architecture", "Learned adjacency versus projected AST+DDG; R6.2/R7.2"),
    ("projected_ast", "architecture", "Projected AST alone; R6.2/R7.2"),
    ("projected_ddg", "architecture", "Projected DDG alone; R6.2/R7.2"),
    ("density_heat", "architecture", "Eigenvalue/density/heat features without signal energies; spectral attribution"),
    ("energy_only", "architecture", "Signal energy features without density/heat features; spectral attribution"),
    ("without_spectral", "losses", "Remove spectral contrastive objective; R6.5"),
    ("without_auc_ranking", "losses", "Remove AUC ranking objective; R6.5"),
    ("without_hard_negative", "losses", "Remove hard-negative objective; R6.5"),
    ("without_reconstruction", "losses", "Remove feature reconstruction objective; R6.5"),
    ("without_topology_reconstruction", "losses", "Remove topology reconstruction objective; R6.5"),
    ("without_spectral_variance", "losses", "Remove spectral-variance objective; R6.5"),
    ("without_graph", "losses", "Remove graph regularization objective; R6.5"),
    ("without_spectral_and_hard_negative", "losses", "Remove two interacting objectives together; R6.5"),
    ("auxiliary_half", "losses", "Half all auxiliary weights; coarse joint-weight sensitivity, R6.5"),
    ("auxiliary_double", "losses", "Double all auxiliary weights; coarse joint-weight sensitivity, R6.5"),
)


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def _fixed_copy(base, relative, target_relative, sources, description):
    nb = json.loads((base / relative).read_text(encoding="utf-8"))
    config = "".join(nb["cells"][1]["source"])
    assignments = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(config).body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and isinstance(n.value, ast.Constant)}
    expected = dict(RESUME_CHECKPOINT=None, ALLOW_NUMERICAL_UPGRADE=False, FINAL_TEST=False,
                    MAX_EPOCHS=100, MIN_EPOCHS=10, PATIENCE=10, SESSION_HOURS=10.0)
    if any(assignments[k] != v for k, v in expected.items()):
        raise ValueError(f"Unexpected settings in {relative}")
    bootstrap = "".join(nb["cells"][2]["source"])
    node = next(n for n in ast.parse(bootstrap).body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_sources")
    old = ast.literal_eval(node.value)
    if fingerprint({n: old[n] for n in TRAINING_SOURCES}) != VALIDATION_RECOVERY_SOURCE:
        raise ValueError(f"Unexpected original implementation in {relative}")
    if {n for n in TRAINING_SOURCES if old[n] != sources[n]} != {"training.py", "numerics.py"}:
        raise ValueError("Only the approved numerical amendment may differ")
    nb["cells"][2]["source"] = bootstrap.replace(ast.get_source_segment(bootstrap, node),
                                                  "_sources = " + repr(sources), 1).splitlines(keepends=True)
    nb["cells"][0]["source"] = (
        f"# {Path(relative).stem}: planned independent revision experiment\n\n{description}\n\n"
        "Fresh run: attach only its matching existing dataset. GPU on, Internet off, FINAL_TEST=False. "
        "An invalid AMP prediction pass is retried once for all pairs in FP32. Persistent FP32 failure stops; "
        "finite AMP predictions and optimizer updates are unchanged. Same input-v2 representation, model, losses "
        "and 100-epoch ceiling / minimum 10 / patience 10 as the frozen original. "
        "The 10-hour session includes a 45-minute reserve; a budget pause needs continuation.\n"
    ).splitlines(keepends=True)
    run = "".join(nb["cells"][3]["source"])
    nb["cells"][3]["source"] = run.replace('_input_v2"', '_input_v2_validation_fix"').splitlines(keepends=True)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), target_relative, "exec")
    target = base / target_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return hashlib.sha256(target.read_bytes()).hexdigest(), assignments


def prepare(root=None):
    base = Path(root) if root is not None else ROOT / "revision"
    if training_implementation_hash() != VALIDATION_RECOVERY_HASH or not can_upgrade(VALIDATION_RECOVERY_SOURCE):
        raise ValueError("Approved numerical snapshot has changed; review before generation")
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "spectral_code/revision").glob("*.py"))}
    baseline = {r["job_id"]: r for r in _read_csv(base / "results/run_progress.csv")}
    controls = {r["job_id"]: r for r in _read_csv(base / "results/control_progress.csv")}
    # Full models are prespecified core methods; these repeats do not depend on
    # choosing the top two external baselines or finishing CodeNet seed 123.
    followup_path = base / "results/selected_followup_progress.csv"
    followups = _read_csv(followup_path)
    if not followups:
        followups = [dict(job_id=f"{d}_full_seed{s}", dataset=d, arm="full", seed=s,
                         status="not_reported", evidence="", completed_epochs="", test_evaluated=False,
                         updated="2026-11-09", notes="Prespecified full-model repeat; independent of external finalist selection",
                         implementation_hash="") for d in ("atcoder", "bigclonebench") for s in (123, 231)]
        _write_csv(followup_path, followups, list(followups[0]))
    progress = {**controls, **{r["job_id"]: r for r in followups}}
    manifest = {r["notebook"]: r for r in _read_csv(base / "queues/notebook_manifest.csv")}
    plans = []
    for arm, group, purpose in CONTROLS:
        plans.append(("codenet", arm, 42, f"02_controls_seed42/codenet/{group}/codenet_{arm}_seed42.ipynb",
                      f"02_controls_seed42/codenet/validation_recovery/{group}/codenet_{arm}_seed42_validation_recovery.ipynb",
                      purpose, 2, 8, 8.0))
    for d, low, high, cap in (("atcoder", 2, 6, 7.7), ("bigclonebench", 1, 4, 6.9)):
        reference = baseline[f"{d}_full_seed42"]
        if reference["status"] != "complete_verified" or reference["implementation_hash"] != VALIDATION_RECOVERY_SOURCE:
            raise ValueError(f"Verified current seed-42 reference required: {d}")
        for s in (123, 231):
            plans.append((d, "full", s, f"03_selected_repeats/{d}/baselines/{d}_full_seed{s}.ipynb",
                          f"03_selected_repeats/{d}/validation_recovery/{d}_full_seed{s}_validation_recovery.ipynb",
                          "Already planned full-model cross-benchmark seed replication; R2 uncertainty", low, high, cap))
    jobs = []
    for dataset, arm, seed, relative, target, purpose, low, high, cap in plans:
        if hashlib.sha256((base / relative).read_bytes()).hexdigest() != manifest[relative]["sha256"]:
            raise ValueError(f"Frozen notebook bytes changed: {relative}")
        digest, config = _fixed_copy(base, relative, target, sources, purpose)
        if config["ARM"] != arm or config["SEED"] != seed:
            raise ValueError("Notebook arm/seed differs from queue")
        job_id = f"{dataset}_{arm}_seed{seed}"
        status = progress[job_id]["status"]
        bundle, data_digest, _ = DATASETS[dataset]
        jobs.append(dict(slot=len(jobs)+1, job_id=job_id, dataset=dataset, arm=arm, seed=seed,
                         notebook=target, notebook_sha256=digest, original_notebook=relative,
                         dataset_zip=bundle+".zip", dataset_sha256=data_digest,
                         implementation_hash=VALIDATION_RECOVERY_HASH, primary_metric="F1" if dataset=="bigclonebench" else "Accuracy",
                         status=status, estimated_hours_low=low, estimated_hours_high=high,
                         estimated_100_epoch_hours_with_margin=cap, gpu=True, internet=False,
                         resume_checkpoint="None", final_test=False, purpose=purpose))
    fresh = [j for j in jobs if j["status"] in {"not_reported", "not_started"}]
    core_progress = _read_csv(base/"results/core_repeat_progress.csv")
    core_done = len(core_progress)==12 and all(r["status"]=="complete_verified" for r in core_progress)
    core_note = ("All 12 CodeNet core repeats are complete; do not repeat them." if core_done else
                 "Run independently of the CodeNet classification-only/123 restart; do not launch a duplicate of any live job.")
    _write_csv(base/"queues/next_independent_roster.csv", jobs, list(jobs[0]))
    _write_csv(base/"queues/next_independent_run_queue.csv", fresh, list(jobs[0]))
    lines = ["# Next independent batch: controls and full-model repeats", "",
             f"**{len(fresh)} fresh jobs ready from a 19-job preplanned roster.** {core_note}", "",
             "The original roster contains 15 CodeNet seed-42 controls and four full-model repeats on AtCoder/BigCloneBench. Verified completions are excluded from the fresh list below. All were already in the revision plan. The four full repeats are independent of external-finalist selection. The other selected external repeats and transfer study still wait for completed baseline screening. Nothing is selected using test results.", "",
             "## Exact notebooks and T4 estimates", "",
             "| Slot | Dataset | Condition | Seed | Hours per notebook | Notebook |",
             "| ---: | --- | --- | ---: | --- | --- |"]
    for j in fresh:
        lines.append(f"| {j['slot']} | {j['dataset']} | {j['arm']} | {j['seed']} | {j['estimated_hours_low']}-{j['estimated_hours_high']} | [{Path(j['notebook']).name}](../{j['notebook']}) |")
    lines += ["", "The 15 CodeNet controls address spectral attribution, learned/projected graph views, descriptor components, loss contributions and coarse auxiliary-weight sensitivity (especially R6.2/R6.5). Full-model repeats measure variability beyond CodeNet; they do not prove superiority or complete baseline comparisons.", "",
              "## Inputs and settings", "", "Reuse the existing uploads; no dataset rebuild or upload is needed:", "",
              "- CodeNet controls: `codenet_revision_common_support.zip`.",
              "- AtCoder repeats: `atcoder_revision_common_support_v2.zip`.",
              "- BigCloneBench repeats: `bigclonebench_revision_common_support_v2.zip`.", "",
              "Attach only the matching dataset. GPU on, Internet off, RESUME_CHECKPOINT=None, ALLOW_NUMERICAL_UPGRADE=False, FINAL_TEST=False. All are fresh jobs. The files include the approved numerical validation fix; use these linked files rather than the frozen originals. Model/data/loss/stopping settings and tqdm are preserved. New output paths end in `_input_v2_validation_fix`.", "",
              "## How the estimates were obtained", "",
              "These are planning ranges per notebook on one T4, including normal setup, not measured timings for the 15 new arms. CodeNet's completed reference conditions took about 1.7-6.0h (24-86 epochs), roughly 3.8-4.1 minutes/epoch. Using the slowest observed core rate, 100 epochs plus setup and 15% planning margin is about 8.0h. The new graph/loss variants may behave differently; the common range deliberately avoids inventing an arm-specific speedup.", "",
              "AtCoder full/42 took 4.00h for 60 epochs, with 3.96h training/validation. Its extrapolated 100-epoch estimate with setup and 15% margin is 7.7h. BigCloneBench full/42 took 1.44h for 24 epochs (1.43h training/validation); the comparable cap estimate is 6.9h. The 2-6h and 1-4h ranges assume earlier validation stopping, not a promise for seeds 123/231.", "",
              "Each session still has a 10h internal budget with a 45-minute reserve (work stops around 9h15). Keep the 100-epoch ceiling, minimum 10 and patience 10. Numerical retries or slower hardware can require another session; preserve progress rather than shortening the schedule. Each job's own runtime forecast becomes more informative after its first completed epochs.", "",
              f"The {len(fresh)} listed jobs require {len(fresh)} GPU-job slots; account for any other active jobs within your 24-slot limit. Check existing Kaggle jobs before starting duplicates. Allow roughly 8-10h for the parallel batch if no job needs continuation; this is not an estimate for completing the entire revision.", "",
              "## Modern baselines are still pending", "",
              "CodeBERT/C4 remain important review gaps. The old CodeNet jobs need their own saved checkpoints/token caches and original archived notebooks; see [continuation instructions](../00_plan/TEXT_RUN_CONTINUATION.md). They completed only about six epochs in the first approximately 9h work session, so further 10h sessions may be needed. Starting AtCoder/BigCloneBench text baselines also needs runtime pilots; do not label them as 2-8h graph jobs.", "",
              "Qwen nearly filled a session with epoch-1 training alone, and CDLH also paused early. They remain in the roster but are outside this short batch; no convergence time is promised. Do not launch extra seeds or transfer jobs for unfinished external methods. Final tests remain off.", "",
              "See [protocol](../00_plan/EXPERIMENT_PROTOCOL.md) and [reviewer coverage](../00_plan/REVIEWER_COVERAGE.md). Check existing Kaggle jobs before launching a duplicate. Generating this queue does not mean any job has started.", ""]
    (base/"queues/RUN_NEXT_INDEPENDENT_BATCH.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Prepared {len(jobs)} fixed notebooks; {len(fresh)} fresh launches ready. Training runtime unchanged.")
    return jobs


if __name__ == "__main__":
    prepare()
