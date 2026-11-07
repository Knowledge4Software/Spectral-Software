# Kaggle run order

Current first-stage instructions: [START_HERE_FA.md](../START_HERE_FA.md),
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), and
[stage1_run_queue.csv](../queues/stage1_run_queue.csv).
These replace the earlier recommendation to run three seeds of every control
before broad baseline screening. First run one CodeNet pilot, then all 29
seed-42 baseline jobs; architecture/loss controls and selected repetitions follow.
The user has now reported the CodeNet pilot complete. Reuse it and launch
the [next 24 independent jobs](../queues/LAUNCH_24.md) in parallel.

Each notebook has its own 12-hour ceiling. Training is configured for a 10-hour
session minus a 45-minute output reserve (about 9h15 for setup/training/analysis).
No fixed epoch count guarantees completion. A time-limited checkpoint is an
unfinished run, which can continue in another session. Do not run the old
`rq2`/`rq3` four-epoch notebooks for the revised comparisons.

## 1. Upload the new data

- CodeNet: `../data/benchmarks/codenet_revision_common_support.zip`.
- AtCoder: `../data/benchmarks/atcoder_revision_common_support_v2.zip`.
- BigCloneBench: `../data/benchmarks/bigclonebench_revision_common_support_v2.zip`.

Attach only the relevant bundle to each notebook, or set `CLEAN_DATA_DIR`.
Existing graph extraction is reused; no full parser pipeline is needed on Kaggle.
All compared methods must use the same bundle version. The older
`codenet_revision_problem_disjoint.zip` has a different support and must not be
mixed into the common-support result table.

## 2. Check speed and validity with seed 42

1. `01_baselines_seed42/codenet/codenet_full_seed42.ipynb`.
2. `01_baselines_seed42/codenet/codenet_astnn_seed42.ipynb`.
3. `01_baselines_seed42/codenet/codenet_codebert_seed42.ipynb`.
4. Remaining rows of `queues/stage1_run_queue.csv`; hardware checks can run
   alongside the other independent jobs. Check each method's initial logs.

Enable a GPU. Transformer notebooks also require Internet/model access. Keep
`FINAL_TEST = False`. Inspect `status.json`: validation history, stop reason,
runtime forecast, pair/data hashes and parameter counts. Compare validation
trajectories, not test scores. If tuning learning rate, declare the candidate
set before running it, choose using seed-42 validation only, and freeze that
choice for seeds 123/231. Defaults are method-specific starting settings, not
claimed optimal hyperparameters.
For text models, copy the pilot's resolved model commit from `tokens/tokens.json`
into `MODEL_REVISION` for the remaining seeds. Aggregation rejects revision or
hyperparameter changes across seeds instead of mixing them silently.

`MAX_EPOCHS=100` is a ceiling. A smaller common cap can be chosen from pilots,
but reaching it is a fixed-budget result, not proof of convergence. A completed
run reports `maximum_epochs` separately from `validation_patience`. Do not
automatically choose different epoch caps for whichever model looks strongest.
Keep physical batch sizes fixed within each method after its pilot. Changing
them changes contrastive/hard-negative objectives, even with accumulation.

## 3. Complete the central scientific controls

After baseline screening, run seed 42 for the full, mean/max and attention arms. Add the
corresponding classification-only controls: these distinguish readout effects
from graph/spectral regularizer effects. Run projected AST+DDG, projected AST,
projected DDG, density/heat-only and energy-only controls. Then run loss removal,
joint spectral/hard-negative removal, and half/double auxiliary-weight controls.

Add seeds 123/231 only for selected central comparisons. Run the new external baselines on the same support. Use the dataset folders in `01_baselines_seed42/`
for AtCoder and BigCloneBench instead of mixing their old result archives
with the corrected implementation. BigCloneBench is a monolingual control,
not an additional cross-language benchmark. Its new test support is limited
after endpoint/source-disjoint filtering; report pair AND unique-program counts.

## 4. Continue an unfinished job

Save its outputs and attach them as input to a new version of the SAME notebook.
Set `RESUME_CHECKPOINT` to the attached `progress.pt`. For transformers also set
`RESUME_TOKENS_DIR` to the attached `tokens` directory to avoid retokenizing.
Keep dataset, source snapshot, method, seed and scientific settings unchanged.
Source/data/configuration changes deliberately invalidate continuation.

Frozen Qwen embeddings have separate `encode` and `probe` phases. Follow that
notebook's `RESUME_EMBEDDINGS_DIR`, `EMBEDDINGS_DIR` and `RESUME_PROBE_DIR` settings.

## 5. Expand matched transfer

The focused Java-to-Python design is frozen in `EXPERIMENT_PROTOCOL.md` and
`focused_transfer_plan.csv`. It adds genuine source-only training, a common
Python-Python test and explicit train/validation exposure quotas. Generate
these later, after selecting comparator models:

```powershell
python scripts/build_revision_notebooks.py --transfer-key 'focused|source_only' --transfer-key 'focused|direct' --transfer-key 'focused|two_hop' --transfer-key 'focused|three_hop' --transfer-key 'focused|target_reference'
```

The generator writes SPECTRA AND external-baseline jobs for these conditions;
run only the prespecified selected models. For a non-spectral SPECTRA sibling,
set ARM to mean_max or attention; the arm has its own label and output path.
The older table7_cross condition is a different mixed-pair test, not source-only.
It also supports `--all-transfer` for all 166 configurations, including the
reinforcement variants. Equal pair/step budgets do not mean equal compute or
equal target-language exposure. Target-language labels are permitted in bridge
training and must be stated explicitly. This is held-out pair-configuration
generalization, not unseen-language transfer.

## 6. Freeze choices and release final tests

After choices are frozen, reattach each completed checkpoint and set
`FINAL_TEST=True`. Final test metrics are released once and cached. Preserve
the associated `test_predictions.csv.gz` and metadata. Do not return to method,
loss, epoch or threshold selection using those scores.

Use `05_analysis/validation_spectral_diagnostics.ipynb` with a main SPECTRA
checkpoint and its exact dataset. It uses validation only. Use
`05_analysis/paired_problem_uncertainty.ipynb` after final evaluation, with
complete matched conditions and three shared seeds for both methods. Its
bootstrap also checkpoints if it hits the time guard. BigCloneBench cannot use
problem-clustered inference without authoritative problem metadata.

Benchmark timing is measured after training. If its status is pending because
the session expired, rerun with the completed checkpoint; training is not repeated.
Different physical batches are reported. Timing excludes loading, device copies,
optimizer steps and checkpoint I/O; report full training/validation wall time too.

## 7. What a run cannot supply

GPU results cannot fix unsupported prose automatically. Rewrite the manuscript
from the new evidence, narrow language-invariance/superiority claims when needed,
and distinguish absolute accuracy from relative degradation. Clone-type labels,
historical parser success logs and user IDs are unavailable locally. Exact
FSD-CLCD/ZC3 reproductions are not supplied under invented method names.
