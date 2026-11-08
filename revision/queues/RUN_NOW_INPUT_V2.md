# Run now: input v2, seed 42

Run these **eight** jobs in parallel if GPU slots are available. Import the updated local files. External baselines can wait for now.

**Use existing dataset uploads. No new dataset upload or preprocessing job is required.**

| Slot | Dataset | Notebook | Dataset ZIP already uploaded |
| ---: | --- | --- | --- |
| 1 | codenet | [full](../01_baselines_seed42/codenet/codenet_full_seed42.ipynb) | `codenet_revision_common_support.zip` |
| 2 | atcoder | [full](../01_baselines_seed42/atcoder/atcoder_full_seed42.ipynb) | `atcoder_revision_common_support_v2.zip` |
| 3 | bigclonebench | [full](../01_baselines_seed42/bigclonebench/bigclonebench_full_seed42.ipynb) | `bigclonebench_revision_common_support_v2.zip` |
| 4 | codenet | [mean_max](../02_controls_seed42/codenet/architecture/codenet_mean_max_seed42.ipynb) | `codenet_revision_common_support.zip` |
| 5 | codenet | [attention](../02_controls_seed42/codenet/architecture/codenet_attention_seed42.ipynb) | `codenet_revision_common_support.zip` |
| 6 | codenet | [classification_only](../02_controls_seed42/codenet/losses/codenet_classification_only_seed42.ipynb) | `codenet_revision_common_support.zip` |
| 7 | codenet | [mean_max_classification_only](../02_controls_seed42/codenet/architecture/codenet_mean_max_classification_only_seed42.ipynb) | `codenet_revision_common_support.zip` |
| 8 | codenet | [attention_classification_only](../02_controls_seed42/codenet/architecture/codenet_attention_classification_only_seed42.ipynb) | `codenet_revision_common_support.zip` |

Settings: GPU on (T4 is supported), Internet off, seed 42, `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`. Keep the default 100-epoch ceiling / minimum 10 / patience 10. Each session has a 10-hour budget with 45 minutes reserved for outputs, below the Kaggle 12-hour cap. A checkpointed budget pause may still need continuation.

Use only the matching training dataset as input. These are fresh runs: attach no old SPECTRA output. New SPECTRA output folders end in `_input_v2`; use that output when continuing a v2 run. Preserve every v1 result separately.

The three full-model runs must be repeated because their Java node features changed. All five controls use that same corrected representation. Existing completed external baselines retain their results because their feature preparation is unchanged.

For a previously paused CodeNet CodeBERT/C4/CDLH/Qwen job, use its original notebook under [the v1 archive](../archive/input_v1/README.md) and its own checkpoint/token cache. The new generated runtime cannot resume an old checkpoint. Do not enable numerical-upgrade overrides.

Observed input-v1 full-model times were about 3.1h CodeNet, 4.1h AtCoder and 1.3h BigCloneBench on T4. These are context, not predictions: v2 convergence and control runtimes are not yet measured. Preserve partial outputs if the session guard stops a run.

See [the input-fix amendment](../00_plan/INPUT_V2_FIX.md) for the evidence, compatibility rules and limits. The machine-readable list is `input_v2_run_queue.csv`.
