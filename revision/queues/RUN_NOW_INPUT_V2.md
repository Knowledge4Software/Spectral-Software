# Run now: input v2, seed 42

Run these **5** unfinished priority jobs in parallel if GPU slots are available. Import the updated local files. External baselines can wait for now.

**Use existing dataset uploads. No new dataset upload or preprocessing job is required.**

| Slot | Dataset | Notebook | Dataset ZIP already uploaded |
| ---: | --- | --- | --- |
| 1 | codenet | [mean_max](../02_controls_seed42/codenet/architecture/codenet_mean_max_seed42.ipynb) | `codenet_revision_common_support.zip` |
| 2 | codenet | [attention](../02_controls_seed42/codenet/architecture/codenet_attention_seed42.ipynb) | `codenet_revision_common_support.zip` |
| 3 | codenet | [classification_only](../02_controls_seed42/codenet/losses/codenet_classification_only_seed42.ipynb) | `codenet_revision_common_support.zip` |
| 4 | codenet | [mean_max_classification_only](../02_controls_seed42/codenet/architecture/codenet_mean_max_classification_only_seed42.ipynb) | `codenet_revision_common_support.zip` |
| 5 | codenet | [attention_classification_only](../02_controls_seed42/codenet/architecture/codenet_attention_classification_only_seed42.ipynb) | `codenet_revision_common_support.zip` |

Settings: GPU on (T4 is supported), Internet off, seed 42, `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`. Keep the default 100-epoch ceiling / minimum 10 / patience 10. Each session has a 10-hour budget with 45 minutes reserved for outputs, below the Kaggle 12-hour cap. A checkpointed budget pause may still need continuation.

Use only the matching training dataset as input. These are fresh runs: attach no old SPECTRA output. New SPECTRA output folders end in `_input_v2`; use that output when continuing a v2 run. Preserve every v1 result separately.

The input-v2 full models and all five controls use the corrected Java node features. Verified current full-model outputs are excluded from this list: reuse them. Existing completed external baselines also retain their results because their feature preparation is unchanged.

For a previously paused CodeNet CodeBERT/C4/CDLH/Qwen job, use its original notebook under [the v1 archive](../archive/input_v1/README.md) and its own checkpoint/token cache. The new generated runtime cannot resume an old checkpoint. Do not enable numerical-upgrade overrides.

Control runtimes are not yet measured. Preserve partial outputs if the session guard stops a run.

Completed input-v2 priority jobs: codenet_full_seed42, atcoder_full_seed42, bigclonebench_full_seed42.

See [the current full-model review](../results/SPECTRA_INPUT_V2_REVIEW.md) and [input-fix amendment](../00_plan/INPUT_V2_FIX.md). The machine-readable list is `input_v2_run_queue.csv`.
