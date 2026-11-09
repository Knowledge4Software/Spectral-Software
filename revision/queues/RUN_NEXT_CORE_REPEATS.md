# Remaining CodeNet core repeats

Original roster: 12 jobs. **8 verified complete, 1 incomplete/needs attention, 3 without submitted results.** Reuse completed outputs. No seed-42 job is repeated.

**Every notebook uses the existing `codenet_revision_common_support.zip` dataset.** These are the seed-123/231 repetitions, separate from the remaining seed-42 ablations.

## No output submitted

| Slot | Condition | Seed | Notebook |
| ---: | --- | ---: | --- |
| 1 | classification_only | 231 | [codenet_classification_only_seed231_validation_recovery.ipynb](../03_selected_repeats/codenet/validation_recovery/codenet_classification_only_seed231_validation_recovery.ipynb) |
| 2 | mean_max_classification_only | 231 | [codenet_mean_max_classification_only_seed231_validation_recovery.ipynb](../03_selected_repeats/codenet/validation_recovery/codenet_mean_max_classification_only_seed231_validation_recovery.ipynb) |
| 3 | attention_classification_only | 123 | [codenet_attention_classification_only_seed123_validation_recovery.ipynb](../03_selected_repeats/codenet/validation_recovery/codenet_attention_classification_only_seed123_validation_recovery.ipynb) |

If one of these is already running on Kaggle, let that existing job finish. No local output does not establish that a job has never started.

## Incomplete / needs attention

| Condition | Seed | Saved status | Action |
| --- | ---: | --- | --- |
| [classification_only](../03_selected_repeats/codenet/validation_recovery/codenet_classification_only_seed123_validation_recovery.ipynb) | 123 | validation_pending | Restart from epoch 1 with the fixed notebook and CodeNet data only; failed checkpoint unavailable. |

For the failed seed 123 follow [validation recovery instructions](../00_plan/VALIDATION_NUMERICAL_RECOVERY.md). The current queue uses a fresh restart because its saved Kaggle checkpoint is unavailable. Attach only CodeNet data. The separately preserved resume notebook is optional if that checkpoint becomes available.

For fresh jobs only: use GPU (T4 supported), Internet off, `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`. Start each new seed from scratch without old model outputs attached. Keep the encoded seed and all model/loss/training settings unchanged. The 100-epoch ceiling, minimum 10 and patience 10 stay fixed; each session uses the existing 10-hour budget and 45-minute output reserve. A budget pause may still require continuation with that run's own checkpoint.

Seed-42 recorded times for the six conditions ranged from about 1.7h to 6.0h on T4, including setup and training/validation. Other seeds can stop at different epochs. Timing is an estimate; preserve incomplete outputs.

The purpose is to estimate the readout/objective interaction across seeds. Mean/max classification-only exceeded full by only 10 net correct validation pairs at seed 42; attention classification-only exceeded it by two. Do not infer significance or paired error overlap from those aggregate counts.

The other 15 seed-42 controls, external finalists and transfer study remain in the roadmap. This batch brings the already planned core replication forward before those more costly extensions. Final tests remain off.

See [the current repeat progress](../results/CORE_REPEATS_PROGRESS.md) and [seed-42 results](../results/CORE_CONTROLS_REVIEW.md). `core_repeat_run_queue.csv` contains only jobs without submitted results; `core_repeat_resume_queue.csv` separates incomplete jobs; `core_repeat_roster.csv` retains all 12 planned jobs.
