# Next: 12 CodeNet core repeats

The six seed-42 conditions are verified complete. This is the 12-run CodeNet core portion of the preplanned 28 selected-repeat jobs; it does not expand the original scope. Run the same six conditions with seeds 123 and 231. No seed-42 run is repeated.

**Every notebook uses the existing `codenet_revision_common_support.zip` dataset.** All 12 can run in parallel with enough GPU slots.

| Slot | Condition | Seed | Notebook |
| ---: | --- | ---: | --- |
| 1 | full | 123 | [codenet_full_seed123.ipynb](../03_selected_repeats/codenet/baselines/codenet_full_seed123.ipynb) |
| 2 | full | 231 | [codenet_full_seed231.ipynb](../03_selected_repeats/codenet/baselines/codenet_full_seed231.ipynb) |
| 3 | mean_max | 123 | [codenet_mean_max_seed123.ipynb](../03_selected_repeats/codenet/architecture/codenet_mean_max_seed123.ipynb) |
| 4 | mean_max | 231 | [codenet_mean_max_seed231.ipynb](../03_selected_repeats/codenet/architecture/codenet_mean_max_seed231.ipynb) |
| 5 | attention | 123 | [codenet_attention_seed123.ipynb](../03_selected_repeats/codenet/architecture/codenet_attention_seed123.ipynb) |
| 6 | attention | 231 | [codenet_attention_seed231.ipynb](../03_selected_repeats/codenet/architecture/codenet_attention_seed231.ipynb) |
| 7 | classification_only | 123 | [codenet_classification_only_seed123.ipynb](../03_selected_repeats/codenet/losses/codenet_classification_only_seed123.ipynb) |
| 8 | classification_only | 231 | [codenet_classification_only_seed231.ipynb](../03_selected_repeats/codenet/losses/codenet_classification_only_seed231.ipynb) |
| 9 | mean_max_classification_only | 123 | [codenet_mean_max_classification_only_seed123.ipynb](../03_selected_repeats/codenet/architecture/codenet_mean_max_classification_only_seed123.ipynb) |
| 10 | mean_max_classification_only | 231 | [codenet_mean_max_classification_only_seed231.ipynb](../03_selected_repeats/codenet/architecture/codenet_mean_max_classification_only_seed231.ipynb) |
| 11 | attention_classification_only | 123 | [codenet_attention_classification_only_seed123.ipynb](../03_selected_repeats/codenet/architecture/codenet_attention_classification_only_seed123.ipynb) |
| 12 | attention_classification_only | 231 | [codenet_attention_classification_only_seed231.ipynb](../03_selected_repeats/codenet/architecture/codenet_attention_classification_only_seed231.ipynb) |

Use GPU (T4 supported), Internet off, `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`. Start each new seed from scratch without old model outputs attached. Keep the encoded seed and all model/loss/training settings unchanged. The 100-epoch ceiling, minimum 10 and patience 10 stay fixed; each session uses the existing 10-hour budget and 45-minute output reserve. A budget pause may still require continuation with that run’s own checkpoint.

Seed-42 recorded times for the six conditions ranged from about 1.7h to 6.0h on T4, including setup and training/validation. Other seeds can stop at different epochs. Timing is an estimate; preserve incomplete outputs.

The purpose is to estimate the readout/objective interaction across seeds. Mean/max classification-only exceeded full by only 10 net correct validation pairs at seed 42; attention classification-only exceeded it by two. Do not infer significance or paired error overlap from those aggregate counts.

The other 15 seed-42 controls, external finalists and transfer study remain in the roadmap. This batch brings the already planned core replication forward before those more costly extensions. Final tests remain off.

See [the audited seed-42 results](../results/CORE_CONTROLS_REVIEW.md). The machine-readable list is `core_repeat_run_queue.csv`.
