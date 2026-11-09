# Input-v2 priority batch is complete

All eight priority jobs are verified complete: full SPECTRA on the three datasets and five CodeNet readout/objective controls. Reuse them; no seed-42 rerun is queued.

See [the six-condition core results](../results/CORE_CONTROLS_REVIEW.md). The next planned batch is [12 CodeNet core repeats](RUN_NEXT_CORE_REPEATS.md): the same six conditions with seeds 123 and 231. Existing CodeNet data, models, losses and training settings are reused.

The other 15 seed-42 architecture/loss controls remain in the roadmap; they have not been completed or removed. Expensive external baselines and the transfer study remain deferred pending the central comparison. Final tests stay off.

Completed input-v2 priority jobs: codenet_full_seed42, atcoder_full_seed42, bigclonebench_full_seed42, codenet_mean_max_seed42, codenet_attention_seed42, codenet_classification_only_seed42, codenet_mean_max_classification_only_seed42, codenet_attention_classification_only_seed42.

The machine-readable `input_v2_run_queue.csv` has a header and zero pending rows.
