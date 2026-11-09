# CodeNet core repeats: complete

**All 12 seed-123/231 core repeats are verified complete.** Classification-only/123 finished its fresh restart after 53 epochs and selected epoch 43. The latest archive passed all 36 checks. Across the 12 authoritative repeats there are 256 passed checks; the historical failed attempt has another 19 checkpoint checks and is excluded from completed results.

The six seed-42 conditions are reused, so the complete core now contains six conditions x three seeds = 18 completed runs. All use the same input-v2 CodeNet support (69538/14896/14906 train/validation/test pairs) and frozen model/loss/stopping settings for each arm. The explicit numerical-only runtime amendment is retained in original hashes and comparison metadata. Final tests remain off.

| Condition | Seed | Validated epochs | Best epoch | Validation Accuracy | Validation F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 123 | 40 | 30 | 79.26% | 79.91% |
| full | 231 | 36 | 26 | 79.20% | 79.02% |
| mean_max | 123 | 31 | 21 | 79.71% | 80.16% |
| mean_max | 231 | 35 | 25 | 79.56% | 80.32% |
| attention | 123 | 28 | 18 | 78.79% | 79.04% |
| attention | 231 | 26 | 16 | 78.56% | 78.75% |
| classification_only | 123 | 53 | 43 | 80.58% | 80.99% |
| classification_only | 231 | 36 | 26 | 80.87% | 81.28% |
| mean_max_classification_only | 123 | 30 | 20 | 79.87% | 80.81% |
| mean_max_classification_only | 231 | 36 | 26 | 80.14% | 80.51% |
| attention_classification_only | 123 | 33 | 23 | 79.54% | 79.49% |
| attention_classification_only | 231 | 26 | 16 | 79.67% | 79.92% |

## Successful seed-123 restart

The new classification-only/123 run has Accuracy 80.58%, F1 80.99%, AUROC 0.8899, and AP 0.8841. Recorded setup plus training/validation was 3.52h. It stopped normally at validation patience 10, with batch cursor zero and 28832 optimizer updates. Current/best weights and optimizer/scaler/history are finite and load consistently. Identities, settings, metrics reconstructed from confusion counts, thresholds, RNG state and benchmark all passed.

It logged 14 recovered same-group AMP retries, no FP32 training fallback and no FP32 validation retry. Thus the fixed notebook completed on T4, but this successful run did not exercise the conditional FP32 prediction-recovery branch. It is a fresh run with empty migration lineage, not a checkpoint continuation.

## Earlier failed attempt retained

The earlier classification-only/123 run failed validation after epoch 43 training, with 42 validated epochs and provisional best Accuracy 70.50%. Its archive was moved, with SHA-256 preserved, to `_archive/failed_attempts/codenet_classification_only_seed123_input_v2_validation_pending.zip`. It is retained under `historical_failed_attempts` in the audit. The new archive is `codenet_classification_only_seed123_input_v2_validation_fix_restart_complete.zip`.

The restart was requested because the failed Kaggle output could not be reattached, not as a search for a better score. Count only the completed restart for seed 123; do not average both attempts as independent seeds. The finite AMP training path/settings were unchanged; no bitwise reproducibility across GPU restarts is claimed. Extract only authoritative completed archives for analysis and keep historical failures outside the active collection.

## Three-seed descriptive validation summary

| Condition | Accuracy seed 42 | Accuracy seed 123 | Accuracy seed 231 | Mean Accuracy +/- sample SD | Mean F1 +/- sample SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 79.67% | 79.26% | 79.20% | 79.38 +/- 0.26% | 79.72 +/- 0.62% |
| mean_max | 79.09% | 79.71% | 79.56% | 79.45 +/- 0.32% | 79.97 +/- 0.48% |
| attention | 78.52% | 78.79% | 78.56% | 78.62 +/- 0.15% | 78.51 +/- 0.67% |
| classification_only | 77.21% | 80.58% | 80.87% | 79.55 +/- 2.04% | 80.19 +/- 1.64% |
| mean_max_classification_only | 79.74% | 79.87% | 80.14% | 79.91 +/- 0.20% | 80.33 +/- 0.59% |
| attention_classification_only | 79.69% | 79.54% | 79.67% | 79.63 +/- 0.08% | 79.85 +/- 0.32% |

These means and sample SD describe three validation-selected runs, not confidence intervals, final-test performance or statistical significance. Accuracy remains the prespecified primary metric. Mean/max classification-only has the highest descriptive mean Accuracy (79.91%); full spectral has 79.38%. There is no consistent demonstrated accuracy advantage for full spectral readout in this core. Classification-only spectral has substantial across-seed variation; auxiliary effects must be described by readout and seed, not claimed universally beneficial. Independent test reporting and paired uncertainty remain future work after choices are frozen.

## Next work

No job from these 12 needs continuation or repetition. [The independent batch](../queues/RUN_NEXT_INDEPENDENT_BATCH.md) still has 11 CodeNet seed-42 controls and four AtCoder/BigCloneBench full repeats without supplied output; some may already be running. Four additional graph/descriptor controls are already [verified](ARCHITECTURE_CONTROLS_REVIEW.md). External baseline completion, other selected repeats, transfer, diagnostics and manuscript revision remain separate tasks. Keep final tests off.

Local evidence: `_audit/repeat_inventory.json`, `_audit/classification_only_seed123_restart_audit.json`, `_audit/core_three_seed_validation.csv`, `_audit/core_three_seed_summary.json`, and `core_repeat_progress.csv`. Archive renaming/moving changed no bytes. AUROC/AP were range/finiteness checked, not independently recomputed without validation predictions. No new training or test evaluation was performed during this audit.
