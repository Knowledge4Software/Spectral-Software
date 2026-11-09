# CodeNet core repeats: batch 03 review

**8 verified complete, 1 failed during validation, 3 without submitted output**, from the 12 planned seed-123/231 core jobs. All nine archives were renamed with condition/seed/status; all 179 applicable integrity/result checks passed. The failed run passed checkpoint checks, not completion checks.

All outputs use input v2 and the same CodeNet support (69538 training, 14896 validation, 14906 test pairs); validation has 7448 positives and 7448 negatives. Data/settings/models/losses match the planned jobs. Current/best weights and optimizer state are finite, histories and best-epoch selection agree, tests remain off. Each completed job stopped on the prescribed validation patience.

| Condition | Seed | Validated epochs | Best epoch | Validation Accuracy | Validation F1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full | 123 | 40 | 30 | 79.26% | 79.91% | complete_verified |
| full | 231 | 36 | 26 | 79.20% | 79.02% | complete_verified |
| mean_max | 123 | 31 | 21 | 79.71% | 80.16% | complete_verified |
| mean_max | 231 | 35 | 25 | 79.56% | 80.32% | complete_verified |
| attention | 123 | 28 | 18 | 78.79% | 79.04% | complete_verified |
| attention | 231 | 26 | 16 | 78.56% | 78.75% | complete_verified |
| classification_only | 123 | 42 | 33 | 70.50% | 72.70% | validation_pending |
| classification_only | 231 | - | - | - | - | not_reported |
| mean_max_classification_only | 123 | 30 | 20 | 79.87% | 80.81% | complete_verified |
| mean_max_classification_only | 231 | - | - | - | - | not_reported |
| attention_classification_only | 123 | - | - | - | - | not_reported |
| attention_classification_only | 231 | 26 | 16 | 79.67% | 79.92% | complete_verified |

## Failed classification-only seed 123

This is the former `results(8).zip`, now `codenet_classification_only_seed123_input_v2_validation_pending.zip`. The user supplied the traceback: `choose_threshold` rejected non-finite validation probabilities after epoch 43 training. This is **not a session-budget stop**. The validation set has both classes.

There are 42 validated epochs, provisional best epoch 33, and nine non-improving validations. Epoch 43 training is fully saved at batch 2174 / 23392 optimizer updates. Training/validation took about 2.76 hours plus 0.15 hours setup. The checkpoint contains finite weights/optimizer state, but training recorded 53 FP32 fallback recoveries and the scaler reached float32 tiny. Do not count the provisional 70.50% Accuracy as a completed run or average it with completed seeds.

[The recovery instructions](../00_plan/VALIDATION_NUMERICAL_RECOVERY.md) preserve this checkpoint/history and retry validation first with conditional whole-pass FP32 fallback. Continuing the original unpatched notebook can reproduce the error. No optimizer cursor, data, seed or loss is reset.

Full/231 recorded three recovered FP32 training groups and completed normally with finite state. Logged, recovered AMP overflows alone are not a reason to discard a run.

## What remains

- Resume classification-only/123 using its dedicated recovery notebook and its own saved Kaggle output.
- No submitted output: classification-only/231, mean-max-classification-only/231, attention-classification-only/123. Use their fixed notebooks if starting now; do not duplicate any existing live job.
- Reuse the eight complete repeats. All four notebooks use the existing CodeNet dataset. GPU on, Internet off, final tests off.

[Exact pending notebooks](../queues/RUN_NEXT_CORE_REPEATS.md). The complete 12-job original roster remains in `queues/core_repeat_roster.csv`; the tracker includes all 12 statuses. These jobs are separate from the 15 deferred seed-42 controls, external baselines and transfer study.

## Interpretation and limits

These are validation scores, not final-test performance. The completed full model has 79.67%, 79.26%, 79.20% Accuracy for seeds 42, 123, 231. Mean/max with the shared auxiliary objective has 79.09%, 79.71%, 79.56%. This does not show a consistent accuracy advantage for spectral readout. Finish the missing conditions and audit the failed seed before the balanced readout/objective analysis; no aggregate superiority or statistical significance is claimed here.

Detailed local evidence: `_audit/repeat_inventory.json`, archive SHA-256 values and `core_repeat_progress.csv`. Renaming changed no archive bytes. Validation recovery changes only the explicitly versioned numerical handling and retains original runtime hashes/lineage. GPU recovery still needs a Kaggle run.
