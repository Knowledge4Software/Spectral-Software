# CodeNet core readout/objective controls — seed 42

**All five new input-v2 controls are complete and verified. All 175 new checks passed. Together with the already completed full model, the six-condition core is complete for seed 42. None needs continuation or a seed-42 rerun.**

**Current batch status:** [20/20 controls are verified](README.md), including [seven loss controls](LOSS_CONTROLS_REVIEW.md). All [12 core repeats](CORE_REPEATS_PROGRESS.md) are also complete. The seed-42 evidence and batch counts below describe the original core audit.

The five ZIPs in `C:\PyProjects\spectrals\outputs\revesion\02` were renamed by dataset, arm, seed, input version and status. All archive hashes were preserved. The reference full model remains in batch `01`; its old input-v1 result is not used.

Every comparison below uses the same CodeNet support (14,896 validation pairs; 7,448 positives), input-v2 runtime, shared encoder settings, seed 42, physical/effective batches 32/128 and validation-Accuracy selection policy. Each run followed minimum 10 epochs, a 100-epoch ceiling and patience 10. All final tests remain off. Readout heads have different parameter counts.

## The three readouts with and without auxiliary objectives

| Readout | With auxiliaries: Accuracy | Classification only: Accuracy | With auxiliaries: F1 | Classification only: F1 |
| --- | ---: | ---: | ---: | ---: |
| Spectral | 79.67% | 77.21% | 80.22% | 78.31% |
| Mean/max | 79.09% | 79.74% | 79.42% | 79.68% |
| Attention | 78.52% | 79.69% | 77.76% | 80.13% |

Accuracy is the prespecified primary metric on CodeNet. F1 is secondary; do not switch the primary metric because the full model has the highest F1 in this seed.

## What the comparisons establish

- Mean/max classification-only has 79.7395% Accuracy versus full 79.6724%, a **0.0671 percentage-point lead** or 10 more correct validation decisions in aggregate. Attention classification-only has 79.6858%, a **0.0134-point lead** or two more correct decisions in aggregate. These totals do not reveal paired disagreement counts; no paired significance test can be derived from confusion matrices alone.
- Among the three models using the same auxiliary weights, full spectral is ahead: 79.67% versus 79.09% mean/max and 78.52% attention. Its margins are 0.58 and 1.15 points, respectively.
- With classification-only training, spectral is lower: 77.21% versus 79.74% mean/max and 79.69% attention. The spectral classification-only run trained for 86 epochs, selected epoch 76 and stopped on patience; this is not a four-epoch or session-budget failure.
- The observed effect of auxiliaries differs by readout: **+2.46 Accuracy points for spectral, -0.65 for mean/max and -1.17 for attention**. These are seed-specific differences between validation-selected runs, not proof of a universal causal effect or objective necessity.
- The strongest simple readouts match or slightly exceed the full model on the primary metric in this seed. These results do not establish that spectral readout is necessary or generally superior. Repeat the full six-cell comparison before drawing a stable conclusion.

This directly addresses the reviewer request to distinguish readout from auxiliary objectives. It provides evidence to revise the claims; it does not by itself settle the remaining reviewer issues or guarantee acceptance.

## Stopping, compute and integrity

| Condition | Epochs | Best epoch | Parameters | Recorded hours | Inference pairs/s | Forward/backward pairs/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spectral + auxiliaries | 39 | 29 | 2,977,383 | 2.70 | 1467 | 482 |
| Mean/max + auxiliaries | 31 | 21 | 3,053,311 | 2.10 | 1364 | 495 |
| Attention + auxiliaries | 24 | 14 | 3,014,656 | 1.73 | 1428 | 505 |
| Spectral, classification only | 86 | 76 | 2,977,383 | 6.02 | 1350 | 489 |
| Mean/max, classification only | 40 | 30 | 3,053,311 | 2.68 | 1481 | 544 |
| Attention, classification only | 30 | 20 | 3,014,656 | 2.03 | 1518 | 535 |

Current and best model states loaded strictly into their planned architectures. Weights, optimizer/scaler/history values are finite. Archived training hashes match the reference full model, and per-arm model configuration and all auxiliary weights match `arm_settings`. Dataset identities, metrics reconstructed from TP/FP/TN/FN, best epochs/thresholds, RNG states, update counts, completed cursors and patience counters are consistent.

The attention classification-only run used one recovered FP32 fallback; other logged recovery actions were same-group AMP retries. There is no fatal numerical artifact or unrecovered fallback. Successful replay preserves the update cursor; these events are not skipped batches.

The saved benchmarks use the same Tesla T4, Torch version, physical/effective batches, warmup setting and seeded 256-pair support as the reference full model. They measure model compute after device input and exclude loading, transfers, optimizer steps and checkpoint I/O. Timings are single-run measurements, not uncertainty estimates. Recorded training hours include setup and training/validation but exclude final benchmarking/export.

AUROC/AP require validation predictions for independent reproduction. No new GPU training or held-out test evaluation was performed in this review.

## Next planned batch

Reuse all six seed-42 results. The existing plan calls for the six core arms to receive seeds 123 and 231: **12 new jobs**, with no seed-42 repetition. [The exact notebook list](../queues/RUN_NEXT_CORE_REPEATS.md) uses the existing CodeNet dataset and unchanged input-v2 settings. All six conditions are repeated to estimate the readout/objective interaction across seeds.

This brings the preplanned core repetitions forward in execution order. The remaining 15 seed-42 architecture/loss controls are still in the roadmap and are not marked complete or removed. Broad external repetitions and transfer runs remain deferred while the central claim is assessed.

## Tracking and files

- Baseline roster: 18 verified completions, four paused jobs and seven without submitted results; unchanged by the five control jobs.
- Control roster: five verified completions and 15 without submitted results, tracked separately in `revision/results/control_progress.csv`.
- `_audit/core_controls_audit.json`: all 175 new checks and the reference full-model record.
- `_audit/core_results.csv`: six-condition metrics, epochs, parameters and compute measurements.
- `_audit/core_comparisons.json`: exact changes in Accuracy and net correct counts.
- `_audit/archive_rename_manifest.json`: original names, current names and preserved SHA-256 hashes.

These are validation-only exploratory point estimates from one training seed. Keep the unfavorable spectral classification-only result and all other conditions when revising the paper.
