# Keep experiment outputs here by dataset and job

**Latest: [batch 03 has eight complete repeats, one validation failure and three missing outputs](CORE_REPEATS_PROGRESS.md).**
All nine supplied archives passed 179 applicable checks; the pending run passed
checkpoint checks only. [Four pending notebooks](../queues/RUN_NEXT_CORE_REPEATS.md)
include the explicit validation numerical repair. Classification-only/123
must continue its own failed checkpoint; the other three use only existing
CodeNet data. Reuse the eight complete outputs. Final tests remain off.

**Seed 42: [the five CodeNet controls passed all 175 checks](CORE_CONTROLS_REVIEW.md).**
The six-cell readout/objective core, including the existing full model, is
complete for seed 42. Mean/max classification-only has 79.74% Accuracy versus
full 79.67%, a 10-net-correct-pair difference. The
[12 core repeats](../queues/RUN_NEXT_CORE_REPEATS.md) use seeds 123 and 231;
eight are now complete, one failed validation and three have no submitted output.
`control_progress.csv` separately records five verified controls and 15
remaining planned controls; the baseline roster still has 18 complete,
four paused and seven without submitted results.

**Current input-v2 results:** [all three SPECTRA full runs passed 102 checks](SPECTRA_INPUT_V2_REVIEW.md)
and are complete: CodeNet Accuracy 79.67%, AtCoder Accuracy 78.34%,
BigCloneBench F1 97.15%. Reuse them. The active baseline roster again has
**18 complete, four paused and seven without submitted results**. The next
priority is now [the core repetitions](../queues/RUN_NEXT_CORE_REPEATS.md).

[Additional GMN review](GMN_RESULTS_REVIEW.md): AtCoder GMN completed 38 epochs,
selected epoch 28, and has 80.32% validation Accuracy / 80.76% F1. Its full
checkpoint passed all 32 checks. The new BigCloneBench upload duplicates all
56 files of the already verified GMN archive; it does not add a new seed/run.

The three old SPECTRA ZIPs were moved to `_archive/input_v1/` in the local
result folder, with hashes preserved. Old scores are excluded from current
rankings and retained in `_audit/audit.json` under `historical_runs`.
The [previous combined review](BATCH_2026-11-09.md) and the CodeNet details
below describe historical input-v1 evidence, not current full-model scores.

`run_progress.csv` distinguishes reported completions, verified local outputs,
resumable budget pauses, and `output_reported_pending_review`. The latter means
the user has a saved output but its completion and metrics have not been checked;
it is excluded from fresh launches. `not_reported` does not establish whether a
job has started. A verified output is not a final test result.

The 2026-11-08 batch is retained at
`C:\PyProjects\spectrals\outputs\revesion\01`. Its `RESULTS_REVIEW.md` and
`_audit/` contain validation scores, integrity checks and the archive rename
manifest. Five CodeNet baselines are complete and verified: ASTNN, RtvNN,
DeepSim, FA-AST-GGNN and FA-AST-GMN. CDLH must resume its saved checkpoint
partway through epoch 9. CodeNet/full/42 is now also verified from its full
archive: 43 completed epochs, best epoch 33, validation Accuracy 79.91% and
F1 80.67%. DeepSim leads the completed runs at 81.33% Accuracy. These are
single-seed validation results; final tests remain off.

The three text summaries have now been reviewed and renamed to
`codenet_text_seed42_session01_reports.zip`. All three paused on the session
budget. CodeBERT and C4 have six validated epochs, with provisional Accuracy
79.42% and 92.66%; Qwen finished epoch-1 training but has no validation result.
All 57 summary checks passed, but their weights were not supplied or audited.
Use [the text continuation guide](../00_plan/TEXT_RUN_CONTINUATION.md) with each
original checkpoint and token cache. The CPU collector still handles only
those three text outputs. CDLH also needs continuation: six CodeNet runs are
complete and four are paused.

Download saved output versions into this local structure when available:

```text
revision/results/kaggle/
  codenet/codenet_full_seed42/<saved-version>/revision_full_seed42/
  atcoder/atcoder_astnn_seed42/<saved-version>/...
  bigclonebench/bigclonebench_codebert_seed42/<saved-version>/...
```

`kaggle/` is ignored by Git. Preserve the whole output: `progress.pt`,
`status.json`, `benchmark.json`, numerical events and transformer token caches.
Keep older output versions when continuing a session. Final tests are released
later, after choices are frozen; retain their predictions and metadata too.

Run the CPU validation collector on attached Kaggle outputs. Do not fill in
unreported best metrics or final scores from an intermediate epoch log.
