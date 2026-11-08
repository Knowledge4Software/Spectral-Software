# Keep experiment outputs here by dataset and job

**Input-v2 amendment:** the 17 completions below are historical input-v1
evidence. Fourteen external baselines remain reusable; the three SPECTRA full
runs need fresh corrected-input training. See [the amendment](../00_plan/INPUT_V2_FIX.md)
and [eight immediate jobs](../queues/RUN_NOW_INPUT_V2.md). Do not mix old
SPECTRA full checkpoints with the corrected controls.

**2026-11-09 update:** [the combined validation review](BATCH_2026-11-09.md)
adds eleven verified completions: five AtCoder and six BigCloneBench runs.
The 29-job roster now has **17 complete, four paused, eight without submitted
results**. Reuse every completed run. The older CodeNet evidence below remains
valid. SPECTRA has 76.10% AtCoder Accuracy and 67.92% BigCloneBench F1, below
the leading completed baselines at 82.91% and 98.27%, respectively.
Input diagnostics found extremely sparse BigCloneBench node lexical features;
the cause of the performance gap is not yet established. Prioritize the central
matched controls before launching all repeat seeds or transfer conditions.

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
