# Keep experiment outputs here by dataset and job

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
F1 80.67%. DeepSim leads this batch at 81.33% Accuracy. These are single-seed
validation results; final tests remain off. The three text-run reports are
still pending review. The CPU collector continues to need only those three.

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
