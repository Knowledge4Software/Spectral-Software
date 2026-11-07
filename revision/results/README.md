# Keep experiment outputs here by dataset and job

`run_progress.csv` records reported progress. The first completion comes from
the user's Kaggle log; the complete `status.json` is still needed locally.
Other rows marked `not_reported` do not establish whether a job has started.

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
