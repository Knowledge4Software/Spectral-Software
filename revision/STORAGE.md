# Where to find datasets and results

Paths below start at `C:/PyProjects/spectrals/`, the parent of this repository.

```text
data/
  benchmarks/              three current Kaggle ZIPs + SHA-256 files
  working/revision/        unpacked copies for local audits
  sources/                 upstream downloads and prepared corpora
  archive/
    original_exports/      earlier paper dataset ZIPs
    preparations/          superseded preparation versions
    transfer_bundles/      earlier transfer inputs
outputs/
  revesion/                active reviewer-response results
    01/                    baselines and full models, seed 42
    02/                    CodeNet controls, seed 42
    03/                    CodeNet core repeats, seeds 123/231
  paper/
    kaggle/                original submitted-paper result archives/figures
    local_spectral_representation_baselines/
Spectral-Software/
  revision/                current notebooks, queues, protocol and trackers
  output/revision/         local release/audit/maintenance artifacts
```

Use [the dataset index](../../data/benchmarks/README.md) for uploads and
[the result index](../../outputs/revesion/README.md) for returned ZIPs.
Batch 02 has [one current summary](../../outputs/revesion/02/RESULTS_REVIEW.md)
and [a CSV of all 21 completed seed-42 reference/control results](../../outputs/revesion/02/validation_summary.csv).
The [independent queue](queues/RUN_NEXT_INDEPENDENT_BATCH.md) lists only AtCoder/full/231
without submitted output; check Kaggle for live jobs before launching duplicates.

The `revesion` spelling remains for compatibility with existing analysis paths.
Current Kaggle uploads need no rename, rebuild or replacement. Raw metadata and
historical audit records retain their original paths; the maintenance move map
resolves those paths to the current locations. Failed/incomplete attempts and
unfavorable results remain preserved, separately from verified completed runs.

The first cleanup removed 1.65 GiB of verified redundant copies. The second
organization made 24 moves and deleted no scientific files. Audit inventories,
move logs and document backups are under
`output/revision/maintenance/2026-11-11/` inside the repository's local output.
