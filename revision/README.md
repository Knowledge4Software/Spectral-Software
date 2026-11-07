# SPECTRA paper revision

**[از اینجا شروع کن — راهنمای فارسی](START_HERE_FA.md)**

This is the single workspace for the new reviewer-response experiments.
Open the `.ipynb` files directly in the folders below and import them into Kaggle.
ZIP packets are optional copies for transport; extraction is not required here.

| Folder | Use |
| --- | --- |
| [00_plan](00_plan/README.md) | Full roadmap, Persian plan, protocol and reviewer coverage |
| [01_baselines_seed42](01_baselines_seed42/README.md) | **Run now:** 29 jobs across the three datasets |
| [02_controls_seed42](02_controls_seed42/README.md) | Next: architecture/loss controls addressing spectral attribution |
| [03_selected_repeats](03_selected_repeats/README.md) | Later: seed 123/231 candidates; run only selected comparisons |
| [04_transfer](04_transfer/README.md) | Later: generate the focused conditions after choosing comparators |
| [05_analysis](05_analysis/README.md) | Validation screening, spectral diagnostics and paired uncertainty |
| [datasets](datasets/README_FA.md) | Exact dataset versions, checksums and reuse rules |
| [queues](queues/README.md) | Exact notebook paths, checksums and 24-job launch list |
| [results](results/README.md) | Run tracker and instructions for retaining Kaggle outputs |
| [optional](optional/README.md) | Frozen Qwen probe; outside the 29-job baseline roster |

On 2026-11-07 the user reported CodeNet/full/seed42 completed at epoch 43 on a
Tesla T4, with test evaluation off. Its saved `status.json` has not been imported
locally. Reuse that successful run; obtain its best validation fields before
assessing accuracy. The remaining 28 baseline jobs are independent.

The folder reorganization preserves the bytes of all 193 existing notebooks,
the training implementation hash and dataset releases. Old-to-new paths are
listed in [queues/path_migration.csv](queues/path_migration.csv). Retain the
source notebook used by an existing run when continuing its checkpoint.

Dataset ZIPs remain outside Git at `../data/benchmarks/` relative to the project
root. Local release packets, verification reports and archived planning drafts
are under `output/revision/`. Historical paper experiments remain under
`experiments/kaggle/`; their results belong to the original protocol.
