# SPECTRA paper revision

**[از اینجا شروع کن — راهنمای فارسی](START_HERE_FA.md)**

This is the single workspace for the new reviewer-response experiments.
Open the `.ipynb` files directly in the folders below and import them into Kaggle.
ZIP packets are optional copies for transport; extraction is not required here.

**Next: [five CodeNet controls](queues/RUN_NOW_INPUT_V2.md).** All three
corrected-input SPECTRA full runs are now [verified complete](results/SPECTRA_INPUT_V2_REVIEW.md):
CodeNet Accuracy 79.67%, AtCoder Accuracy 78.34%, BigCloneBench F1 97.15%.
Reuse these three full models and the completed external baselines. Old
SPECTRA ZIPs are archived outside the active result folder. Existing dataset
uploads stay unchanged. Remaining expensive baselines can wait; final tests stay off.

[AtCoder GMN is now verified](results/GMN_RESULTS_REVIEW.md) at 80.32% Accuracy
and 80.76% F1. The new BigCloneBench GMN upload duplicates its existing reviewed
result (97.77% F1), so it is archived without counting an additional run.

If a text job fails with `RobertaConfig has no attribute '_commit_hash'`, use
the updated notebook in a fresh session; see [HF setup recovery](00_plan/HF_SETUP_RECOVERY.md).
Its dependency installer now enforces the supported versions, without changing
the training implementation or restarting successful graph jobs.

For multi-gigabyte text-run outputs, use the updated
[CPU report collector](05_analysis/validation_screening.ipynb) and download only
`revision_small_reports.zip`. [Instructions](00_plan/SMALL_REPORT_DOWNLOADS.md)
explain how to retain checkpoints on Kaggle while sending small validation reports.

| Folder | Use |
| --- | --- |
| [00_plan](00_plan/README.md) | Full roadmap, Persian plan, protocol and reviewer coverage |
| [01_baselines_seed42](01_baselines_seed42/README.md) | Original 29-job roster; consult tracker before starting a job |
| [02_controls_seed42](02_controls_seed42/README.md) | Next: architecture/loss controls addressing spectral attribution |
| [03_selected_repeats](03_selected_repeats/README.md) | Later: seed 123/231 candidates; run only selected comparisons |
| [04_transfer](04_transfer/README.md) | Later: generate the focused conditions after choosing comparators |
| [05_analysis](05_analysis/README.md) | Validation screening, spectral diagnostics and paired uncertainty |
| [datasets](datasets/README_FA.md) | Exact dataset versions, checksums and reuse rules |
| [queues](queues/README.md) | Exact notebook paths, checksums and 24-job launch list |
| [results](results/README.md) | Run tracker and instructions for retaining Kaggle outputs |
| [optional](optional/README.md) | Frozen Qwen probe; outside the 29-job baseline roster |

The [historical input-v1 review](results/BATCH_2026-11-09.md) contains 17
verified completions and four paused CodeNet runs. Retain those results;
the corrected SPECTRA representation has not yet been trained. To resume an old
external baseline, use its [archived original notebook](archive/input_v1/README.md)
and [saved checkpoint/token cache](00_plan/TEXT_RUN_CONTINUATION.md).
See [the results tracker](results/README.md) for evidence and
[the current launch list](queues/LAUNCH_24.md) before starting another job.

The maintained suite has 193 generated notebooks. The input fix changes their
runtime hash; dataset releases stay unchanged. Twenty-nine original baseline
notebooks and their old protocol/manifest are archived for continuation.
Earlier folder-migration paths are in [queues/path_migration.csv](queues/path_migration.csv).

Dataset ZIPs remain outside Git at `../data/benchmarks/` relative to the project
root. Local release packets, verification reports and archived planning drafts
are under `output/revision/`. Historical paper experiments remain under
`experiments/kaggle/`; their results belong to the original protocol.
