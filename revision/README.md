# SPECTRA paper revision

**[از اینجا شروع کن — راهنمای فارسی](START_HERE_FA.md)**

This is the single workspace for the new reviewer-response experiments.
Open the `.ipynb` files directly in the folders below and import them into Kaggle.
ZIP packets are optional copies for transport; extraction is not required here.

**Run next: [four inference-only checkpoint review notebooks](05_analysis/continuation/README_FA.md),
then their CPU collector.** They diagnose AtCoder's weak seeds and compare all
18 CodeNet core checkpoints with validation F1 thresholds. Existing training
outputs and dataset uploads are reused; new training is not requested yet.

**[All 20 CodeNet controls are verified complete](results/README.md).** All 12 CodeNet core repeats are also complete. The latest seven uploads passed 252/252 checks. Reuse completed results; final tests remain off.

**[No fresh launches remain in the independent roster](queues/RUN_NEXT_INDEPENDENT_BATCH.md).** All 19 job identities have completed evidence. Two distinct BCB/full/231 attempts remain pending provenance; neither is selected for a seed average. AtCoder/full/123 and /231 completed with weak F1 56.10% and 47.43%; both are retained. See [the follow-up review](results/FOLLOWUP_RESULTS_REVIEW.md). Diagnose this instability before more training. Older baseline and transfer work remains separate.

The current uploads are in `../data/benchmarks/`; unpacked audit copies are in
`../data/working/revision/`. Original paper inputs/results are grouped under
`../data/archive/` and `../outputs/paper/`. Reviewer-response outputs remain at
`../outputs/revesion/01`, `02` and `03`.

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
| [02_controls_seed42](02_controls_seed42/README.md) | All 20 CodeNet controls verified complete |
| [03_selected_repeats](03_selected_repeats/README.md) | All 12 core repeats complete; all full-repeat identities supplied; BCB/231 provenance unresolved |
| [04_transfer](04_transfer/README.md) | Later: generate the focused conditions after choosing comparators |
| [05_analysis](05_analysis/README.md) | Validation screening, spectral diagnostics and paired uncertainty |
| [datasets](datasets/README_FA.md) | Exact dataset versions, checksums and reuse rules |
| [queues](queues/README.md) | Exact notebook paths, checksums and 24-job launch list |
| [results](results/README.md) | Run tracker and instructions for retaining Kaggle outputs |
| [optional](optional/README.md) | Frozen Qwen probe; outside the 29-job baseline roster |

The [historical input-v1 review](results/BATCH_2026-11-09.md) contains 17
verified completions and four paused CodeNet runs. Retain those results;
the corrected SPECTRA representation is recorded separately above. To resume an old
external baseline, use its [archived original notebook](archive/input_v1/README.md)
and [saved checkpoint/token cache](00_plan/TEXT_RUN_CONTINUATION.md).
See [the results tracker](results/README.md) for evidence and
[the current launch list](queues/LAUNCH_24.md) before starting another job.

The maintained suite has 193 frozen generated notebooks plus separately
versioned numerical-fix copies listed in the queue manifests. Dataset releases
stay unchanged. Twenty-nine original baseline
notebooks and their old protocol/manifest are archived for continuation.
Earlier folder-migration paths are in [queues/path_migration.csv](queues/path_migration.csv).

Dataset ZIPs remain outside Git at `../data/benchmarks/` relative to the project
root. Local release packets, verification reports and archived planning drafts
are under `output/revision/`. Historical paper experiments remain under
`experiments/kaggle/`; their results belong to the original protocol.
