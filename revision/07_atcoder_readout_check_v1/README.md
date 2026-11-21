# Run next: three AtCoder mean/max controls

These three notebooks were submitted from Spain on2026-11-21. **Do not upload
and run duplicate copies.** The server monitor checks them every3min and
collects small reports. See the [server guide](../queues/KAGGLE_SERVER.md).

| Seed | Live notebook | Status at submission check |
| --- | --- | --- |
| 42 | [account01 / version1](https://www.kaggle.com/code/koushamoeini/spectra-atcoder-meanmax-f1-seed42-v1) | Running |
| 123 | [account02 / version1](https://www.kaggle.com/code/drmorgan00/spectra-atcoder-meanmax-f1-seed123-v1) | Running |
| 231 | [account03 / version1](https://www.kaggle.com/code/frankarianpour/spectra-atcoder-meanmax-f1-seed231-v1) | Queued |

These are a snapshot, not live statuses. `/opt/spectral-kaggle/state.json` on
Spain is the current monitoring ledger. Scientific completion remains pending.

| Seed | Notebook |
| --- | --- |
| 42 | [atcoder_mean_max_classification_only_seed42_readout_check_v1.ipynb](atcoder_mean_max_classification_only_seed42_readout_check_v1.ipynb) |
| 123 | [atcoder_mean_max_classification_only_seed123_readout_check_v1.ipynb](atcoder_mean_max_classification_only_seed123_readout_check_v1.ipynb) |
| 231 | [atcoder_mean_max_classification_only_seed231_readout_check_v1.ipynb](atcoder_mean_max_classification_only_seed231_readout_check_v1.ipynb) |

Attach **only the existing `atcoder_revision_common_support_v2` dataset**.
No previous notebook outputs or checkpoints are needed for these fresh jobs.
GPU: one T4. Internet: off. Keep `FINAL_TEST=False`, `RESUME_CHECKPOINT=None`
and the other default settings. No dataset reupload is required.

The three spectral classification-only runs already completed in phase06 are
the matched references; **do not train them again**. Their mean validation F1
is81.89%. Keep the three failed full-objective outcomes as separate evidence.

## Purpose and selection rule

This is a control experiment, not a change to the proposed spectral method.
It addresses whether the spectral readout adds value beyond simple pooling on
AtCoder under the working classification-only objective. Mean/max was selected
before these new AtCoder readout results because it had the highest three-seed
mean F1 among the simple CodeNet controls (80.89% classification-only, versus
attention80.60%). This implements the optional cross-benchmark sibling check
with an explicit extra-seed ledger; it does not require every optional notebook.

Shared with the completed spectral references: dataset/support, encoder,
zero auxiliary weights, AdamW learning rate1e-4, physical/effective batch32/128,
AMP/recovery policy, F1 epoch and threshold selection, 100-epoch ceiling,
minimum10 and patience10. Readout heads differ in parameter count; retain the
benchmark reports rather than claiming identical capacity.

All seeds and outcomes must be reported. This comparison is conditional on the
classification-only objective; it cannot prove universal spectral superiority
or repair the original full-loss failures. If mean/max matches or exceeds the
spectral references, that result must remain in the paper. Original CodeNet
results were calibrated at Accuracy-selected epochs; that selection limitation
remains explicit and is not retroactively removed.

## Time and continuation

Plan for approximately **2-6 hours per notebook on T4** if patience stops early.
The 100-epoch ceiling extrapolates to roughly8h with margin, not a guarantee;
the new readout may differ. Every session retains a 10h budget with45min output
reserve, below the 12-hour Kaggle session limit. A budget-paused job is incomplete:
attach its own output to the same notebook and set `RESUME_CHECKPOINT` to its
`progress.pt`. Do not resume a spectral checkpoint into a mean/max job.

## Return the results

After these three jobs, run
[collect_atcoder_readout_check_v1.ipynb](collect_atcoder_readout_check_v1.ipynb)
on CPU, Internet off. Attach six saved outputs:

- New mean/max classification-only, seeds42/123/231.
- Completed phase06 spectral `classification_only`, seeds42/123/231.

No training dataset or checkpoint download is needed for this collector.
Download `atcoder_readout_check_v1_small_reports.zip` and put it in:

```text
C:\PyProjects\spectrals\outputs\revesion\07_atcoder_readout_check_v1
```

Keep the full saved outputs on Kaggle. The small report supports status/metric
review but does not independently verify weights. All three training jobs were
submitted; the [progress tracker](../results/atcoder_readout_progress.csv)
records their submission evidence. The original
[queue](../queues/atcoder_readout_check_v1.csv) and JSON manifest remain the
pre-run registration, retaining input/reference links and notebook hashes.

Modern baselines, controlled transfer and final-test uncertainty remain further
review gaps. These three controls do not complete the entire revision.
