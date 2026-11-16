# Next Kaggle runs: matched AtCoder objective confirmation

The short local diagnostics are complete. The next useful experiment trains
on the full audited AtCoder support and needs hours rather than ten minutes.
The six jobs below can run in parallel using six GPU slots.

| Seed | Original full objective | Classification-only objective |
| --- | --- | --- |
| 42 | [atcoder_full_seed42_f1_confirmation_v1.ipynb](atcoder_full_seed42_f1_confirmation_v1.ipynb) | [atcoder_classification_only_seed42_f1_confirmation_v1.ipynb](atcoder_classification_only_seed42_f1_confirmation_v1.ipynb) |
| 123 | [atcoder_full_seed123_f1_confirmation_v1.ipynb](atcoder_full_seed123_f1_confirmation_v1.ipynb) | [atcoder_classification_only_seed123_f1_confirmation_v1.ipynb](atcoder_classification_only_seed123_f1_confirmation_v1.ipynb) |
| 231 | [atcoder_full_seed231_f1_confirmation_v1.ipynb](atcoder_full_seed231_f1_confirmation_v1.ipynb) | [atcoder_classification_only_seed231_f1_confirmation_v1.ipynb](atcoder_classification_only_seed231_f1_confirmation_v1.ipynb) |

## Inputs and settings

1. Import one notebook into Kaggle.
2. Attach **only the existing `atcoder_revision_common_support_v2` dataset**.
   No previous notebook output/checkpoint is needed for a fresh run. No dataset
   rebuild or reupload is required.
3. Enable one T4 GPU and turn Internet off.
4. Keep `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`, `MAX_EPOCHS=100`,
   `MIN_EPOCHS=10`, `PATIENCE=10`, and `SESSION_HOURS=10.0`.
5. Save and run. Preserve every result, including weak or incomplete outcomes.

All six use **F1 for epoch selection, threshold selection and early stopping**.
They share the original input representation, model architecture, AMP/recovery
policy, AdamW learning rate1e-4 and physical/effective batches32/128. The only
objective difference is whether the seven auxiliary weights are active.
Seed42 is retained as a healthy reference, alongside the two previously weak
seeds. There is no architecture or loss-weight search in this batch.

## Why full must be trained again too

The previous full-model checkpoints were selected and stopped using Accuracy.
Recalibrating their thresholds for F1 did not recover the best F1 epoch. Comparing
those old checkpoints against a new F1-selected classification-only model would
change both objective and selection policy. These six prospective jobs apply
the same F1 policy to both objectives. Keep all old results separately; these
jobs do not erase the earlier failed learning or the tiny diagnostic outcomes.

The local controls showed objective effects that vary by seed; removing losses
is a hypothesis to confirm, not an established general repair. Full-precision
training did not rescue the failed seed231 tiny probe, so AMP remains fixed
for this matched batch. Any later precision experiment requires its own plan.

## Time and continuation

Planning estimate per T4 notebook: roughly **1-6 hours if validation patience
stops early**. The old AtCoder timing extrapolates to about **7.7 hours for a
100-epoch ceiling with a planning margin**, not a guarantee for this new F1
schedule. A run can legitimately stop early; do not force it to 100 epochs.

The 10-hour internal session includes a 45-minute reserve, giving approximately
9h15 of work below the 12-hour Kaggle session limit. If the saved status is
incomplete, attach that exact job's output to the same notebook and set
`RESUME_CHECKPOINT` to its `progress.pt`. Keep the matching dataset attached.
Do not attach an old Accuracy-selected checkpoint to this new protocol;
the identity check will reject it.

If any full run scores near 66.67% F1 with AUROC near 0.5, report that as the
all-positive baseline behavior, not recovered learning. F1 remains primary;
AUROC/AP/MCC and confusion counts diagnose misleading threshold outcomes.

## Sending results back

Keep the full outputs on Kaggle for continuation and later validation inference.
After the jobs save outputs, run
[collect_atcoder_f1_confirmation_v1.ipynb](collect_atcoder_f1_confirmation_v1.ipynb)
on **CPU**, Internet off, attaching only these six new output versions.
No training dataset or checkpoint download is needed for the collector.
It exports `atcoder_confirmation_v1_small_reports.zip`, including statuses,
available benchmarks, validation scores and provenance. Incomplete runs are
listed and excluded from ranking.

Save the small report ZIP under:

```text
C:\PyProjects\spectrals\outputs\revesion\06_atcoder_confirmation_v1
```

The queue and notebook hashes are in
[atcoder_f1_confirmation_v1.csv](../queues/atcoder_f1_confirmation_v1.csv).
These six training jobs are prepared, **not yet run**. Final tests stay off.
