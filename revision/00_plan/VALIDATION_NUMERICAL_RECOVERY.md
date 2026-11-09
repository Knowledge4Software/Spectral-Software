# CodeNet classification-only seed 123: recover validation

**Current choice: restart from scratch.** The user cannot reattach the failed
Kaggle output. Import [the fresh seed-123 notebook](../03_selected_repeats/codenet/validation_recovery/codenet_classification_only_seed123_validation_recovery.ipynb)
into a fresh Kaggle session. Attach only the existing CodeNet dataset, GPU on,
Internet off, then Save & Run. `RESUME_CHECKPOINT=None`,
`ALLOW_NUMERICAL_UPGRADE=False`, `FINAL_TEST=False` are already set.
This repeats training from epoch 1 with seed 123 and the same model, losses,
data and stopping schedule, including the validation fix. The previous
42 validated epochs cannot be reused without their checkpoint. Preserve the
failed archive separately; it is an incomplete attempt, not another completed seed.
Outputs use `_input_v2_restart_validation_fix` to distinguish the restart.
The eight completed repeats and three other pending notebooks are unchanged.

Regenerate the current four-job queue with
`python -m scripts.prepare_revision_validation_recovery --restart-seed123`,
then `python -m scripts.prepare_revision_core_repeats`.
The original resume notebook below is retained for use only if its checkpoint
becomes available. The fresh restart is the current queued choice.

This run failed during validation after epoch 43 training. It is not a time-limit
pause and is not complete. There are 42 validated epochs, best epoch 33, with
provisional validation Accuracy 70.50%. Keep this result; do not replace it with
a different seed or treat it as a completed score.

The supplied checkpoint contains finite current/best weights and optimizer state.
Its epoch-43 training cursor is at batch 2174 with 23392 optimizer updates.
Both validation classes are present (7448 each). The error is non-finite model
probabilities reaching threshold selection. Training AMP replay does not protect
validation; this run also recorded 53 FP32 training-group recoveries and an
extremely small AMP loss scale. The scale alone does not establish a diagnosis
of model quality.

## Continue this failed run

1. Import [the recovery notebook](../03_selected_repeats/codenet/validation_recovery/codenet_classification_only_seed123_resume_validation.ipynb)
   into a new Kaggle session.
2. Attach the existing `codenet_revision_common_support` dataset and the **saved
   output of the failed classification-only seed-123 job**. Its folder must
   contain `revision_classification_only_seed123_input_v2/progress.pt`.
   Attach the Kaggle output directly; no multi-GB download/upload is required.
3. Enable GPU (T4 supported), Internet off, then Save & Run. The first cell
   automatically locates exactly one matching checkpoint. Keep `FINAL_TEST=False`.
4. The runtime checks data, architecture, losses, seed, schedule and original
   implementation identity. It records the explicit approved migration,
   restores optimizer/scaler/RNG/history and retries epoch 43 validation before
   doing further optimizer updates.

If an AMP prediction pass contains non-finite probabilities, the entire pass is
retried once in FP32 using all pairs. Scores are never clipped, replaced or
discarded. `validation_numerical_events` records the recovery. If FP32 also
fails, `failed_validation_nonfinite` is saved and the run stops for diagnosis.
This repair cannot guarantee that an unresolved FP32 model failure will recover.

Normal early stopping remains unchanged: minimum 10 epochs, patience 10,
maximum 100. There were nine non-improving validations before the error; the
next validation determines whether patience stops the run or training continues.
Each session remains 10 hours including the 45-minute output reserve.

## Three jobs without submitted results

Use the three other notebooks in [validation_recovery](../03_selected_repeats/codenet/validation_recovery/)
for classification-only/231, mean-max-classification-only/231 and
attention-classification-only/123. Attach only the existing CodeNet dataset,
keep `RESUME_CHECKPOINT=None`, and start from scratch **if no existing Kaggle
run is already underway**. They include the same conditional validation repair.

## Provenance and reuse

Original input-v2 hash:
`8ad938f286142084e2c89ef8439bf9310db61e30019f3d61d02b90426df52e26`.
Recovery runtime:
`bc9d583ca609c7929cf4373737e07a53b1bfdc79e14a23672e40cb3d8d02e00b`.
Only `training.py` and `numerics.py` differ within the scientific source set.
Finite AMP predictions and optimizer updates retain the same path; failed
prediction passes now get the explicit whole-pass FP32 retry. Original hashes
and migration lineage remain in outputs. Analysis treats only this exact
numerical amendment as comparable to input v2; it does not admit old input-v1
SPECTRA outputs or arbitrary runtime changes.

The eight completed repeats are reused, with no rerun queued. The 193 frozen
notebooks are preserved; four separately named recovery notebooks are listed in
`queues/validation_recovery_manifest.csv`. Generate them with
`python -m scripts.prepare_revision_validation_recovery`, then refresh the
pending roster with `python -m scripts.prepare_revision_core_repeats`.

Local checks cover finite-path preservation, whole-pass retry, persistent failure,
deadline handling, exact resume without repeated optimizer updates, migration
opt-in and identity guards. GPU execution still needs verification on Kaggle.
