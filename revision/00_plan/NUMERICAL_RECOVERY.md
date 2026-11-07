# Recover the T4 mixed-precision failure

Execution update, 2026-11-07: the user restarted with the fixed notebook and
reported completion at epoch 43 on T4, including successful overflow recovery.
That run needs no numerical upgrade or restart. The instructions below apply
only when continuing the earlier pre-fix checkpoint.

The observed run completed epoch 12 and then failed at the finite-gradient
guard. This was not the 12-hour session guard. Finite loss does not imply finite
FP16 backward gradients, particularly after dynamic loss-scale growth.
The old trainer raised before the scaler could recover from an overflow.

## What changed

- Up to three AMP attempts of the same effective batch, with lower loss scales.
- If these fail, one FP32 attempt of the same effective batch.
- RNG and model buffers are restored before replay, preserving dropout inputs
  and running statistics. No failed attempt updates the optimizer or consumes
  a training cursor/step. Training pairs are not silently skipped.
- Non-finite clipping raises before altering the gradients. Invalid gradients
  are not replaced with zeros or masked with nan_to_num.
- Successful recoveries are logged in status/checkpoint/final-result metadata.
- A persistent FP32 failure saves a safe pre-group checkpoint and
  `numerical_failure.json`, then stops incomplete. It is not a completed score.

Architecture, losses, LR, physical/effective batch, data and seed are unchanged.
The numerical policy and implementation hash are new. Ordinary source/data/
configuration mismatch checks remain strict.

## Continue the saved epoch-12 run

1. Keep the failed run's whole output. If it contains
   `revision_full_seed42/progress.pt`, attach that output as a Kaggle input,
   together with the exact same CodeNet dataset. If no checkpoint was retained
   or is accessible, the log alone cannot reconstruct weights; start a fresh
   run using the fixed notebook.
2. Import the updated [codenet_full_seed42.ipynb](../01_baselines_seed42/codenet/codenet_full_seed42.ipynb).
   Use a fresh kernel/session, not a running
   kernel with the old embedded modules already imported. Enable the T4 GPU.
3. In the first configuration cell set:

```python
RESUME_CHECKPOINT = '/kaggle/input/YOUR-FAILED-RUN-OUTPUT/revision_full_seed42/progress.pt'
ALLOW_NUMERICAL_UPGRADE = True
FINAL_TEST = False
```

4. Keep `ARM='full'`, `SEED=42`, `MAX_EPOCHS=100`, `MIN_EPOCHS=10`, `PATIENCE=10`
   and `SESSION_HOURS=10.0`. Save & Run All.
5. Expect `approved_numerical_upgrade` in the log. The exact checkpoint cursor,
   optimizer, scaler, best validation and RNG are retained. The last saved file
   may be at the beginning of epoch 13 or an earlier periodic boundary; the
   notebook resumes that file's cursor, not the last printed line by assumption.
6. Recovery messages state the failed scale, replay action and affected parameter
   names. If FP32 is needed, it is logged. The latest status records all events.

The explicit upgrade accepts only the known prior implementation
`6da6f8fffdc37744ce177a3f573260a6e52f5f8909d23124bcb5aae61e6e4f7a`
and the frozen numerical-only target. It verifies the original identity against
the exact data/model/loss/configuration/arm and rejects non-finite checkpoint
weights or optimizer state. Changing seed, LR, architecture or support is not
an approved upgrade. Provenance is preserved in `lineage`, including the old
identity/hash and upgrade cursor. Later continuation of an upgraded checkpoint
can use `ALLOW_NUMERICAL_UPGRADE=False` with this same fixed release.

## Expected duration and what to report

The supplied log measured approximately 246 seconds per completed epoch. At
that speed the remaining 88 epochs to the ceiling are approximately six hours,
plus setup and any numerical replay overhead. Validation stopping may finish
earlier; this estimate is not a runtime guarantee.

Retain `progress.pt` and `status.json`. If the FP32 attempt still fails, send
`numerical_failure.json` and the traceback. Do not manually edit a checkpoint,
disable the failure guard, reduce the scientific schedule to four epochs, or
report the partial validation score as a final test result.

The user has reported successful native T4/CUDA execution on a fresh run. Local
tests cover deterministic overflow injection, FP32 fallback, cursor/RNG/buffer
rollback, approved migration and checkpoint safety; they do not replace that
hardware check.
