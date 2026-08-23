
# === Section 6.5: write the timing tables ===
import json as _d5_json
from pathlib import Path as _d5_Path

# Refuse to write a report from a run whose timers never attached.
d5_assert_instrumented()

# run_experiment() returns the run summary; the notebook binds it under
# one of these names depending on the cell that called it.
RESULT = {}
for _cand in ("result", "RESULT", "experiment_result", "_result"):
    if isinstance(globals().get(_cand), dict):
        RESULT = globals()[_cand]
        break

_d5_out = _d5_Path("/kaggle/working")
_batch = _d5_pd.DataFrame(D5_BATCH_ROWS)
_epoch = _d5_pd.DataFrame(D5_EPOCH_ROWS)
_suffix = D5_DATASET.lower().replace(" ", "_")

_batch.to_csv(_d5_out / ("d5_batch_timings_" + _suffix + ".csv"), index=False)
_epoch.to_csv(_d5_out / ("d5_epoch_timings_" + _suffix + ".csv"), index=False)

# Per-stage summary over training batches, with a 95% CI for the mean. There
# are thousands of batches here, so the normal interval applies (unlike the
# five-seed case in Section 6.3, which needs Student-t).
_train = _batch[_batch.Phase.eq("train")]
_rows = []
for _name in D5_STAGES:
    if _name not in _train.columns:
        continue
    _v = _train[_name].to_numpy(dtype=float)
    if _v.sum() == 0:
        continue
    _mean, _sd, _n = _v.mean(), _v.std(ddof=1), len(_v)
    _half = 1.96 * _sd / _d5_np.sqrt(_n)
    _rows.append({
        "Stage": _name,
        "Nested": _name in D5_NESTED,
        "Batches": _n,
        "MeanMs": _mean,
        "StdMs": _sd,
        "CI95Low": _mean - _half,
        "CI95High": _mean + _half,
        "MedianMs": float(_d5_np.median(_v)),
        "P95Ms": float(_d5_np.percentile(_v, 95)),
        "TotalSeconds": _v.sum() / 1000.0,
        "ShareOfMeasured": _v.sum() / max(float(_train.MeasuredMs.sum()), 1e-9),
    })
_summary = _d5_pd.DataFrame(_rows).sort_values("TotalSeconds", ascending=False)
_summary.to_csv(_d5_out / ("d5_stage_summary_" + _suffix + ".csv"), index=False)

# The same breakdown for inference, which has no backward pass and
# is what a deployment would actually pay per pair.
_valid = _batch[_batch.Phase.eq("valid")]
_vrows = []
for _name in D5_STAGES:
    if _name not in _valid.columns or len(_valid) == 0:
        continue
    _v = _valid[_name].to_numpy(dtype=float)
    if _v.sum() == 0:
        continue
    _mean, _sd, _n = _v.mean(), _v.std(ddof=1), len(_v)
    _half = 1.96 * _sd / _d5_np.sqrt(_n)
    _vrows.append({
        "Stage": _name, "Nested": _name in D5_NESTED, "Batches": _n,
        "MeanMs": _mean, "StdMs": _sd,
        "CI95Low": _mean - _half, "CI95High": _mean + _half,
        "MedianMs": float(_d5_np.median(_v)),
        "P95Ms": float(_d5_np.percentile(_v, 95)),
        "TotalSeconds": _v.sum() / 1000.0,
        "ShareOfMeasured": _v.sum() / max(float(_valid.MeasuredMs.sum()), 1e-9),
    })
if _vrows:
    _d5_pd.DataFrame(_vrows).sort_values("TotalSeconds", ascending=False).to_csv(
        _d5_out / ("d5_inference_summary_" + _suffix + ".csv"), index=False)

_totals = {
    "Dataset": D5_DATASET,
    "Epochs": int(_batch.Epoch.max()) if len(_batch) else 0,
    "TrainBatches": int(len(_train)),
    "TrainPairs": int(_train.Pairs.sum()),
    "WallSecondsTrain": float(_train.WallMs.sum() / 1000.0),
    "MeasuredSecondsTrain": float(_train.MeasuredMs.sum() / 1000.0),
    "UnattributedSecondsTrain": float(_train.UnattributedMs.sum() / 1000.0),
    "MeanBatchMs": float(_train.WallMs.mean()) if len(_train) else 0.0,
    "MedianBatchMs": float(_train.WallMs.median()) if len(_train) else 0.0,
    "PairsPerSecondTrain": float(
        _train.Pairs.sum() / max(float(_train.WallMs.sum()) / 1000.0, 1e-9)),
    "ValidBatches": int(len(_valid)),
    "ValidPairs": int(_valid.Pairs.sum()) if len(_valid) else 0,
    "WallSecondsValid": float(_valid.WallMs.sum() / 1000.0) if len(_valid) else 0.0,
    "MsPerPairInference": float(
        _valid.WallMs.sum() / max(float(_valid.Pairs.sum()), 1e-9)) if len(_valid) else 0.0,
    "PairsPerSecondInference": float(
        _valid.Pairs.sum() / max(float(_valid.WallMs.sum()) / 1000.0, 1e-9)) if len(_valid) else 0.0,
    "TotalEpochSeconds": float(_epoch.EpochSeconds.sum()) if len(_epoch) else 0.0,
    "CudaEvents": bool(D5_CUDA),
    "Device": _d5_torch.cuda.get_device_name(0) if D5_CUDA else "cpu",
    "BatchSize": int(globals().get("BATCH_SIZE", 0)),
    # `model` is local to run_experiment(), so read the count the run
    # already recorded rather than reaching for the model object here.
    "TrainableParameters": int(
        globals().get("trainable_parameters")
        or RESULT.get("run_metadata", {}).get("TrainableParameters", 0)),
}
(_d5_out / ("d5_timing_summary_" + _suffix + ".json")).write_text(
    _d5_json.dumps(_totals, indent=2), encoding="utf-8")

print(_d5_json.dumps(_totals, indent=2))
print()
print("per-stage breakdown over " + str(len(_train)) + " training batches:")
display(_summary)
print()
print("per-epoch totals:")
display(_epoch)
