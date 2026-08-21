# Section 6.5: timing breakdown

Three runs, one per benchmark, that rerun the canonical SPECTRA-Siam (lexical)
configuration and record how long every stage of the methodology takes.

| Notebook | Benchmark | Attach | Expected runtime |
| --- | --- | --- | --- |
| `d5_timing_codenet.ipynb` | CodeNet | `codenet_4l_clean_data.zip` | ~20 min |
| `d5_timing_atcoder.ipynb` | AtCoder | `atcoder_v3_clean_data.zip` | ~2.5 h |
| `d5_timing_xglue.ipynb` | BigCloneBench | `codexglue_v3_clean_data.zip` | ~4.5 h |

Select a T4 GPU and Run All. Run CodeNet first: it is short, and it confirms
the instrumentation works before committing to the long runs.

## What is measured

Fourteen stages, from the input inwards. `s07` is nested inside `s06_to_s08`
and is reported separately because it is the step whose cost is hardest to
predict; it is excluded from the measured total so nothing is double-counted.

| Stage | What it covers |
| --- | --- |
| `h01_data_wait` | blocked waiting on the DataLoader (host) |
| `h02_host_to_device` | copying the batch to the GPU |
| `s01_embed_input` | canonical + lexical embedding, input norm |
| `s02_relation_gnn` | the stacked relational graph layers |
| `s03_slot_attention` | soft assignment of nodes to latent slots |
| `s04_latent_adjacency` | query/key projections building the latent adjacency |
| `s05_latent_refine` | latent-graph refinement layers |
| `s06_to_s08_spectral` | the whole spectral block (Laplacian → descriptors) |
| `s07_eigendecomposition` | `eigvalsh` alone, **nested inside s06_to_s08** |
| `s09_pair_features` | symmetric comparison of the two descriptors |
| `s10_classifier` | the pair MLP head |
| `s11_loss` | the composite loss |
| `s12_backward` | backpropagation |
| `s13_optimizer_step` | grad clip, optimizer step, scaler update |

## Outputs

Written to `/kaggle/working`, one set per benchmark:

- `d5_batch_timings_<dataset>.csv` — one row per batch, one column per stage.
  This is the raw data: thousands of rows, enough for confidence intervals on
  any stage.
- `d5_epoch_timings_<dataset>.csv` — per-epoch train/validation/total seconds
  and throughput.
- `d5_stage_summary_<dataset>.csv` — per-stage mean, std, median, p95, a 95%
  confidence interval, total seconds, and share of measured time.
- `d5_inference_summary_<dataset>.csv` — the same breakdown for validation
  batches, which have no backward pass. This is what a deployment pays per
  pair, as opposed to the cost of training on one.
- `d5_timing_summary_<dataset>.json` — totals, throughput for training and
  inference separately, device name, batch size, and parameter count.

Collect them under `outputs/kaggle/d5/`.

## How it is measured

GPU work is asynchronous. A wall-clock timer around a CUDA call measures the
kernel *launch*, not the kernel, so every device stage is timed with CUDA
events recorded on the stream and read after a synchronize. Host stages (loader
wait, host-to-device copy) use `perf_counter`, because they are real CPU work.

Instrumentation wraps each module's `forward` instead of editing the model, so
the timed code is the same code the other sections ran. The model, data,
optimizer, and seed are untouched: the accuracy these runs report should match
the RQ2 run for the same benchmark. If it does not, the run is not comparable
and should be investigated rather than reported.

Before writing anything, the report calls `d5_assert_instrumented()`, which
fails the run if fewer than nine stages recorded data. An earlier version
defined the timers but never attached them to the model: every in-model
stage reported zero while the run still finished cleanly, and the result
looked plausible. The check exists so that failure is loud.

`WallMs` is the true per-batch wall time; `MeasuredMs` is the sum of the
stages, and `UnattributedMs` is the difference. A small gap is normal
(Python-level glue); a large one means a stage is missing.

## Rebuilding

```bash
python research/discussions/build_d5_notebooks.py
```

Regenerates all three from the current `kaggle/rq2/<dataset>/method/
spectra_siam_lex.ipynb`, so the timed model never drifts from the real one. The
builder asserts every anchor it patches, so if the method notebook changes
shape the build fails instead of silently producing an un-instrumented run.
