# Section 6.5: timing

Built by `research/discussions/build_d5_latex.py` from the three d5 runs in
`outputs/kaggle/d5/`.

| File | What it is |
| --- | --- |
| `table_d5_stages.tex` | per-stage training cost, one column per benchmark |
| `table_d5_epochs.tex` | per-epoch wall time and throughput |
| `figure_d5.tex` | include block for the figure |
| `figures/d5_timing_breakdown.pdf` | training and inference profiles side by side |
| `section_6_5_numbers.md` | the values the prose needs |

## Reading the numbers

**Two totals that differ, on purpose.** The stage table sums to slightly more
than the wall-clock time of a batch (88.3 ms against 82.0 ms on
BigCloneBench). Device stages are timed with CUDA events, which measure GPU
occupancy, and GPU work overlaps with the host: the backward pass is still
running while Python has moved on. The stage table therefore answers "what does
each stage cost", and the epoch table answers "how long does this take".

**Eigendecomposition is nested.** It is measured inside the spectral block, so
it appears indented and is never added into a column total. The share columns
sum to exactly 100% without it.

**The optimizer step fires every fourth batch**, because gradient accumulation
is four. Its mean is therefore an average over all batches, three quarters of
which contribute zero.

## What the runs show

The spectral machinery, which is the method's contribution, costs **5.1–5.2%**
of training time and **7.1–7.4%** at inference, on all three benchmarks. The
eigendecomposition inside it is **1.1%**. The dominant training cost is the
backward pass (~47%), and input handling — waiting on the loader plus the
host-to-device copy — is ~32% of training and ~60% of inference.

That last figure is worth stating plainly in the paper rather than leaving
implicit: these runs are input-bound at inference, so the measured throughput
is a property of the data pipeline as much as of the model.

Each run reproduces the accuracy of the corresponding RQ2 run exactly, which is
what makes these timings describe the models the paper reports rather than a
separate experiment.

## Rebuilding

```bash
python research/discussions/build_d5_latex.py
```

It picks the newest archive per benchmark and skips any that recorded fewer
than nine stages, so an early run whose timers were never attached cannot be
averaged in by accident.
