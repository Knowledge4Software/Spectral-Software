# Kaggle run profiles

Every notebook in the four dataset folders now has the same selector at the
top of its configuration cell:

```python
RUN_PROFILE = "quick_1h"
```

Use `quick_1h` for the initial comparable results.  It is deliberately
conservative: all runs are capped below roughly an hour on a Kaggle T4/P100,
with the expensive pair-graph baselines receiving smaller caps.

For a directly comparable method table, set every notebook to:

```python
RUN_PROFILE = "comparison_50k"
```

This profile uses the same **50k train / 10k validation / 10k test** cap in
every method wherever the dataset contains that many pairs.  Each baseline
keeps its own justified epoch and early-stopping schedule; its output CSV now
records `TrainableParameters`, `RuntimeSeconds`, and `RuntimeMinutes`, so the
paper can report both capacity and measured end-to-end runtime rather than
claiming that different architectures have exactly equal parameter counts.
Deckard is explicitly recorded as zero-parameter.  SPECTRA-Siam writes the
same fields in its final JSON output.

SPECTRA-Siam uses six epochs in this profile: `50k × 6` preserves the same
number of pair presentations/optimizer updates as its former `75k × 4`
quick run.  This avoids unintentionally under-training SPECTRA while making
the sampled data cap comparable to the baselines.

## Final baseline protocol

All baseline notebooks now default to:

```python
RUN_PROFILE = "final_full"
```

`final_full` sets `max_train_pairs`, `max_valid_pairs`, and `max_test_pairs`
to `None`. Thus it uses every graph-evaluable pair in the official split; a
small difference from the original pair count is reported in the final CSV
when a code record has no valid graph. SPECTRA-Siam now also defaults to this
profile and writes a flat `*_spectra_siam_results.csv` with the same core
columns as baseline result files. Its parameter/ablation experiments can be
run by manually selecting `comparison_50k` or `diagnostic_10k`.

For the follow-up run, change **only** that value to:

```python
RUN_PROFILE = "extended_6_7h"
```

`extended_6_7h` restores the 100k/20k/20k baseline protocol.  For
SPECTRA-Siam it uses 650k/80k/80k and eight epochs, targeting the 6--7 hour
budget on the large CodeXGLUE and AtCoder datasets.  GPTCloneBench and
SemanticCloneBench contain fewer pairs, so they finish sooner even with this
profile; they are run on all available pairs rather than padded or duplicated.

The wall-time labels are upper-bound targets rather than guarantees: Kaggle
GPU availability and the number/size of graphs selected by a random subset
both affect the final runtime.
