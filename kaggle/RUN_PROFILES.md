# Kaggle run profiles

The established notebooks expose the same profile selector near the top of the
configuration cell.

```python
RUN_PROFILE = "final_full"
```

Use `final_full` for the final in-domain benchmark. It uses every
graph-evaluable pair in the official split. Pair counts can be slightly lower
than the source release when a referenced program has no valid graph; the
notebook records those counts rather than hiding them.

Use `comparison_50k` for a controlled 50,000 train / 10,000 validation /
10,000 test comparison. SPECTRA-Siam uses six epochs, so 50,000 x 6 preserves
the number of pair presentations in the former 75,000 x 4 quick run.

Use `diagnostic_10k` only for pipeline checks and representation diagnostics.
Use `extended_6_7h` only for a separately labelled large-budget follow-up.

The forward-ready CodeNet 4L notebook defaults to `comparison_50k` because the
final graph-evaluable pair count and graph-size distribution are not yet known.
Select `final_full` only after the incoming clean-data release is audited for
memory and wall-time requirements.

The three notebooks under `experiments/06_codenet_nonclone_scopes/` use the
fixed 8,000-pair protocols described in their README. Pair caps are disabled
there, and `final_full` controls the training schedule without altering the
selected 5,600/1,200/1,200 train/validation/test counts.
