# XGLUE Spectral Baselines

This folder runs classical spectral baselines from the clean XGLUE export produced by:

```bash
python notebooks/datasets/xglue/run_pipeline/run_all.py
```

The expected default input is:

```text
outputs/xglue/clean_data/
```

Baselines are intentionally split into separate runners and output folders:

- `run_spectral.py`: spectral/no-train distance score only; threshold is selected on `valid`, then reported on `test`.
- `run_lr.py`: logistic regression over pairwise spectral features.
- `run_rf.py`: random forest over the same pairwise spectral features.

All methods use the same split protocol:

```text
train -> fit LR/RF
valid -> choose threshold
test  -> final P/R/F1/Acc
```

Run each baseline separately:

```bash
python notebooks/datasets/xglue/baselines/run_spectral.py
python notebooks/datasets/xglue/baselines/run_lr.py
python notebooks/datasets/xglue/baselines/run_rf.py
```

Outputs:

```text
outputs/xglue/baselines/spectral/xglue_spectral_results.csv
outputs/xglue/baselines/lr/xglue_lr_results.csv
outputs/xglue/baselines/rf/xglue_rf_results.csv
```

For a Kaggle input directory, pass `--clean-data-dir` to the runner you want:

```bash
python notebooks/datasets/xglue/baselines/run_spectral.py --clean-data-dir /kaggle/input/datasets/koushamoeini/xgluedata
```
