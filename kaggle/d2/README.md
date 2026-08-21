# Section 6.2: epoch budget

One long CodeNet run that records validation loss and test accuracy after every
epoch, to show where training plateaus.

Writes `epoch_study_validation_bce.csv` and `epoch_study_test_accuracy.csv`.
Collect them under `outputs/kaggle/d2/`, then:

```bash
python research/discussions/build_d2_latex.py
```
