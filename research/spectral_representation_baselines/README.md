# Local spectral-representation baselines

Each dataset folder contains four launchers: No Train, Random Forest, Logistic
Regression. They use only the precomputed graph spectra in the V3 `clean_data`
export and preserve the official splits. The SNN baseline is run on Kaggle.

Run a launcher from the project root, for example:

```powershell
.venv\Scripts\python.exe research\spectral_representation_baselines\datasets\gptclonebench\01_no_train.py
```

Default runs use every graph-evaluable pair in train/valid/test. `PDG` is
written as `unavailable` instead of failing because the four V3 exports contain
AST, CFG, DDG and CPG only. Results, timing, thresholds, and metadata are in
`outputs/local_spectral_representation_baselines/<dataset>/<method>/`.

`04_run_all.py` in each dataset folder runs these three local methods sequentially.

Useful temporary cap:

```powershell
.venv\Scripts\python.exe research\spectral_representation_baselines\datasets\codexglue\02_random_forest.py --max-train-pairs 100000
```
