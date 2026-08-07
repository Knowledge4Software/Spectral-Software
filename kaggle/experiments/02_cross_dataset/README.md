# Cross-dataset transfer: one run per method

This experiment has exactly four Kaggle notebooks. In every notebook, attach
the same four final clean-data ZIPs:

```text
codexglue_v3_clean_data.zip
atcoder_v3_clean_data.zip
gptclonebench_v3_clean_data.zip
semanticclonebench_v3_clean_data.zip
```

The notebook first constructs a private, namespaced data view:

```text
train + valid = CodeXGLUE V3
test          = AtCoder V3, GPTCloneBench V3, SemanticCloneBench V3
```

It trains **once** on CodeXGLUE, chooses one threshold only on CodeXGLUE
validation, freezes it, and returns one test-result row per target dataset.
The run profile is `transfer_250k`: 250,000 source train pairs, 20,000 source
validation pairs, and up to 20,000 test pairs for each target. Therefore GPT
and SemanticCloneBench use their complete test sets, while AtCoder is capped at
20,000 pairs. A capped target test set is sampled stratified by label, so its
clone/non-clone ratio remains the official test-set ratio (AtCoder is exactly
10,000 / 10,000). This is the same protocol and budget for every method.

| Method | Notebook |
| --- | --- |
| SPECTRA-Siam | `01_spectra_siam_train_on_codexglue.ipynb` |
| ASTNN | `02_astnn_train_xglue_test_all.ipynb` |
| RtvNN | `03_rtvnn_train_xglue_test_all.ipynb` |
| DeepSim | `04_deepsim_train_xglue_test_all.ipynb` |

Use a **2×T4 GPU** and `Save Version` / `Run All`. Do not attach transfer
bundles for this version of the experiment. Each baseline writes one combined
CSV named `cross_dataset_all_<method>_all_targets_results.csv`; SPECTRA writes
its corresponding combined transfer table.

The temporary namespacing (`cx_`, `at_`, `gp_`, `sc_`) prevents overlapping raw
code ids between the benchmark exports from referring to the wrong program.
