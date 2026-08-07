# Final full-data baseline run order

Every baseline notebook now defaults to `RUN_PROFILE = "final_full"`. Attach
the matching dataset, import the notebook, and choose **Run All**. The output
CSV includes F1 metrics, `TrainableParameters`, `RuntimeSeconds`, and
`RuntimeMinutes`.

Each completed run also writes `<dataset>_<run-label>_run_metadata.json` to
`/kaggle/working/`. It records the dataset key, profile, seed, requested pair
caps, configured epochs/batch size/optimizer values, GPU model/capability,
PyTorch version, completion time, measured runtime, and output file paths.

Run the following order for each dataset folder (`atcoder_v3`,
`codexglue_v3`, `gptclonebench_v3`, then `semanticclonebench_v3`). It provides
early sanity checks before the expensive graph models.

1. `baselines/deckard_baseline.ipynb` — non-neural structural reference.
2. `baselines/cdlh_baseline.ipynb` — strong lexical baseline.
3. `baselines/rtvnn_baseline.ipynb` — sequence neural baseline.
4. `baselines/astnn_baseline.ipynb` — AST tree baseline.
5. `baselines/snn_baselines.ipynb` — spectral baseline (AST/CFG/DDG/CPG).
6. `baselines/deepsim_baseline.ipynb` — graph similarity baseline.
7. `baselines/fa_ast_ggnn_baseline.ipynb` — gated graph baseline.
8. `baselines/fa_ast_gmn_baseline.ipynb` — graph matching baseline.
9. `baselines/gnn_baselines.ipynb` — final multi-graph GNN baseline.

Use a T4-class Kaggle accelerator. Each notebook measures end-to-end runtime
(data loading, preprocessing, training, validation, and final test), so do
not compare its time with a notebook run under a different accelerator or
with a different `RUN_PROFILE`.

After the baseline batch, run `method/spectra_siam.ipynb` in the same dataset
folder. It now defaults to `final_full` and writes
`<dataset>_spectra_siam_results.csv`, which has the same core metric, pair
count, parameter, runtime, seed, and environment columns as the baseline
result files. Use `comparison_50k` or `diagnostic_10k` only for separate
SPECTRA-Siam parameter and ablation experiments.
