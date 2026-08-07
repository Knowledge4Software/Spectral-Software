# XGLUE

## Layout

| Path | Purpose |
| --- | --- |
| `run_pipeline/` | Local extraction and clean-data export stages. |
| `baselines/` | Separate local spectral no-train, RF, and LR baselines. |
| `analysis/` | Dataset inspection, graph analysis, and combined result table. |
| `../../../kaggle/` | GPU notebooks for GNN, SNN, and graph-based baselines. |

## Local pipeline

Run the numbered scripts in `run_pipeline/` to prepare raw code, graphs,
spectra, and the portable clean export. The final export is the dataset that
can be uploaded to Kaggle.

Re-run stage `05_export_clean_data.py` once after the current update before
using the structural ASTNN notebook. The export now includes AST node types
and node labels in addition to sparse adjacency and eigenvalues.

## Baseline ownership

- Local CPU: no-training spectral, Random Forest, and Logistic Regression.
- Kaggle GPU: GNN, SNN, RtvNN, CDLH, DeepSim, ASTNN, and FA-AST variants.

All Kaggle notebooks read the clean export rather than rerunning Joern.
