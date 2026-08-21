# RQ1 — Discriminative power of the learned latent graph

**Research question.** To what extent does the learned graph latent space improve the discriminative power of spectral representations for source code compared with conventional program graphs?

## Controlled comparison

RQ1 isolates the graph representation from the SPECTRA-Siam pair classifier. The current descriptor-only SPECTRA-Siam is trained on the official training pairs. Its best checkpoint is selected using validation data and then frozen. The frozen encoder exports one soft 32-node latent adjacency and its normalized-Laplacian eigenvalues for every code endpoint appearing in the complete benchmark pair universe.

The local evaluation applies the repository's same Program Spectral Similarity (PSS) implementation to five graph views:

- conventional AST;
- conventional CFG;
- conventional DDG;
- conventional CPG;
- the learned latent graph.

No SPECTRA-Siam classifier output or code embedding is used by this comparison. All graph views are evaluated on the same common-support pairs. Both training-checkpoint selection and PSS threshold calibration follow the dataset balance: an exactly 50/50 validation split uses validation accuracy, while an imbalanced validation split uses validation F1. The selected threshold is then frozen for the official test split. ROC-AUC and average precision remain threshold-free primary measures.

This design measures the discriminative information in the learned graph spectrum. Because the latent graph is supervised by training-pair labels while conventional graphs are fixed, the result should be described as the benefit of a **task-learned graph representation**, not as an unsupervised graph-quality comparison.

## Kaggle notebooks

1. `notebooks/01_atcoder_export_latent_graphs.ipynb`
2. `notebooks/02_xglue_export_latent_graphs.ipynb`

Attach the corresponding clean-data ZIP, enable a T4-class GPU, and run every cell. Each notebook trains the latest descriptor-only SPECTRA-Siam from scratch and creates one downloadable file in `/kaggle/working`:

- `rq1_atcoder_v3_latent_graphs.zip`
- `rq1_codexglue_v3_latent_graphs.zip`

Each archive contains the exact pair table, a manifest, the selected checkpoint, and compressed latent-graph shards. Adjacencies are stored as `float16` for transfer size; normalized-Laplacian eigenvalues are stored independently as `float32` and are the values consumed by PSS.

With the current clean datasets, ATCoder uses validation Accuracy, while CodeXGLUE/XGLUE uses validation F1. This is inferred from the actual validation label counts at runtime rather than hard-coded by dataset name.

## Local table generation

With the standard repository/output layout used by this project, run both
datasets (resume is enabled by default) with one command from the repository
root:

```powershell
.\.venv\Scripts\python.exe research\rq1\04_run_all.py
```

This launcher reads the complete Kaggle output ZIPs directly from
`outputs/kaggle/RQ1`, including their nested latent-graph folders. To run a
single dataset instead, pass it to `run_table.py` (see below) or call
`run_default_dataset("atcoder")` / `run_default_dataset("xglue")`.

Run the complete table generator from the repository root after downloading both Kaggle archives:

```powershell
python research/rq1/run_table.py `
  --atcoder-clean-data "C:\PyProjects\spectrals\outputs\atcoder_v3\clean_data" `
  --atcoder-latent-export "C:\path\to\rq1_atcoder_v3_latent_graphs.zip" `
  --xglue-clean-data "C:\PyProjects\spectrals\outputs\codexglue_v3\clean_data" `
  --xglue-latent-export "C:\path\to\rq1_codexglue_v3_latent_graphs.zip" `
  --output-dir "C:\PyProjects\spectrals\outputs\rq1\table" `
  --device auto
```

The script writes all rows required by the paper table:

- `SPECTRA-Siam` from the frozen Kaggle checkpoint;
- `SPECTRA-Siam Spectrum + No Train/RF/LR/SNN`;
- `AST/CFG/DDG/CPG + No Train/RF/LR/SNN`;
- `Predict All Clone`.

`No Train` is PSS; RF/LR/SNN consume the identical 128-eigenvalue-plus-statistics spectral representation. Every learned classifier trains only on the official train split. Validation selects both threshold and checkpoint with Accuracy for an exactly balanced validation split and F1 otherwise. The script writes `rq1_all_table_rows.csv`, `rq1_paper_table_values.csv`, `rq1_paper_table_values.tex`, and `rq1_paper_table.png`.

`evaluate_pss.py` remains available only for a focused PSS diagnostic; it is not the generator for the paper table.

## Interpretation rule

RQ1 is answered by the full table, not by a single ROC-AUC delta. Compare the learned latent spectrum rows with the matching AST/CFG/DDG/CPG rows under the same downstream choice (`No Train`, `RF`, `LR`, or `SNN`). Training and test pairs must never be merged for model selection or threshold selection.
