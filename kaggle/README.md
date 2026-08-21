# Kaggle notebooks

Organised by the section of the paper each study feeds, mirroring
`kaggle/latex/`, so every notebook folder has a matching output folder.

| Folder | Paper section | Builds | Notebooks |
| --- | --- | --- | --- |
| `rq1/` | RQ1: spectral discriminability | `latex/rq1/` | 2 |
| `rq2/` | RQ2: comparison against baselines | `latex/rq2/` | 61 |
| `rq3/` | RQ3: cross-language transfer | `latex/rq3/` | 96 |
| `d1/` | 6.1 hyperparameter sensitivity | `latex/d1/` | 4 |
| `d2/` | 6.2 epoch budget | `latex/d2/` | 1 |
| `d3/` | 6.3 seed stability | `latex/d3/` | 11 |
| `d4/` | 6.4 latent-graph interpretation | `latex/d4/` | 1 |
| `exploratory/` | not in the paper | — | 33 |

`exploratory/` holds earlier studies (latent capacity, cross-dataset transfer,
feature ablations, non-clone scopes). They are kept for reference and feed the
older report under `research/latent_graph_learning/reports/`, not the current
paper.

## RQ2 layout

One folder per benchmark, each with the same nine baselines and the same three
input-only SPECTRA-Siam variants, so the only difference between them is the
attached dataset:

| Folder | Clean-data attachment |
| --- | --- |
| `rq2/atcoder/` | `atcoder_v3_clean_data.zip` |
| `rq2/xglue/` | `codexglue_v3_clean_data.zip` |
| `rq2/codenet/` | `codenet_4l_clean_data.zip` |
| `rq2/gptclonebench/` | `gptclonebench_v3_clean_data.zip` |
| `rq2/semanticclonebench/` | `semanticclonebench_v3_clean_data.zip` |

The attachment names keep their original `_v3` / `_4l` spelling: they identify
published Kaggle datasets and the `DATASET_KEY` inside each notebook, which are
independent of the folder names here.

The three method variants are cumulative in their encoder inputs:

- `method/spectra_siam_topo.ipynb` — topology only;
- `method/spectra_siam_label.ipynb` — topology plus canonical node labels;
- `method/spectra_siam_lex.ipynb` — plus lexical sketches (the primary method).

All three share one learned latent graph and one descriptor-only spectral pair
head, so any difference between them is attributable to the encoder input
alone.

## Running

Attach the benchmark's clean-data ZIP, select a T4 GPU, Run All. `RUN_ORDER.md`
gives the order the paper's results were produced in, and `RUN_PROFILES.md`
documents the pair budgets (`final_full`, and `session_50k` for the three slow
baselines on the large datasets).
