# Research code

Organised by the section of the paper each package serves, matching
`kaggle/` (the notebooks that produce the results) and `kaggle/latex/`
(the tables and figures they build).

| Package | Serves | Reads | Writes |
| --- | --- | --- | --- |
| `rq1/` | RQ1: spectral discriminability | `outputs/kaggle/RQ1` | `kaggle/latex/rq1` |
| `rq2/` | RQ2: comparison against baselines | `outputs/kaggle/RQ2` | `kaggle/latex/rq2` |
| `rq3/` | RQ3: cross-language transfer | `outputs/kaggle/RQ3` | `kaggle/latex/rq3` |
| `discussions/` | Sections 6.1–6.5 | `outputs/kaggle/d1…d5` | `kaggle/latex/d1…d5` |
| `design/` | the design/protocol tables | `outputs/kaggle/RQ2` | `kaggle/latex/design` |

## Supporting packages

| Package | What it is |
| --- | --- |
| `faithful_graph_baselines/` | the shared baseline runtime the Kaggle baseline notebooks embed |
| `spectral_representation_baselines/` | fixed-graph spectral controls (No-Train / RF / LR / SNN) |
| `dataset_analysis/` | CodeNet hardness and pair-graph leakage analysis, with its outputs |
| `latent_graph_learning/` | the earlier benchmark report, superseded by the current paper |
| `pretrained_code_models/` | UniXcoder-family evaluation |

## Discussions

One builder per discussion section. Each reads the Kaggle archives for its
section and writes the LaTeX the paper includes:

```bash
python research/discussions/build_d1_latex.py       # 6.1 hyperparameter sensitivity
python research/discussions/build_d2_latex.py       # 6.2 epoch budget
python research/discussions/build_d3_latex.py       # 6.3 seed stability
python research/discussions/run_d4_local.py         # 6.4 latent-graph figure (local, ~20 min)
python research/discussions/build_d5_notebooks.py   # 6.5 builds the timing notebooks
```

`run_d4_local.py` is the exception: it runs a Kaggle notebook locally rather
than reading a finished archive, because the figure needs the trained model
rather than its metrics.

## Conventions

- Every builder resolves paths from `_ROOT`, never the working directory, so it
  can be run from anywhere.
- Kaggle result archives live outside the repo, under `outputs/kaggle/`.
- Dataset keys inside notebooks (`at-coder`, `codexglue`, `codenet-4l`) name
  published Kaggle datasets. They are independent of the folder names here and
  must not be renamed to match them.
