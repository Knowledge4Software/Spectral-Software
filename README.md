# Spectral Software

SPECTRA-Siam: clone detection by inducing a latent graph over a program's
syntax and comparing multi-scale spectral descriptors of that graph.

This repository holds the implementation, the notebooks that produce every
number in the paper, and the builders that turn those results into the paper's
tables and figures.

## Layout

The paper's sections drive the layout. For any table or figure, the notebook
that produced it, the code that reads the result, and the LaTeX it becomes all
sit at the same name:

```
kaggle/<section>/          the notebooks, run on Kaggle
research/<section>/        the code that reads their results
kaggle/latex/<section>/    the tables and figures the paper includes
```

| Section | What it answers |
| --- | --- |
| `rq1` | Is the learned spectral representation discriminative on its own? |
| `rq2` | How does the method compare against nine baselines? |
| `rq3` | Does bridge-assisted transfer help across languages? (156 configurations) |
| `d1`–`d5` | Discussions: hyperparameter sensitivity, epoch budget, seed stability, what the latent graph captures, and where the time goes. |

| Path | Purpose |
| --- | --- |
| `kaggle/` | Notebooks run on Kaggle, one folder per paper section, plus `exploratory/` for studies not in the paper. |
| `kaggle/latex/` | The tables and figures the paper includes. |
| `research/` | Builders that turn Kaggle result archives into `kaggle/latex/`, plus the shared baseline runtime. |
| `spectral_code/` | Reusable implementation: parsing, graphs, spectra, evaluation, and exports. |
| `pipelines/` | Shared graph and spectral extraction stages. Not customised per dataset. |
| `notebooks/datasets/` | Dataset preparation entry points and analysis notebooks. |
| `scripts/` | Dataset build, publication, notebook synchronisation, and storage maintenance. |
| `tests/` | Regression tests, including the notebook and method contracts. |
| `paper/` | Manuscript drafts, bibliography, and figures. Compiled PDFs are not tracked. |

See `kaggle/README.md` and `research/README.md` for the per-section detail.

## Data and results live outside the repository

Only code and the paper's LaTeX are tracked here. The datasets and the Kaggle
result archives sit beside the repository, because they are large and
regenerable:

```
spectrals/
  Spectral-Software/     this repository
  data/                  benchmarks/ (what a notebook attaches)
                         sources/    (what those were built from)
  outputs/               kaggle/     (result archives, one folder per section)
```

`data/README.md` and `outputs/README.md` describe both. The in-repo `data/` and
`bench_data/` folders hold small local inputs only and are ignored by Git.

## Reproducing a result

1. Attach the benchmark's export from `data/benchmarks/` to the notebook in
   `kaggle/<section>/`, select a T4 GPU, and Run All.
2. Download the result archive into `outputs/kaggle/<section>/`.
3. Run the matching builder, for example:

```bash
python research/rq1/04_run_all.py                    # the RQ1 table
python research/rq2/build_latex.py                   # the RQ2 comparison
python research/discussions/build_d3_latex.py        # Section 6.3
```

Section 6.4 is the exception: it needs the trained model rather than its
metrics, so `research/discussions/run_d4_local.py` runs it locally on CPU in
about twenty minutes.

## Dataset entry points

| Dataset | Start here |
| --- | --- |
| XGLUE | `notebooks/datasets/xglue/README.md` |
| BigCloneBench | `notebooks/datasets/bigclonebench/README.md` |
| Semantic Benchmark | `notebooks/datasets/semantic_benchmark/run_pipeline/` |
| CodeNet 4L | `notebooks/datasets/codenet_4l/` |

## Conventions

- Run scripts from the repository root; every builder resolves its own paths,
  so the working directory does not matter.
- Use the numbered scripts in `run_pipeline/` in ascending order.
- Do not run `pipelines/01_*`, `02_*`, or `03_*` directly unless developing the
  shared pipeline; the dataset entry points set their required environment.
- The `DATASET_KEY` inside a notebook (`at-coder`, `codexglue`, `codenet-4l`)
  names the dataset as published on Kaggle. It is independent of the folder and
  file names here, and renaming one does not rename the other.
- Extraction keeps only durable artifacts. Joern scratch (`dataset_features`,
  `java_files`, `cpg`, `dot`, batch files, raw exports) is removed after
  spectral extraction; `scripts/reorganize_project_storage.py` clears what
  survives a crash.

## Tests

```bash
python -m pytest tests -q
```

Beyond the usual unit tests, these enforce contracts that a silent change would
otherwise break: every baseline trains on the same split and reports the same
metrics, every method notebook matches the reference model, and no RQ3
configuration trains on its own evaluation target.
