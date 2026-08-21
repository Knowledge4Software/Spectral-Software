# Spectral Software

This repository extracts program graphs, derives spectral features, tunes
spectral similarity baselines, and prepares portable datasets for experiments.

## Project map

| Path | Purpose |
| --- | --- |
| `kaggle/` | Notebooks run on Kaggle, organised by paper section (`rq1`–`rq3`, `d1`–`d5`). |
| `kaggle/latex/` | The tables and figures the paper includes, one folder per section. |
| `research/` | Builders that turn Kaggle result archives into `kaggle/latex/`, plus the shared baseline runtime. |
| `spectral_code/` | Reusable implementation: parsing, graphs, spectra, evaluation, and exports. |
| `pipelines/` | Shared graph and spectral extraction stages. Not customised per dataset. |
| `notebooks/datasets/` | Dataset preparation entry points and analysis notebooks. |
| `scripts/` | Dataset build, publication, and notebook-synchronisation commands. |
| `tests/` | Regression tests for preprocessing, graph support, and the notebook contracts. |
| `paper/` | Manuscript drafts, bibliography, figures, and compiled PDFs. |
| `data/`, `bench_data/`, `output/` | Local data and artifacts. Ignored by Git. |

The paper's own sections drive the layout: `kaggle/<section>/` holds the
notebooks, `research/` holds the code that reads their results, and
`kaggle/latex/<section>/` holds what the paper includes. See `kaggle/README.md`
and `research/README.md` for the per-section detail.

Kaggle result archives are written outside the repository, under
`outputs/kaggle/`.

## Dataset entry points

| Dataset | Start here |
| --- | --- |
| XGLUE | `notebooks/datasets/xglue/README.md` |
| BigCloneBench | `notebooks/datasets/bigclonebench/README.md` |
| Semantic Benchmark | `notebooks/datasets/semantic_benchmark/run_pipeline/` |
| Kaggle V3 benchmarks | `kaggle/README.md` |

## Output policy

The extraction pipeline keeps only durable research artifacts:

- cleaned graph shards;
- spectral feature shards;
- models, reports, publication PDFs, and tuning results;
- small diagnostics such as manifests and timing metadata.

Temporary Joern artifacts (`dataset_features`, `java_files`, `cpg`, `dot`,
batch files, and raw Joern exports) are removed automatically after spectral
extraction.

## Conventions

- Run dataset scripts from the repository root.
- Use the numbered scripts in `run_pipeline/` in ascending order.
- Do not run `pipelines/01_*`, `02_*`, or `03_*` directly unless developing
  the shared pipeline; dataset entry points set their required environment.
- Treat `outputs/` as reproducible artifacts and `data/` as local inputs.
