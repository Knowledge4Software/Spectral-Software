# Spectral Software

This repository extracts program graphs, derives spectral features, tunes
spectral similarity baselines, and prepares portable datasets for experiments.

## Project map

| Path | Purpose |
| --- | --- |
| `spectral_code/` | Reusable implementation: parsing, graphs, spectra, evaluation, and exports. |
| `pipelines/` | Shared implementation of graph and spectral extraction stages. Do not customize these per dataset. |
| `notebooks/datasets/` | Dataset-specific entry points, analysis notebooks, and experiment helpers. |
| `kaggle/` | Notebooks intended to be copied to Kaggle for GPU experiments. |
| `data/` | Local prepared/source data. Ignored by Git. |
| `outputs/` | Generated graphs, spectra, reports, models, and clean exports. Ignored by Git. |

## Dataset entry points

| Dataset | Start here |
| --- | --- |
| XGLUE | `notebooks/datasets/xglue/README.md` |
| BigCloneBench | `notebooks/datasets/bigclonebench/README.md` |
| Semantic Benchmark | `notebooks/datasets/semantic_benchmark/languages/<language>/run_pipeline/` |

## Output policy

The extraction pipeline keeps only durable research artifacts:

- cleaned graph shards;
- spectral feature shards;
- models, reports, and tuning results;
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
