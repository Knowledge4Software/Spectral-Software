# BigCloneBench

## Layout

| Path | Purpose |
| --- | --- |
| `types/type1` through `types/type4` | Positive clone-type datasets, each with `build/` and `tuning/`. |
| `types/type3/{moderate,strong,very_strong}` | Type-3 subsets; `type3/all` is assembled from them. |
| `types/non_clone/` | Curated BCB false-positive non-clone pairs and its `build/` entry points. |
| `run_pipeline/` | Cross-type experiments and portable clean-data export tools. |
| `analysis/` | Result and graph inspection notebooks. |
| `build_once.py` | Preferred one-time data/graph/spectral build entry point. |
| `tune.py` | Preferred repeatable spectral-threshold tuning entry point. |

Every `build/run_all.py` runs data preparation, graph extraction, and spectral
extraction together. Tuning is deliberately outside `build/`, so it is never
run by accident. The scripts share the implementation in
`spectral_code/evaluation/run_pipeline_helpers.py`, so extraction logic is not
duplicated per clone-type folder.

## Type-4 plus non-clone portable dataset

Use Type-4 pairs as positive examples and curated BCB non-clone pairs as
negative examples. Build durable artifacts once; `clean_graphs` and
`spectral_features` remain in `outputs/bcb/type4` and
`outputs/bcb/non_clone` for later tuning and re-export.

For a first build, run from the repository root:

```powershell
python notebooks\datasets\bigclonebench\build_once.py --variant type4 non_clone --start-at data
```

If prepared BCB data already exists and only graph/spectral artifacts need to
be rebuilt, use:

```powershell
python notebooks\datasets\bigclonebench\build_once.py --variant type4 non_clone --start-at graphs
```

Then produce the portable Kaggle dataset:

```powershell
python notebooks\datasets\bigclonebench\run_pipeline\04_create_balanced_subset.py
```

The export is written to:

```text
outputs/bcb/BCB4DATA/clean_data/
```

It contains the same clean-data schema as XGLUE: `codes.jsonl.gz`,
`pairs.csv.gz`, `graph_spectra.jsonl.gz`, `metadata.json`, and `README.md`.
`outputs/bcb/BCB4DATA/BCB4DATA.zip` is the Kaggle-uploadable archive. Its
train/valid/test sizes and clone/non-clone proportions match the XGLUE split
profile: train is nearly balanced, while valid/test retain XGLUE's non-clone
imbalance. Re-running the export replaces only the prior
`outputs/bcb/BCB4DATA/` export; it does not recompute graphs or eigenvalues.

The extraction stage removes only temporary Joern directories and diagnostics.
It preserves graph shards, spectra, reports, tuning outputs, and any trained
models. `--cleanup-raw-intermediates` on the export command follows the same
safe rule.

## Spectral threshold tuning

`tune.py` is only for the local no-training PSS spectral baseline. It is not
needed to prepare the Kaggle dataset or train GNN/SNN/RF/LR baselines. Run it
after the relevant variant and non-clone spectra exist only when that baseline
result is needed.

```powershell
python notebooks\datasets\bigclonebench\tune.py --variant type4 --no-pair-score-csv
```

## Split caveat

The portable subset currently prevents duplicate pair rows but allows a code
identifier to appear in more than one split. It is suitable for the current
pair-level BCB protocol. A code-disjoint exporter is required before making a
directly comparable claim against XGLUE's code-disjoint split.

## Leakage-resistant Type-4 export

`BCB4DATA` is retained for pair-level experiments only. Do not compare its
results directly with XGLUE: its IDs can cross splits and its positive and
negative sources are different.

For a code-disjoint, source-balanced evaluation that reuses the existing
portable graph records, run:

```powershell
python notebooks\datasets\bigclonebench\run_pipeline\05_create_code_disjoint_subset.py
```

This writes `outputs/bcb/BCB4STRICT/`. It intentionally contains far fewer
pairs, because it retains only code IDs occurring in both labelled sources and
discards pairs that would cross a code split.
