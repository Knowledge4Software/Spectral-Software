# Experiments

Six experiment families that go beyond the single-benchmark table. Each folder
holds one self-contained Kaggle notebook per configuration; attach the clean-data
ZIP(s) it names and run all cells.

Every notebook writes a flat `*_results.csv` with the same columns as the main
baseline runs (`P`, `R`, `F1`, `Acc`, `MacroF1`, `BalancedAccuracy`, `ROC_AUC`,
`TrainableParameters`, `RuntimeSeconds`, and `RunProfile`) plus the columns
specific to its experiment, so the results can be concatenated into one table
without post-processing.

| Folder | Question it answers | Attach | Status |
| --- | --- | --- | --- |
| `01_latent_capacity/` | How much does the latent graph size matter? | one benchmark ZIP | ready |
| `02_cross_dataset/` | Does a model trained on CodeXGLUE transfer to the other benchmarks? | all four ZIPs | ready |
| `03_cross_language/` | Does a model trained on one language transfer to another? | one multi-language ZIP | ready |
| `04_feature_ablation/` | Proposed attributed graph-signal spectral readout vs topology, canonical, and lexical controls | one benchmark ZIP | ready |
| `06_codenet_nonclone_scopes/` | How does performance change for A/WA, different-problem, and mixed negatives? | `codenet_4l_clean_data.zip` | ready |
| `rq3/` | Do same-language endpoints and cross-language bridge paths generalize on CodeNet? | final `codenet_4l` clean-data ZIP | ready |

## `rq3` — bridge-assisted CodeNet generalization

`rq3/01_spectra_siam.ipynb` is the full current method: topology, canonical
labels, and lexical node sketches are encoder inputs only; its classifier sees
only separate density, heat-trace, and Chebyshev spectral comparisons.
`02_astnn.ipynb`, `03_deepsim.ipynb`, and `04_rtvnn.ipynb` use the same
leakage-free scopes for the controlled non-pretrained baselines.

The default `RQ3_SUITE = "controls"` creates the four same-language reference
rows and the six endpoint-only rows of Table 7. The `bridge_l1`, `bridge_l2`,
and `bridge_l3` suites enumerate 12, 48, and 96 bridge configurations. Set
`RQ3_SHARD_COUNT`/`RQ3_SHARD_INDEX` to run those long suites across independent
Kaggle sessions. In every bridge configuration the target `Y--Y` scope is
test-only; it never contributes training or validation pairs.

## `02_cross_dataset` — what it actually measures

`01_spectra_siam_train_on_codexglue.ipynb` trains one model on the CodeXGLUE train split, picks
its threshold on the CodeXGLUE **validation** split, freezes it, and applies that
same model and threshold to the test split of AtCoder, GPTCloneBench and
SemanticCloneBench.

Every benchmark numbers its codes from `1`, so the notebook prefixes each code id
with its dataset before merging the corpora. Without that the graphs of one
dataset silently overwrite the other's and the transfer number becomes noise.

Two F1 values are reported per target:

- **`F1`** — the frozen source threshold. This is the transfer result.
- **`OracleF1`** — the best threshold that target could have had. It is the
  ceiling a perfectly calibrated model would reach and must not be quoted as the
  transfer score; the gap between the two is a calibration gap, not skill.

## `03_cross_language` — what it actually measures

`01_spectra_siam_semanticclonebench_v3.ipynb` and
`05_spectra_siam_gptclonebench_v3.ipynb` train one model per
language on that language's mono-language pairs, then evaluate each model on
every language's test split. The output is a full source × target matrix: the
diagonal is in-language performance, everything off it is zero-shot transfer.

Only these two benchmarks can answer the question. CodeXGLUE is Java-only, and
every AtCoder pair is already Java↔Python, so neither has a
train-on-one-language / test-on-another split.

Both run on **AST only** (`INPUT_RELATION_INDICES = (0,)`). DDG is exported
natively for Java and C, but Python and C# graphs come from local tree-sitter/ast
builders whose node ids do not align with Joern's, so their projected DDG
relation is empty. Leaving DDG enabled would give two of the four languages an
extra relation the other two lack, and that asymmetry would surface as a
"language effect" that is really a pipeline artefact.

Per-language train splits are small (1.4k–4k pairs), so these runs are quick but
data-starved. That is a property of the benchmarks, not of the model, and it is
why the diagonal should be read as the reference point rather than compared with
the full-data numbers in the main table.

## Per-language results

Every baseline now also writes `<dataset>_language_breakdown.csv` with test P/R/F1
per language plus an `ALL` row, taken from the same run rather than from separate
per-language training. Pairs whose endpoints differ in language are reported under
a `java->python` style key so cross-language pairs stay visible.

## Shared conventions

- `SEED = 42` and the same stratified pair sampling as the main runs, so numbers
  are comparable with the published table.
- Thresholds are always selected on the **validation** split and frozen for test.
- `RUN_PROFILE` follows the main notebooks; `comparison_50k` is the default here
  because these experiments compare many configurations and need equal data
  budgets more than they need maximum absolute scores.
- Unless an experiment says otherwise, `USE_SOURCE_LEXICAL = False`: the graphs
  keep their node labels (so `Control_If`, `Call_Expr`, identifiers-as-hashed-
  sketch are available) but raw source tokens are not injected.

## Reading the results

`03_cross_language` and `02_cross_dataset` report a zero-shot transfer number.
A model that scores near `F1 = 0.667` with `R = 1.0` has collapsed to predicting
"clone" for every pair — that is a failure mode, not a result, and the notebooks
flag it explicitly in a `Collapsed` column.
