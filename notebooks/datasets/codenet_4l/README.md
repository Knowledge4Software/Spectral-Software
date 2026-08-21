# Project CodeNet 4L graph pipeline

The source artifact is `C:\PyProjects\spectrals\data\codenet dataset.zip`.
This compact release already contains 120 pair Parquets: ten language
configurations times four pair kinds times train/validation/test. Stage 01
preserves these official pair-level 70/15/15 splits; it does not hash or move
pairs between splits.

The complete release contains 400,000 pairs and up to 414,960 referenced
programs. Its split is pair-disjoint but not endpoint/problem-disjoint, exactly
as documented by the source release.

Prepare the complete release without running Joern:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/run_pipeline/run_all.py `
  --stop-after 01 --force-prepare
```

Run the complete graph, spectral, and clean-data export after Stage 01:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/run_pipeline/run_all.py --start-at 02
```

For a quick pipeline smoke test, keep every bucket represented. A size divisible
by 800 gives exact 70/15/15 counts inside all 40 buckets:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/run_pipeline/run_all.py `
  --prepared-dir C:\PyProjects\spectrals\data\codenet_4l_smoke `
  --sample-size 800 --stop-after 01 --force-prepare
```

## Current non-clone scope study (12,000 pairs)

The current study excludes mutation pairs and builds one shared graph package:

- 4,000 `clone` pairs;
- 4,000 `hard_nonclone` pairs (Accepted/Wrong-Answer from the same problem);
- 4,000 `nonclone_diff_problem` pairs.

All graph layers currently consumed by either SPECTRA or the baseline suite are
extracted, spectrally encoded, and included in the clean export: `ast`, `cfg`,
`ddg`, and `cpg`. The SPECTRA notebooks themselves use AST+DDG and synthesize
their `next_token` relation from AST node order; CFG/CPG are retained so the
same package can later run DeepSim, FA-AST, GNN, and SNN baselines.

Before deterministic uniform sampling, both endpoints of every eligible pair
must contain between 20 and 50 physical source lines, inclusive. This removes
very short snippets and expensive long programs while preserving the exact
per-language and 70/15/15 pair quotas.

Every pair kind is uniform over all ten language configurations (400 per
configuration) and keeps the source 70/15/15 split (280/60/60 per
configuration). Prepare and audit the package without starting Joern:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/nonclone_scope_study/run_pipeline.py `
  --stop-after 01 --force-prepare
```

Then build every graph, export spectral features, and package the clean data:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/nonclone_scope_study/run_pipeline.py `
  --start-at 02
```

The fixed runner writes prepared input to
`C:\PyProjects\spectrals\data\codenet_4l_nonclone_12k_prepared` and graph output
to `C:\PyProjects\spectrals\outputs\codenet_4l_nonclone_12k`. The mutation rows
remain in the source ZIP; they are simply not selected.

The older clone-versus-mutation command is intentionally no longer the current
protocol. Its source data and legacy notebooks are retained for reproducibility.

## All clone graphs (100,000 pairs)

To graph every unique endpoint referenced by the complete CodeNet clone split,
run the incremental all-clone builder:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/all_clones/run_pipeline.py
```

The clone release contains 100,000 pairs and 152,867 unique program endpoints.
The runner reuses compatible AST/CFG/DDG/CPG records from the existing 12k
clean package by stable `source_code_id` plus `source_sha256`, remaps their
numeric IDs, and sends only missing programs to Joern. Missing programs are
processed in durable batches of 5,000. If the command is interrupted, run the
same command again: completed cache shards are skipped automatically.

Python and C# use the existing preferred source-parser path directly. The
ordinary pipeline already replaces their Joern AST/CFG/DDG with these fallback
graphs, so the fast path avoids the redundant Joern parse/export/method-map
work. Java and C++ continue to use authoritative Joern graphs. Resume is also
stage-aware inside a batch: completed extraction or cleaned-graph manifests are
reused even if the batch was stopped before being committed to the final cache.

For a one-batch smoke run without packaging:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/all_clones/run_pipeline.py `
  --stop-after graphs --max-batches 1
```

The final uploadable archive is written to
`C:\PyProjects\spectrals\outputs\codenet_4l_all_clones\codenet_4l_all_clones_clean_data.zip`.

## Uniform 50,000-clone subset

The reduced production target keeps exactly 5,000 clone pairs in each of the
ten mono/cross-language configurations. Every configuration independently
contains 3,500 train, 750 validation, and 750 test pairs. Run it with:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/clone_50k/run_pipeline.py
```

This runner shares the validated graph cache at
`outputs/codenet_4l_all_clones/graph_record_cache`; graphs completed by the old
all-clone runs are filtered into the 50k target without copying or rebuilding
them. It can also recover complete cleaned-graph manifests left in interrupted
batch work and resumes their spectral shards. The final archive is
`outputs/codenet_4l_clone_50k/codenet_4l_clone_50k_clean_data.zip`.

The package is minimized against the complete baseline run order, not only the
proposed method. See `clone_50k/CONSUMER_SCHEMA.md` for the audited field-level
contract. All four graph layers and spectra are required; only redundant
serialization fields and non-AST node labels are omitted. Spectral extraction
also constrains nested BLAS fan-out by default, while existing completed feature
shards remain resumable.

## Fixed 50k clone + 50k different-problem non-clone release

The graph-free Kaggle pair release at
`outputs/kaggle_datasets/codenet_4l_clone50k_diff50k.zip` contains the exact
same 50,000 clone pairs as the preceding target plus 50,000 fixed
different-problem non-clone pairs. To build the graph-and-spectral companion
package, first let the clone build finish, then run:

```powershell
.\.venv\Scripts\python.exe notebooks/datasets/codenet_4l/clone50k_diff50k/run_pipeline.py
```

This runner reuses the shared `codenet_4l_all_clones/graph_record_cache` by
stable source ID and source hash. The combined selection has 135,068 unique
endpoints: 86,585 from the clone target and 48,483 introduced by the non-clone
half. It builds only the missing endpoint graphs/spectra and writes
`outputs/codenet_4l_clone50k_diff50k/codenet_4l_clone50k_diff50k_clean_data.zip`.
The original graph-free Kaggle ZIP is not overwritten.

An optional `--sample-size N` takes a deterministic uniform subset across the
selected configuration/pair-kind buckets while retaining the source split
ratio. Stage 02 checks the archive hash, sampling mode, configuration list, pair
kinds, and requested size so stale prepared data cannot be resumed silently.

If a run is stopped with Ctrl+C, the active Joern process tree and temporary
files are cleaned. Rerunning the same command restarts the current language.
