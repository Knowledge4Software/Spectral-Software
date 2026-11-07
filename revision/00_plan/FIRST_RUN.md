# CodeNet pilot setup and continuation

Execution update, 2026-11-07: the user reported this pilot complete at epoch 43
on a Tesla T4, with test evaluation off. Reuse it. The next 24 independent jobs
are in [the launch list](../queues/LAUNCH_24.md). The setup below is retained
for reproducibility and checkpoint continuation.

## Files

1. Current notebooks: open the actual files in `revision/01_baselines_seed42/`.
2. Dataset: `C:/PyProjects/spectrals/data/benchmarks/codenet_revision_common_support.zip`.
3. Pilot notebook: [codenet_full_seed42.ipynb](../01_baselines_seed42/codenet/codenet_full_seed42.ipynb).

The [workspace guide](../START_HERE_FA.md) links the screening queue, protocol,
checksums and CPU validation collector. Later seeds and ablations have separate
folders. ZIPs are optional transport copies; the original v3 packet is preserved
under `output/revision/archive/releases/` with its original paths and bytes.

## Kaggle setup

1. Create a private Kaggle dataset from the CodeNet ZIP. Wait for processing.
   In its file list, confirm `metadata.json`, `pairs.csv.gz` (or `pairs.csv`),
   `codes.jsonl.gz`, `graph_spectra.jsonl.gz` and the DDG alignment sidecar exist.
   Kaggle may decompress files; the loader recognizes compressed/plain payloads.
   If only a ZIP file is visible, extract it before use; the notebook discovers
   directories containing `metadata.json`, not ZIP members.
2. Import `codenet_full_seed42.ipynb` as a new notebook. Attach only this audited
   CodeNet bundle. The old CodeNet archive is a different experiment.
3. Enable a GPU. Internet is not required for this SPECTRA notebook. The current
   trainer uses one visible GPU; selecting a two-GPU accelerator does not double
   its training speed. Do not replace Kaggle's CUDA PyTorch installation.
4. Keep the first configuration cell at its generated defaults:

```python
ARM = 'full'
SEED = 42
MAX_EPOCHS = 100
MIN_EPOCHS = 10
PATIENCE = 10
CLEAN_DATA_DIR = None
RESUME_CHECKPOINT = None
ALLOW_NUMERICAL_UPGRADE = False
FINAL_TEST = False
MEASURE_COST = True
SESSION_HOURS = 10.0
```

5. Choose **Save Version -> Save & Run All**. This is the research run, not a
   four-epoch smoke test. Do not change epochs to 4. The 100 value is a ceiling;
   validation stopping can finish earlier. A long job can span multiple sessions.

## What to expect

After input preparation the log prints the GPU, data hash and pair counts:
69,538 train / 14,896 validation / 14,906 test. Reading test metadata is not
evaluating test predictions; `FINAL_TEST=False` prevents test scoring.
Each finished epoch prints training loss and validation metrics. Periodic saves
print a runtime forecast. Preparation of the large graph file can take time
before the first epoch; CPU checks cannot predict actual Kaggle GPU throughput.
The progress release adds tqdm for notebook cells, compressed/plain file bytes,
training/evaluation batches and token/embedding cache creation. Training bars
start at the saved batch cursor on resume. These display adapters do not change
the scientific training fingerprint, model, optimizer or sampling schedule.
An already running older notebook can finish; do not restart solely for tqdm.

The v3 numerical fix separately updates training code to replay AMP-overflow
batches, with bounded FP32 fallback. An existing pre-fix checkpoint needs the
explicit, checked upgrade in [NUMERICAL_RECOVERY.md](NUMERICAL_RECOVERY.md).
An ordinary resume never silently accepts a different source snapshot.

Output directory: `/kaggle/working/revision_full_seed42/`.
Retain the whole output version, especially:

- `status.json`: completion, history, best validation, GPU and timing estimate.
- `progress.pt`: weights, optimizer, scaler, random states and continuation cursor.
- `benchmark.json`, if training completed with enough time for measurement.

Do not expect `final_test.json` at this stage. A missing benchmark after a clock
stop is not lost training; it can be measured from the completed checkpoint later.

## If the session ends before training completes

`status="incomplete_session_budget"` is an expected safe stop. The internal work
deadline is around 9h15, leaving output margin before your 12-hour session limit.
It is not yet a completed experiment and it must not be ranked as one.

1. Preserve that notebook's output version and attach it as input to a new
   version of the **same notebook**, along with the same CodeNet dataset.
2. Set the first-cell variable to the actual attached checkpoint path, for example:

```python
RESUME_CHECKPOINT = '/kaggle/input/YOUR-PREVIOUS-OUTPUT/revision_full_seed42/progress.pt'
```

3. Keep every scientific setting, seed, dataset and embedded source snapshot
   unchanged. Save & Run All again. Do not start over, shorten the schedule or
   edit the checkpoint. If a restarted interactive session already has files in
   `/kaggle/working`, use a fresh saved session to avoid ambiguous local progress.

If Kaggle reports an OOM, import/dependency error, non-finite loss or identity
mismatch, preserve the output/log and stop that job. A scientific setting change
needs a fresh configuration; it is not a valid resume of the existing optimizer.

## What to send back before the next jobs

Send `status.json` and any error traceback. If training has started but not
finished, the epoch logs and runtime forecast are enough to assess continuation.
Do not send the large checkpoint through chat; keep its saved output on Kaggle.
We need the real GPU speed before deciding how many sessions later jobs require.

The pilot is now reported complete. Continue with the independent seed-42 queue
across CodeNet, AtCoder and BigCloneBench; ASTNN and CodeBERT are hardware checks
for other model paths, and can run alongside the other jobs.
Use the dataset named in each row. Transformer jobs require Internet/model
download and preserve their `tokens` cache. No seeds 123/231 and no transfer
matrix are required at this first stage.

To summarize saved outputs later, run
`05_analysis/validation_screening.ipynb` on CPU with those outputs attached.
Optionally attach `stage1_run_queue.csv` and set `QUEUE_CSV` to its input path
to produce a ledger including jobs that have not started. It reads validation
status only; it does not open test scores or deserialize checkpoints.
