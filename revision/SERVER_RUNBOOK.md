# Linux GPU server runbook

Use `scripts/run_revision_server.py` to run the revision suite directly on a
Linux NVIDIA host. It executes the frozen notebook cells in isolated Python
processes and gives every experiment its own output directory. It needs no
Kaggle account, notebook server, or root-level `/kaggle` directory. It preserves
embedded training sources, model settings, data inputs, and validation metrics.

## Stage the inputs before renting

1. Transfer this checkout or the prepared code ZIP. Keep it on storage that
   survives server shutdown. Run `bash scripts/setup_revision_server.sh` on
   the Linux host. Use a base image with a working CUDA PyTorch environment
   to reduce installation time. Set `SPECTRA_WARM_MODELS=1` to prefetch the
   pinned CodeBERT and Qwen model snapshots into a persistent Hugging Face cache.
2. Stage these three local ZIPs from `C:\PyProjects\spectrals\data\benchmarks`
   on the destination or on storage the destination can read:
   - `codenet_revision_common_support.zip`
   - `atcoder_revision_common_support_v2.zip`
   - `bigclonebench_revision_common_support_v2.zip`
3. Stage any accessible checkpoint outputs. Keep `progress.pt`, `status.json`,
   and the model's own `tokens/` directory together. Use only trusted
   checkpoints. Metrics-only reports do not provide optimizer state.
4. Use persistent storage for `server-output` and the Hugging Face cache.
   For a short National Compute lease, stage the archive and model downloads
   independently of the node. Export outputs to your own persistent storage
   before releasing its scratch disk.

```bash
bash scripts/setup_revision_server.sh
source .venv-server/bin/activate
python scripts/run_revision_server.py unpack --archives /storage/benchmark-zips --data /storage/server-data
python scripts/run_revision_server.py verify --profile all
python scripts/run_revision_server.py verify --profile remaining --hardware --data /storage/server-data
```

The unpack command verifies the three original archive hashes. If you already
transferred the audited unpacked directories, point `--data` to their parent.
Each experiment still uses its own notebook's dataset loader and data identity.

## Pick the work you want to run

| Profile | Contents |
| --- | --- |
| `remaining` | Seven baseline identities without a verified completed report. Fresh defaults are included. Import any recovered checkpoints separately. |
| `training` | The checked-in training suite and newer AtCoder follow-ups. Includes old and new protocol variants where both have an explicit notebook. |
| `focused` | Five conditions × three models × three seeds. Defaults to full SPECTRA, mean/max, and C4-adapted. Pass `--sibling` and `--external` to record the validation-selected comparators. |
| `matrix` | Training suite, additional architecture and loss controls on AtCoder and BigCloneBench, and the 45 selected transfer jobs. The additional cross-dataset controls are exploratory. |
| `all` | Inventory of the matrix, transfer candidates and optional reinforcement, analysis, frozen Qwen probe stages, historical starters, and archived checkpoint-compatible variants. Select the entries you want by ID or glob. |

The three seeds are 42, 123, and 231. BigCloneBench's baseline roster omits
C4-adapted because the audited support lacks the problem IDs its contrastive
objective uses. The historical transfer notebooks are listed separately from
the focused case study. See the existing experiment protocol for reporting.

```bash
# Inspect the exact jobs and resolve explicit IDs.
python scripts/run_revision_server.py list --profile all > experiment-inventory.json
python scripts/run_revision_server.py list --profile focused --external c4_adapted

# Example: first independent batch on an eight-GPU B300 allocation.
python scripts/run_revision_server.py run --profile remaining \
  --data /storage/server-data --output /storage/server-output \
  --gpus 0,1,2,3,4,5,6,7 --lease-hours 3

# Example: selected jobs on one RTX 3090, with resumable 10-hour sessions.
python scripts/run_revision_server.py run --jobs 'codenet_codebert_seed42' 'atcoder_codebert_seed42' \
  --data /storage/server-data --output /storage/server-output --gpus 0

# Use only after selecting the comparators on validation.
python scripts/run_revision_server.py run --profile focused --external c4_adapted \
  --data /storage/server-data --output /storage/server-output --gpus 0,1
```

Use `tmux` to keep the command alive after disconnecting SSH. One worker
draws jobs from the queue per listed GPU. Set `--threads` to match host CPU
capacity and use fewer workers if several concurrent graph loaders would
exhaust system RAM. `--lease-hours` is the time left **after setup**. It
reserves 15 minutes for export plus each experiment's existing 45-minute
checkpoint margin. It does not control provider billing or release the node.

## Resume and release results

Rerun the same command against the same output directory after a budget stop.
Previously completed jobs are skipped and jobs with local progress resume
from their saved cursor. Each invocation attempts a job once. Read its
`session-*.log`, `server-result.json`, and experiment `status.json` before
resuming a numerical or environment failure.

To import a trusted Kaggle checkpoint, put the paths in a JSON input
configuration. Paths are relative to the config file or absolute.

```json
{
  "codenet_codebert_seed42__input_v1": {
    "RESUME_CHECKPOINT": "checkpoints/codenet_codebert/progress.pt",
    "RESUME_TOKENS_DIR": "checkpoints/codenet_codebert/tokens"
  }
}
```

```bash
python scripts/run_revision_server.py run --jobs codenet_codebert_seed42__input_v1 \
  --config /storage/resume-inputs.json --data /storage/server-data \
  --output /storage/server-output --gpus 0
```

Use the archived `__input_v1` entry only for that implementation. Use the
modern entry for input-v2 checkpoints. The runner derives the pinned model
revision from the imported token cache and the frozen runtime rejects
incompatible checkpoint identities.

Final test evaluation is a separate invocation with `--final` and selected
job IDs. The runner requires a completed checkpoint and its report. Freeze the
comparison roster and thresholds first. For analysis notebooks, set
`--inputs` to a directory containing their required checkpoint or report
folders. Per-job `INPUT_ROOT`, `QUEUE_CSV` and checkpoint path overrides can
also be provided in the JSON configuration. Use the same output root for the
optional frozen Qwen extraction and three dependent CPU probes.

The separate local AtCoder learning controls are directly runnable with
`scripts/run_learning_controls_local.py`. They already have completed evidence.
