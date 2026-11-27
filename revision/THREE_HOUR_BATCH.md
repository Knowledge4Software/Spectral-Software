# Three-hour B300 batch

Use the complete [server runbook](SERVER_RUNBOOK.md) to stage Linux, the three
audited datasets, the model cache, and any accessible checkpoints. Stage them
before claiming the GPU node where possible. The job list is generated from
frozen notebook templates and preserves all seeds and training settings.

```bash
source .venv-server/bin/activate
python scripts/run_revision_server.py list --profile lease --external c4_adapted
python scripts/run_revision_server.py verify --profile lease --hardware --data /storage/server-data
python scripts/run_revision_server.py run --profile lease --external c4_adapted \
  --data /storage/server-data --output /storage/server-output \
  --gpus 0,1,2,3,4,5,6,7 --lease-hours 3
```

Set the lease value to the actual time left after setup. Include
`--config /storage/resume-inputs.json` if importing checkpoints. The
seven baseline entries default to fresh runs when given no checkpoints. For
an original input-v1 checkpoint, set `NOTEBOOK_VARIANT` in the config for
the queued job. For example, set it to `codenet_codebert_seed42__input_v1`
under `codenet_codebert_seed42` alongside the checkpoint and token paths.
At handover, confirm CUDA can execute a kernel on all eight devices before the
batch. This catches driver and CUDA architecture problems that a device list
alone can miss:

```bash
python - <<'PY'
import torch
assert torch.cuda.device_count() == 8
for index in range(8):
    x = torch.ones((64, 64), device=f"cuda:{index}")
    assert (x @ x)[0, 0].item() == 64
    print(index, torch.cuda.get_device_name(index))
PY
```

Use the selected external comparator only after freezing its place in the
study. C4-adapted is the current choice based on completed CodeNet validation.

The first eight jobs are the three dataset-specific CodeBERT baselines,
BigCloneBench CDLH, and the seed 42 full SPECTRA and mean/max comparisons in the
source-only and direct transfer conditions. Freed GPUs take the remaining
seed 42 transfer jobs, the three Qwen baselines, and then seeds 123 and 231
of the transfer study. The queue contains 52 jobs. Completion in one lease
depends on measured runtimes and checkpoint availability. It stores progress
for the next lease and leaves time for output export.

The `matrix` profile carries the complete training matrix and the focused
transfer design. The `all` inventory also includes analysis, optional
transfer candidates, and archived variants. Those are alternatives and
prerequisite-dependent jobs, so the paid batch does not rerun the full inventory.

After the GPU batch, collect validation evidence on CPU using the output tree
or a directory containing both old and new status reports:

```bash
python scripts/run_revision_server.py run --jobs validation_screening \
  --inputs /storage/server-output --output /storage/server-analysis
```

Final evaluation needs completed checkpoints and a frozen comparison roster.
Run selected IDs with `--final` to produce test predictions. The
`paired_problem_uncertainty` analysis consumes those predictions. The other
analysis notebooks are in the full inventory and need their documented
checkpoint or report inputs. Run the frozen Qwen extraction before fitting its
three CPU probes. Export checkpoints and reports before the lease ends.

## Live GitHub summaries

Marshall can run a second, detached process on its persistent workspace.
Install an authorized GitHub write credential there before assignment. Use
a repository-scoped write deploy key where possible. Keep it off the GPU
node and do not put it in scripts or notebook inputs. The repository is
public, so the publisher shares only validation progress and metrics.

After copying this checkout to the node, run from the workspace, substituting
the actual host, repository path on the node, and the unique lease ID:

```bash
python3 /data/workspace/spectra/scripts/publish_server_results.py \
  /home/ubuntu/spectra-lease/server-output \
  --ssh ubuntu@NODE --remote-script /home/ubuntu/spectra-lease/spectra/scripts/publish_server_results.py \
  --work /data/workspace/public-research/results/spectra/publish-RUN_ID \
  --remote git@github.com:Knowledge4Software/Spectral-Software.git \
  --run-id RUN_ID --interval 30
```

Run the publisher alongside training. It commits new validation reports as
they appear and retries failed pushes. Each lease has a separate
`gpu-results/RUN_ID` branch. Keep the existing full checkpoint copy-back to
workspace storage, and run one last publisher pass with `--once` before
releasing the node. Live GitHub summaries contain no checkpoints, logs,
training data, predictions or credentials.
