# Spain Kaggle manager

Spain orchestrates jobs that train on Kaggle. The three AtCoder mean/max
controls completed and their reports were collected. The queue now includes
one continuation session each for CodeNet C4-adapted, CodeBERT, Qwen3-LoRA,
and CDLH. C4-adapted and CDLH use account 7, which has two manager slots.
CodeBERT uses account 3 and Qwen3-LoRA uses account 8. The
continuations attach saved Kaggle outputs and use the original Accuracy
selection protocol.

Four seed 42 AtCoder external baselines were also submitted on 2026-11-22.
C4-adapted, CodeBERT, CDLH, and Qwen3-LoRA use accounts 1, 2, 4, and 5,
respectively. They use the audited AtCoder v2 dataset and their registered
Accuracy selection protocol. Final testing stays off.

On 2026-11-23, CodeNet CDLH finished by validation patience. Seven CodeNet
and AtCoder runs saved checkpoints at their session limits, and the queue
registered a new continuation for each. BigCloneBench CodeBERT, CDLH and
Qwen3-LoRA were assigned to accounts 9, 10 and 11 using the audited v2
dataset and F1 selection. The prepared BigCloneBench mean/max readout controls
for seeds 42, 123 and 231 use accounts 6, 12 and 13, respectively. They compare
readouts under the full objective, using the existing BigCloneBench full
runs as references. Both existing full seed 231 attempts remain in the evidence.
The live submission receipts and job phases are in the Spain ledger.

## Files and credentials

- SSH alias: `cloudfpsspain`, in Ubuntu WSL's SSH configuration.
- Installation and mutable state: `/opt/spectral-kaggle/`.
- Python environment: `/opt/spectral-kaggle/venv/`, Kaggle CLI2.2.5 plus `six`.
- Credentials: `/root/.config/spectral-kaggle/accounts.json`, permissions0600,
  parent0700. This file is outside Git and is never included in reports.
- `queue.json`: explicit account, kernel slug, notebook hash, dataset identity,
  expected experiment protocol and pinned container image.
- `jobs/<job_id>/`: unchanged registered notebook and submission metadata.
- `state.json`: submission receipts, version numbers, quota snapshots and status.
- `reports/<job_id>/<version>/`: output inventory and selected small reports.

All three AtCoder notebooks remain private, Internet off and final test off. They use
the existing `koushamoeini/atcoder-revision-common-support-v2` dataset and the
same T4 container image recorded by their completed spectral references.
T4 API allocation is a T4x2 machine; the frozen training code uses one device.
The experiment's internal10h session budget is unchanged; API timeout is11h.

## Monitor and stop

From PowerShell:

```powershell
wsl -d Ubuntu -- ssh cloudfpsspain 'systemctl status spectral-kaggle --no-pager'
wsl -d Ubuntu -- ssh cloudfpsspain 'cat /opt/spectral-kaggle/state.json'
wsl -d Ubuntu -- ssh cloudfpsspain 'journalctl -u spectral-kaggle -n 20 --no-pager'
```

The dedicated service polls every3min, is limited to10% of one CPU and192MiB
RAM, uses low CPU/I/O priority, and keeps at least1GiB free before a new cycle.
No VPN service, firewall, routing, SSH configuration or existing service is
modified or restarted. CPU/RAM limits reduce contention; they cannot guarantee
zero latency effect on shared hardware. Remote client installation required no
system package upgrades.

```powershell
wsl -d Ubuntu -- ssh cloudfpsspain 'systemctl stop spectral-kaggle'
```

Stopping the manager **does not cancel Kaggle jobs**. Do not erase `state.json`:
its pre-request submission marker prevents a timeout or crash from duplicating
an expensive experiment. Each job gets a dedicated slug that this manager never
overwrites. A process lock prevents two manager instances from submitting.

## Quota and output policy

GPU hours used **and reserved** count against admission. Each new job requires
at least11h remaining at submission; actual training can stop earlier. Kaggle
enforces any other account/session admission limits. The manager defaults to one job per account. The queue configuration can
set a different limit for an individual account.

Checkpoints, graph caches, embeddings and token files stay on Kaggle. Only
`status.json`, `benchmark.json`, `cost.json`, and `summary.json` are selected;
each is capped at8MiB and their total at32MiB per job. The inventory records full relative paths from the Kaggle session output endpoint.

Worker COMPLETE does not establish scientific completion: the manager checks
the saved experiment status, completed flag, seed, selection metric, arm, source
hash, data hash and final-test flag. Collected reports still need scientific
review; they are not independent verification of checkpoint weights.

Budget pauses become `needs_continuation`; failed jobs and protocol mismatches
need review. This first deployment does not automatically resume, retry or
change any experiment. A submission timeout becomes `submission_uncertain`
and must be reconciled with Kaggle before another launch. Read/API collection
failures can be checked again on later polls. No retry depends on F1 performance.

The six-input CPU collector remains a later step: the three new outputs plus
three phase06 spectral classification-only outputs. No collector is autoqueued
before all six references and cross-account permissions are verified.
