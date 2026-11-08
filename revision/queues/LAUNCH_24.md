# Remaining baseline roster

For the current input-fix priorities use [the eight-run list](RUN_NOW_INPUT_V2.md).
This roster also retains expensive baselines deferred for now; it is not an instruction to launch all rows immediately.

Paths below open the actual `.ipynb` files in this workspace.
Attach the dataset named in each row. Keep seed 42 and `FINAL_TEST=False`.
Account numbers are planning labels; no credentials are stored.

Complete/running jobs, resumable pauses and outputs awaiting review are excluded.
Confirm the tracker before starting
another copy of a job. Runtime and memory still need verification per method.

## Pending baseline jobs: 11

| Slot | Dataset | Notebook | Internet |
| ---: | --- | --- | --- |
| 1 | codenet | [codenet_full_seed42](../01_baselines_seed42/codenet/codenet_full_seed42.ipynb) | False |
| 2 | atcoder | [atcoder_full_seed42](../01_baselines_seed42/atcoder/atcoder_full_seed42.ipynb) | False |
| 3 | atcoder | [atcoder_codebert_seed42](../01_baselines_seed42/atcoder/atcoder_codebert_seed42.ipynb) | True |
| 4 | atcoder | [atcoder_cdlh_seed42](../01_baselines_seed42/atcoder/atcoder_cdlh_seed42.ipynb) | False |
| 5 | atcoder | [atcoder_fa_ast_gmn_seed42](../01_baselines_seed42/atcoder/atcoder_fa_ast_gmn_seed42.ipynb) | False |
| 6 | atcoder | [atcoder_c4_adapted_seed42](../01_baselines_seed42/atcoder/atcoder_c4_adapted_seed42.ipynb) | True |
| 7 | atcoder | [atcoder_qwen3_lora_seed42](../01_baselines_seed42/atcoder/atcoder_qwen3_lora_seed42.ipynb) | True |
| 8 | bigclonebench | [bigclonebench_full_seed42](../01_baselines_seed42/bigclonebench/bigclonebench_full_seed42.ipynb) | False |
| 9 | bigclonebench | [bigclonebench_codebert_seed42](../01_baselines_seed42/bigclonebench/bigclonebench_codebert_seed42.ipynb) | True |
| 10 | bigclonebench | [bigclonebench_cdlh_seed42](../01_baselines_seed42/bigclonebench/bigclonebench_cdlh_seed42.ipynb) | False |
| 11 | bigclonebench | [bigclonebench_qwen3_lora_seed42](../01_baselines_seed42/bigclonebench/bigclonebench_qwen3_lora_seed42.ipynb) | True |

## Start when slots free: 0 jobs

| Slot | Dataset | Notebook | Internet |
| ---: | --- | --- | --- |

## Exact dataset versions

See [the dataset guide](../datasets/README_FA.md) for ZIP paths and hashes.
Preserve every saved output and [update the run tracker](../results/run_progress.csv).
