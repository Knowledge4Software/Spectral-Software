# Fix the Hugging Face configuration error

The failure `RobertaConfig has no attribute '_commit_hash'` happened before
tokenization/training. The previous installer checked only a minimum Transformers
version and could accept version 5 despite the declared `<5` requirement.
Authentication is a separate issue; supplying a token does not repair this error.

The updated text notebooks install and verify Transformers 4.57.3, Hub 0.36.0
and tokenizers 0.22.1. Qwen LoRA also uses PEFT 0.17.1 and Accelerate 1.10.1.
Installation constrains PyTorch to the already installed build, preserving
Kaggle's CUDA installation. These versions are printed before model loading.
If a replaced library is already imported, setup requires a fresh kernel rather
than continuing with old in-memory modules.

## What to run now

1. Import the updated notebook for the failed job:
   [CodeBERT](../01_baselines_seed42/codenet/codenet_codebert_seed42.ipynb),
   [C4-adapted](../01_baselines_seed42/codenet/codenet_c4_adapted_seed42.ipynb), or
   [Qwen3-LoRA](../01_baselines_seed42/codenet/codenet_qwen3_lora_seed42.ipynb).
2. Use a fresh notebook/kernel or a fresh **Save Version -> Save & Run All**.
   Do not continue the failed kernel after replacing its libraries.
3. Attach the same audited CodeNet dataset. Enable GPU and Internet.
   Keep seed 42, `RESUME_CHECKPOINT=None`, `FINAL_TEST=False`.
4. Look for `status="hf_environment_ready"` and the versions listed above.
   Save the dependency log with the model's output.

The reported failure occurred before optimizer training, so it needs no training
checkpoint. Preserve outputs if a different job has already trained; consult its
recorded dependencies before changing its environment mid-experiment.

The fix applies to text jobs on every dataset/seed and their transfer templates,
plus optional frozen Qwen extraction. The six graph baselines and completed
SPECTRA job do not need to restart. Training modules, dataset releases, objectives,
sampling, optimization settings and the training implementation hash are unchanged:
`793a3d2debc4cac7c045fb34df66f4f12f0694538e3da7a046c1b0623cee2de4`.

`HF_TOKEN` is optional for these public repositories. If downloads are rate
limited, configure a read token privately using your environment/Kaggle secret;
never paste it into the notebook source or chat. See
[Hugging Face authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication).

Local verification checks real CodeBERT and Qwen configurations/tokenizers with
the supported libraries, as well as installer/version/restart behavior. It does
not replace a full CUDA training run on Kaggle.
