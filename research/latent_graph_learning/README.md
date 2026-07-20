# Latent Graph Learning

This is the self-contained research track beyond the benchmark baselines.
Nothing in this folder belongs to `kaggle/` baseline experiments.

## Layout

| Path | Purpose |
| --- | --- |
| `notebooks/01_siamese_latent_graph_kaggle.ipynb` | Self-contained Kaggle notebook. Copy this one file into Kaggle, attach the XGLUE4 input at `/kaggle/input/datasets/koushamoeini/xglue4`, then run all cells once. It performs full XGLUE4 training and emits one result set. |
| `notebooks/04_xglue4_kaggle_baseline_results.ipynb` | Imports the ten XGLUE4 Kaggle result ZIPs from `outputs/kaggle/`, creates a standardized 18-row comparison table, and renders a paper-style grouped table with the proposed method at the top. |
| `scripts/latent_graph_model.py` | Reusable PyTorch implementation of the AST encoder, learned latent graph, differentiable Laplacian spectrum, Siamese scorer, and BCE/contrastive/hybrid loss. |
| `README.md` | Method rationale, execution protocol, and ablation plan. |

## Method

The first experiment is intentionally compact:

1. AST node types plus tree depth form a fixed-length padded input.
2. One or two attention layers pool the AST into a small learned latent graph.
3. A differentiable latent Laplacian yields an eigenvalue representation.
4. A shared Siamese encoder learns clone similarity with BCE, contrastive, or
   hybrid supervision. High-similarity non-clones receive extra metric weight
   as in-batch hard negatives.

The model is not an LLM or an AST parser replacement. It is a task-specific
clone encoder with a deliberately small parameter count. Use
`notebooks/01_siamese_latent_graph_kaggle.ipynb` on XGLUE4.

## Training Protocol

The notebook defaults to `MAX_TRAIN_PAIRS = None`, so it trains on every
XGLUE4 training pair. It selects the threshold from validation only,
then evaluates once on the held-out test split. It saves a checkpoint, epoch
history, results CSV, and learning-curve plot for each dataset under
`/kaggle/working/`.

## First Ablations

Keep the data protocol fixed and compare `OBJECTIVE` (`"bce"`,
`"contrastive"`, `"hybrid"`), then vary `LATENT_NODES` (16, 24, 32),
`HIDDEN_DIM` (64, 96, 128), and `ATTENTION_LAYERS` (1, 2). Record F1,
precision, recall, parameter count, and runtime for each setting.
