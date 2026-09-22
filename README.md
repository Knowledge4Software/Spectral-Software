<h1 align="center">SPECTRA-Siam</h1>

<p align="center">
  <strong>Learning Spectral Representations of Code through Latent Graph Learning<br>
  for Generalizable Cross-Language Code Clone Detection</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.22383"><img src="https://img.shields.io/badge/arXiv-2608.22383-b31b1b.svg" alt="arXiv: 2608.22383"></a>
  <a href="https://github.com/Knowledge4Software/Spectral-Software"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&amp;logoColor=white" alt="GitHub repository"></a>
  <a href="#datasets"><img src="https://img.shields.io/badge/Datasets-Kaggle-20BEFF?logo=kaggle&amp;logoColor=white" alt="Datasets on Kaggle"></a>
  <a href="#benchmark-construction"><img src="https://img.shields.io/badge/Artifact-Reproducibility-2ea44f" alt="Reproducibility package"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.22383">Paper</a> ·
  <a href="#method-overview">Method</a> ·
  <a href="#datasets">Datasets</a> ·
  <a href="#benchmark-construction">Benchmark construction</a> ·
  <a href="#citation">Citation</a>
</p>

Replication package for the paper **“Learning Spectral Representations of Code
through Latent Graph Learning for Generalizable Cross-Language Code Clone
Detection.”** The paper is available as an
[arXiv preprint](https://arxiv.org/abs/2608.22383).

## Overview

Code clone detectors commonly compare fixed, language-specific structures such as
abstract syntax trees and program dependence graphs. Functionally equivalent code can
have substantially different structures, especially across programming languages.
Their graph spectra may therefore provide weak separation between clones and
non-clones.

SPECTRA-Siam instead **learns the graph used for comparison**. It starts from a code
fragment's AST and data dependencies, induces a fixed-size weighted latent graph through
soft slot assignment and multi-head attention, and derives a multi-scale spectral
representation from the graph's normalized Laplacian. Every fragment is mapped into the
same latent space, allowing spectra to remain comparable across languages and supporting
transfer to language pairs not observed during training.

| | |
| --- | --- |
| **Task** | Cross-language and within-language code clone detection |
| **Core idea** | Learn a shared latent graph instead of comparing fixed source-code graphs |
| **Representation** | Multi-scale spectra of the learned graph's normalized Laplacian |
| **Evaluation resources** | CodeNet 4L, AtCoder, and BigCloneBench through CodeXGLUE |

## Results at a glance

| Evaluation | Comparison | Reported result |
| --- | --- | ---: |
| BigCloneBench | Fixed graphs → learned latent graphs, using the same classifier | F1: **0.37 → 0.67** |
| AtCoder | Fixed graphs → learned latent graphs, using the same classifier | Accuracy: **0.60 → 0.71** |
| CodeNet 4L | Full SPECTRA-Siam model | Accuracy: **0.69** at 4 epochs, **0.79** at 30 epochs |
| Unseen language transfer | Performance degradation across 60 bridge-assisted paths | **0.058**, compared with **0.112–0.228** for the baselines |

## Method overview

1. **Construct source relations.** Extract AST structure and data-dependency relations
   from each code fragment.
2. **Learn a latent graph.** Map the variable source structure to a fixed-size weighted
   graph in a shared latent space.
3. **Compute spectral features.** Derive a multi-scale representation from the
   normalized graph Laplacian.
4. **Compare code pairs.** Use the shared spectral representation to distinguish clone
   and non-clone pairs, including pairs from languages not seen together during
   training.

## Reproducibility resources

### Datasets

The paper evaluates SPECTRA-Siam on three benchmarks:

| Dataset | Name used in the paper | Kaggle |
| --- | --- | --- |
| CodeNet 4L | CodeNet, Java, Python, C++, and C# across ten language configurations | [Open dataset](https://www.kaggle.com/datasets/koushamoeini/codenet-4l) |
| AtCoder | AtCoder, Java–Python | [Open dataset](https://www.kaggle.com/datasets/koushamoeini/atcoder) |
| CodeXGLUE | BigCloneBench, Java–Java | [Open dataset](https://www.kaggle.com/datasets/koushamoeini/codexglue) |

> **Naming note.** The `codexglue` Kaggle dataset is reported as
> **BigCloneBench** in the paper.

### Reported training-duration experiment

The 100-epoch run of the proposed method reported in Section 6.2 is available as a
Kaggle notebook:

- [Section 6.2: effect of training duration](https://www.kaggle.com/code/kaggle7kousha/section-6-2-effect-of-training-duration-all)

### Benchmark construction

The CodeNet benchmark was produced through the following pipeline:

| Step | Output or operation | Resource |
| ---: | --- | --- |
| **1** | Extract four languages from Project CodeNet, form clone pairs, and build both different-problem negatives and hard negatives from accepted and rejected submissions to the same problem. | [Construction notebook](https://www.kaggle.com/code/alisdqi2002/codenet-four-language-high-diversity-clone-pairs) |
| **1a** | Clone-pair corpus produced by Step 1. | [Intermediate dataset](https://www.kaggle.com/datasets/alisdqi2002/project-codenet-4l-high-diversity-clone-pairs) |
| **2** | Generate additional non-clones from the clone pairs through mutation. | [Generation notebook](https://www.kaggle.com/code/alisdqi2002/cetbench-derived-non-clone-generation) |
| **3** | Draw the uniform random 100,000-pair benchmark, with 10,000 pairs for each of the ten language configurations. | [Sampling notebook](https://www.kaggle.com/code/alisdqi2002/project-codenet-4l-uniform-random-10k-pair-splits) |
| **3a** | Forty-bucket, 10,000-pair random splits produced by Step 3. | [Intermediate dataset](https://www.kaggle.com/datasets/alisdqi2002/codenet-4l-40bucket-10k-random-splits) |
| **4** | Extract graphs and spectral features locally to produce the final `codenet-4l` dataset. | [`create_datasets_graphs/codenet_4l/`](create_datasets_graphs/codenet_4l/) |

## Citation

If you use SPECTRA-Siam, its datasets, or this replication package, please cite the
paper:

```bibtex
@misc{hesamolhokama2026learningspectralrepresentationscode,
  title         = {Learning Spectral Representations of Code through Latent Graph
                   Learning for Generalizable Cross-Language Code Clone Detection},
  author        = {Mohsen Hesamolhokama and Ali Sadeghi and Kousha Moeini and
                   Behnam Rohani and Mohammadamin Fazli and Jafar Habibi},
  year          = {2026},
  eprint        = {2608.22383},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2608.22383}
}
```

## Repository

Source code and updates are available at
[Knowledge4Software/Spectral-Software](https://github.com/Knowledge4Software/Spectral-Software).
