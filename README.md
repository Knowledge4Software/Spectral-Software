# SPECTRA-Siam

Replication package for the paper

> **Learning Spectral Representations of Code through Latent Graph Learning
> for Generalizable Cross-Language Code Clone Detection**

Code clone detectors usually compare fixed, language-specific graphs such as
ASTs or program dependence graphs. Functionally identical fragments can produce
very different structures, so those rigid graphs yield spectra that barely
separate clones from non-clones.

SPECTRA-Siam instead *learns* the graph. From a fragment's AST and data
dependencies it induces a fixed-size weighted latent graph, then takes a
multi-scale spectral representation from that graph's normalized Laplacian.
Every fragment is mapped into the same latent space, so the resulting spectra
stay comparable across programming languages — which is what makes the method
transfer to language pairs it never saw in training.

## Kaggle links

The three benchmarks used in the paper:

| Dataset | Paper name | Link |
| --- | --- | --- |
| CodeNet 4L | CodeNet — four languages, ten configurations | https://www.kaggle.com/datasets/koushamoeini/codenet-4l |
| AtCoder | AtCoder — Java–Python | https://www.kaggle.com/datasets/koushamoeini/atcoder |
| CodeXGLUE | BigCloneBench — Java–Java | https://www.kaggle.com/datasets/koushamoeini/codexglue |

The 100-epoch training-duration run of the proposed method, reported in
Section 6.2:

- https://www.kaggle.com/code/kaggle7kousha/section-6-2-effect-of-training-duration-all

How the CodeNet benchmark was built, in order:

| Step | What it does | Link |
| --- | --- | --- |
| 1 | Extracts the four languages from Project CodeNet, forms clone pairs, and builds the initial non-clones — both different-problem negatives and hard negatives from accepted and rejected submissions to the same problem. | https://www.kaggle.com/code/alisdqi2002/codenet-four-language-high-diversity-clone-pairs |
| → | The clone-pair corpus produced by step 1. | https://www.kaggle.com/datasets/alisdqi2002/project-codenet-4l-high-diversity-clone-pairs |
| 2 | Generates further non-clones from the clone pairs by mutation. | https://www.kaggle.com/code/alisdqi2002/cetbench-derived-non-clone-generation |
| 3 | Draws the uniform random 100K-pair benchmark — 10K pairs for each of the ten language configurations. | https://www.kaggle.com/code/alisdqi2002/project-codenet-4l-uniform-random-10k-pair-splits |
| → | The 40-bucket 10K random splits produced by step 3. | https://www.kaggle.com/datasets/alisdqi2002/codenet-4l-40bucket-10k-random-splits |
| 4 | Graph and spectral extraction, run locally, producing the `codenet-4l` dataset linked above. | `create_datasets_graphs/codenet_4l/` |

`codexglue` is the dataset the paper reports under the name **BigCloneBench**.
