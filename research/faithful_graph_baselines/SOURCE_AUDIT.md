# Baseline source audit

This audit prevents the repository from presenting adaptations as unchanged
upstream implementations.  The graph parser and graph records always come from
this project.

| Baseline | Primary reference checked | Upstream code status | Preserved here | Deliberate difference |
|---|---|---|---|---|
| DECKARD | [official release](https://github.com/skyhover/Deckard) | Author release available | rooted parse-tree/AST characteristic vectors and size-aware vector matching | supplied benchmark pairs replace corpus-wide LSH candidate generation; our AST replaces DECKARD's parser |
| ASTNN | [official clone model](https://github.com/zhangj111/astnn/blob/master/clone/model.py) | Author release available | recursive statement-subtree encoding, bidirectional GRU, temporal max pooling, absolute-difference clone head | our exported AST and node-type vocabulary replace language-specific parsing and Word2Vec preprocessing |
| RtvNN | *Deep Learning Code Fragments for Code Clone Detection* | no maintained author implementation located | recursive AST vector composition and auxiliary reconstruction | our AST categories replace the original parser/token representation; benchmark pairs are scored directly |
| CDLH | [IJCAI 2017 paper](https://www.ijcai.org/proceedings/2017/423) | no author repository located | binarized AST, AST-based LSTM, supervised pairwise hashing, compact binary code | left-child/right-sibling conversion is applied to our exported AST; the PyTorch loss is a documented adaptation |
| DeepSim | [FSE 2018 paper](https://doi.org/10.1145/3236024.3236068) | no stable author repository located | control- and data-flow semantic matrix, learned functional-similarity representation | portable CFG node categories plus CFG/DDG structural roles replace WALA's Java instruction features |
| FA-AST+GGNN/GMN | [SANER 2020 paper](https://arxiv.org/abs/2002.08653) | no author repository located | AST augmented by directed control/data-flow relations; GGNN gated propagation; GMN cross-graph attention | six directed AST/CFG/DDG channels replace the paper's Java-specific CondTrue/CondFalse/NextStmt/etc. construction |

The generic observed-graph GNN and all fixed-spectrum baselines are native
controls designed for this study; they are not claimed as reproductions of a
named external system.  The pretrained baselines are collaborator-supplied and
remain separated by provenance in the paper tables.

## Shared evaluation policy

All baseline variants use the official full train/validation/test split. The
selection rule is computed *after* a method has removed pairs lacking its
required graph view: exactly equal clone and non-clone validation counts select
thresholds/checkpoints by Accuracy (then F1); every other validation split uses
F1 (then balanced accuracy).
