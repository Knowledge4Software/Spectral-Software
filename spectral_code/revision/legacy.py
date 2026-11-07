"""Existing graph-baseline adaptations under the revision training protocol.

The encoder implementations are reused, not renamed SPECTRA encoders. Exported
node vocabularies are learned from training endpoints only. Differences from
the original papers remain documented in faithful_graph_baselines/core.py.
"""
from __future__ import annotations

import hashlib
import json
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import CleanDataCorpus, _open_text, pair_code_ids


@dataclass(frozen=True)
class LegacyConfig:
    method: str
    vocab_size: int
    support_code_hash: str
    family: str = "legacy_graph_adaptation"


def runtime(max_nodes=256):
    """Isolate each experiment's packing settings from module globals."""
    from spectral_code import faithful_graph_baselines
    root = Path(faithful_graph_baselines.__file__).parent
    source = (root / "notebook_runtime.py").read_text(encoding="utf-8")
    module = types.ModuleType("revision_legacy_runtime")
    module.__dict__.update(MAX_NODES=max_nodes, MAX_EDGES=512, MAX_STATEMENTS=64)
    exec(compile(source, str(root / "notebook_runtime.py"), "exec"), module.__dict__)
    digest = hashlib.sha256(((root / "core.py").read_text(encoding="utf-8") + source).encode()).hexdigest()
    return module, digest


class _Rows(Dataset):
    def __init__(self, frame, index):
        self.rows = [(index[str(a)], index[str(b)], float(y))
                     for a, b, y in frame[["left_id", "right_id", "label"]].itertuples(index=False, name=None)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class LegacyCorpus:
    def __init__(self, arrays, index):
        self.arrays, self.index = arrays, index

    def pair_loader(self, frame, config, batches=None):
        kwargs = {"batch_sampler": batches} if batches is not None else {"batch_size": config.batch_size}
        return DataLoader(_Rows(frame, self.index), num_workers=0,
                          generator=torch.Generator().manual_seed(config.seed + 9187), **kwargs)


class LegacyPairModel(torch.nn.Module):
    def __init__(self, config, corpus, module):
        super().__init__()
        self.config = config
        self.network = module._PairModel(config.method, corpus.arrays, config.vocab_size)

    def forward(self, left, right, temperature=None):
        logits, reconstruction, a, b = self.network(left.long(), right.long())
        return logits, {"reconstruction": reconstruction, "left": a, "right": b}

    def compute_loss(self, logits, auxiliary, labels, *, positive_class_weight=1.0, **unused):
        classification = F.binary_cross_entropy_with_logits(
            logits.float(), labels.float(), pos_weight=logits.new_tensor(positive_class_weight))
        extra = logits.new_zeros(())
        if self.config.method == "rtvnn":
            extra = .05 * auxiliary["reconstruction"]
        elif self.config.method == "cdlh":
            extra = .01 * sum((auxiliary[side].abs() - 1).square().mean() for side in ("left", "right"))
        return classification + extra, {"classification": classification.detach(), "auxiliary": extra.detach()}


def prepare_legacy(root, frames, method, config, work_dir):
    """Stream exports into compact arrays; fail rather than change test support."""
    if method not in {"astnn", "rtvnn", "cdlh", "deepsim", "fa_ast_ggnn", "fa_ast_gmn"}:
        raise ValueError(f"Unsupported baseline: {method}")
    module, digest = runtime(config.max_nodes)
    corpus = CleanDataCorpus(root)
    ids = sorted(pair_code_ids(*frames.values()))
    train_ids = pair_code_ids(frames["train"])
    if method in {"astnn", "rtvnn", "cdlh"}:
        arrays, vocab, usable, index = module._load_tree_arrays_streaming(corpus.graphs_path, ids, train_ids)
    elif method.startswith("fa_ast"):
        arrays, vocab, usable, index = module._load_fa_arrays_streaming(
            corpus.graphs_path, ids, train_ids, ("ast", "cfg", "ddg", "cpg"))
    else:
        # Two passes avoid holding decoded JSON graph dictionaries in host RAM.
        frequency = {}
        with _open_text(corpus.graphs_path) as stream:
            for line in stream:
                row = json.loads(line)
                if str(row.get("code_id")) in train_ids:
                    adjacency = module._adjacency(row.get("graphs", {}).get("cfg"))
                    for node in adjacency.get("node_types", [])[:config.max_nodes]:
                        key = str(node)
                        frequency[key] = frequency.get(key, 0) + 1
        vocab = {"<pad>": 0, "<unk>": 1}
        vocab.update({key: i + 2 for i, (key, _) in enumerate(sorted(frequency.items(), key=lambda x: (-x[1], x[0])))})
        index, usable = {key: i for i, key in enumerate(ids)}, set()
        arrays = {"node_types": np.zeros((len(ids), config.max_nodes), np.int32),
                  "numeric": np.zeros((len(ids), config.max_nodes, 12), np.float32),
                  "edge_src": np.full((len(ids), module.MAX_EDGES), -1, np.int32),
                  "edge_dst": np.full((len(ids), module.MAX_EDGES), -1, np.int32)}
        with _open_text(corpus.graphs_path) as stream:
            for line in stream:
                row = json.loads(line)
                key = str(row.get("code_id"))
                if key not in index:
                    continue
                layers = {name: module._adjacency(info) for name, info in row.get("graphs", {}).items()}
                if not layers.get("cfg", {}).get("node_types"):
                    continue
                values = module._deepsim_arrays(layers, vocab)
                for name, value in zip(arrays, values):
                    arrays[name][index[key]] = value
                usable.add(key)
    missing = sorted(set(ids) - usable)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    support = {"method": method, "required_endpoints": len(ids), "usable_endpoints": len(usable),
               "missing_ids": missing, "vocabulary_training_endpoints_only": True,
               "max_nodes": config.max_nodes, "max_edges_per_relation": module.MAX_EDGES,
               "interpretation": "portable adaptation of existing project baseline; not original-paper reproduction"}
    (work_dir / "baseline_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
    if missing:
        raise ValueError(f"{method}: {len(missing)} endpoints missing required graph views. Freeze common support for ALL methods first; see baseline_support.json")
    packed = LegacyCorpus(arrays, index)
    model_config = LegacyConfig(method, len(vocab), digest)
    def factory(settings):
        if settings.support_code_hash != digest:
            raise ValueError("Baseline encoder source changed")
        return LegacyPairModel(settings, packed, module)
    return packed, model_config, factory
