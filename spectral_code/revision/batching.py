"""PyTorch Dataset/DataLoader adapters for canonical code graphs."""

from __future__ import annotations

from .data import CanonicalGraph

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset




@dataclass
class CanonicalGraphBatch:
    canonical_ids: torch.Tensor
    lexical_ids: torch.Tensor
    source_lexical_ids: torch.Tensor
    mask: torch.Tensor
    edges: torch.Tensor

    def to(self, device: str | torch.device) -> "CanonicalGraphBatch":
        return CanonicalGraphBatch(
            self.canonical_ids.to(device, non_blocking=True),
            self.lexical_ids.to(device, non_blocking=True),
            self.source_lexical_ids.to(device, non_blocking=True),
            self.mask.to(device, non_blocking=True),
            self.edges.to(device, non_blocking=True),
        )


class CanonicalPairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, graphs: dict[str, CanonicalGraph]) -> None:
        usable = pairs[pairs["left_id"].isin(graphs) & pairs["right_id"].isin(graphs)].copy()
        self.dropped_pairs = int(len(pairs) - len(usable))
        if usable.empty:
            raise RuntimeError("No usable pairs remain after joining pair endpoints to graph records.")
        self.left_ids = usable["left_id"].astype(str).tolist()
        self.right_ids = usable["right_id"].astype(str).tolist()
        self.labels = usable["label"].to_numpy(dtype=np.float32)
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.graphs[self.left_ids[index]], self.graphs[self.right_ids[index]], self.labels[index]


def pack_graphs(graphs: Sequence[CanonicalGraph], max_nodes: int) -> CanonicalGraphBatch:
    batch_size = len(graphs)
    lexical_slots = graphs[0].lexical_ids.shape[1]
    canonical_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    lexical_ids = torch.zeros(batch_size, max_nodes, lexical_slots, dtype=torch.long)
    source_slots = graphs[0].source_lexical_ids.shape[0]
    source_lexical = torch.zeros(batch_size, source_slots, dtype=torch.long)
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    edges = torch.zeros(batch_size, 3, max_nodes, max_nodes, dtype=torch.float32)
    for batch_index, graph in enumerate(graphs):
        node_count = min(graph.node_count, max_nodes)
        canonical_ids[batch_index, :node_count] = torch.from_numpy(graph.canonical_ids[:node_count])
        lexical_ids[batch_index, :node_count] = torch.from_numpy(graph.lexical_ids[:node_count])
        source_lexical[batch_index] = torch.from_numpy(graph.source_lexical_ids)
        mask[batch_index, :node_count] = True
        for relation, relation_edges in enumerate(graph.edges):
            if not len(relation_edges):
                continue
            valid = relation_edges[(relation_edges[:, 0] < node_count) & (relation_edges[:, 1] < node_count)]
            if len(valid):
                rows = torch.from_numpy(valid[:, 0].astype(np.int64))
                cols = torch.from_numpy(valid[:, 1].astype(np.int64))
                edges[batch_index, relation, rows, cols] = 1.0
    return CanonicalGraphBatch(canonical_ids, lexical_ids, source_lexical, mask, edges)


def make_collate(max_nodes: int):
    def collate(items):
        left, right, labels = zip(*items)
        return pack_graphs(left, max_nodes), pack_graphs(right, max_nodes), torch.tensor(labels, dtype=torch.float32)

    return collate


def make_loader(
    pairs: pd.DataFrame,
    graphs: dict[str, CanonicalGraph],
    *,
    max_nodes: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = CanonicalPairDataset(pairs, graphs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=make_collate(max_nodes),
    )
