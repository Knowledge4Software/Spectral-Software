"""Self-contained Kaggle runtime for paper-faithful baseline adaptations.

This file is embedded into generated notebooks by
the standalone Kaggle notebooks. Keep imports absolute-free:
the core definitions are prepended to the notebook cell before this runtime.
"""

from __future__ import annotations

import copy
import datetime
import gc
import gzip
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    children_from_edges
except NameError:  # Direct local import; generated notebooks prepend core.py.
    from research.faithful_graph_baselines.core import (
        ASTNNEncoder,
        CDLHModel,
        DeepSimEncoder,
        FAASTGGNN,
        FAASTGMN,
        RtvNNEncoder,
        children_from_edges,
        deckard_pair_similarity,
        deckard_subtree_vectors,
        left_child_right_sibling,
        postorder_depths,
    )


METHOD_CONFIGS = {
    "deckard": {"name": "Deckard", "batch": None, "lr": None, "weight_decay": None},
    # Faithful models run on one GPU because their int16 corpus buffers cannot
    # be broadcast by NCCL.  CDLH's level-wise binary Tree-LSTM retains both
    # trees' recurrent states for backward, so it needs a substantially
    # smaller batch than the other encoders to fit safely on a 16-GB T4.
    "astnn": {"name": "ASTNN", "batch": 32, "lr": 1e-3, "weight_decay": 0.0},
    "rtvnn": {"name": "RtvNN", "batch": 32, "lr": 5e-4, "weight_decay": 1e-4},
    "cdlh": {"name": "CDLH", "batch": 16, "lr": 1e-3, "weight_decay": 0.0},
    "deepsim": {"name": "DeepSim", "batch": 32, "lr": 5e-4, "weight_decay": 1e-3},
    "fa_ast_ggnn": {"name": "FA-AST+GGNN", "batch": 8, "lr": 5e-4, "weight_decay": 1e-4},
    "fa_ast_gmn": {"name": "FA-AST+GMN", "batch": 4, "lr": 5e-4, "weight_decay": 1e-4},
}

# A generated CodeNet notebook supplies conservative values in its header.
# Other datasets retain the established defaults.
MAX_NODES = int(globals().get("MAX_NODES", 256))
MAX_EDGES = int(globals().get("MAX_EDGES", 512))
MAX_STATEMENTS = int(globals().get("MAX_STATEMENTS", 64))
# The paper uses directed flow-augmented AST relations.  Keep both directions
# explicitly: collapsing them makes parent/control/data-flow messages
# indistinguishable from their reverse relation.
RELATIONS = ("ast_forward", "ast_reverse", "cfg_forward", "cfg_reverse", "ddg_forward", "ddg_reverse")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _open_text(path: Path):
    with path.open("rb") as probe:
        packed = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8") if packed else path.open("r", encoding="utf-8")


def _kaggle_working_dir() -> Path:
    return Path(globals().get("WORK_DIR_OVERRIDE", "/kaggle/working"))


def _kaggle_input_dir() -> Path:
    return Path(globals().get("KAGGLE_INPUT_OVERRIDE", "/kaggle/input"))


def _extract_clean_artifacts_from_zips() -> list[Path]:
    """Materialize the minimum required clean-data files from Kaggle ZIP inputs.

    Kaggle datasets can be attached either as a folder or as the final clean-data
    ZIP.  The native GNN notebooks already support both layouts; the faithful
    baselines must do the same.  Extracting only pairs/graph records avoids
    unpacking unrelated large source archives.
    """
    import zipfile

    input_root = _kaggle_input_dir()
    if not input_root.exists():
        return []
    required = {
        "pairs.csv.gz", "pairs.csv", "pairs.csv.gz.tmp",
        "graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp",
    }
    cache_root = _kaggle_working_dir() / "faithful_baseline_input_cache"
    extracted_roots: list[Path] = []
    for archive_path in input_root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = [
                    info for info in archive.infolist()
                    if not info.is_dir() and Path(info.filename).name in required
                ]
                basenames = {Path(info.filename).name for info in members}
                has_pairs = any(name.startswith("pairs.csv") for name in basenames)
                has_graphs = any(name.startswith("graph_spectra.jsonl") for name in basenames)
                if not (has_pairs and has_graphs):
                    continue
                digest = hashlib.sha1(str(archive_path).encode("utf-8")).hexdigest()[:12]
                target = cache_root / f"{archive_path.stem}_{digest}"
                marker = target / ".complete"
                if not marker.exists():
                    for info in members:
                        # Reject unsafe archive member paths even though Kaggle
                        # attachments are expected to be trusted.
                        relative = Path(info.filename)
                        if relative.is_absolute() or ".." in relative.parts:
                            continue
                        destination = target / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info, "r") as source, destination.open("wb") as destination_file:
                            shutil.copyfileobj(source, destination_file)
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text("ok", encoding="utf-8")
                extracted_roots.append(target)
        except zipfile.BadZipFile:
            continue
    return extracted_roots


def _find_input(dataset_key: str, *names: str) -> Path:
    input_root = _kaggle_input_dir()
    roots = [input_root / dataset_key, input_root, Path(".")]
    matches = []
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            matches.extend(path for path in root.rglob(name) if path.is_file())
    if not matches:
        for root in _extract_clean_artifacts_from_zips():
            for name in names:
                matches.extend(path for path in root.rglob(name) if path.is_file())
    if not matches:
        visible = [str(path.relative_to(input_root)) for path in input_root.rglob("*") if path.is_file()][:80] if input_root.exists() else []
        raise FileNotFoundError(
            f"Could not find {names} below the Kaggle inputs or an attached clean-data ZIP. "
            "Attach the final clean_data folder/ZIP containing pairs.csv(.gz) and "
            "graph_spectra.jsonl(.gz). Files currently visible: " + ", ".join(visible)
        )
    normalized_key = "".join(character for character in dataset_key.lower() if character.isalnum())
    key_without_version = normalized_key.removesuffix("v3")

    def rank(path: Path) -> tuple[int, int, int, str]:
        normalized_path = "".join(character for character in str(path).lower() if character.isalnum())
        dataset_penalty = 0 if (
            normalized_key in normalized_path or key_without_version in normalized_path
        ) else 1
        return dataset_penalty, len(path.parts), len(path.name), str(path)

    return sorted(set(matches), key=rank)[0]


def _metric_dict(labels, scores, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = (np.asarray(scores) >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "P": float(precision),
        "R": float(recall),
        "F1": float(f1),
        "Acc": float(accuracy_score(labels, predictions)),
        "MacroF1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "BalancedAccuracy": float(balanced_accuracy_score(labels, predictions)),
        "ROC_AUC": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan"),
        "TP": int(((labels == 1) & (predictions == 1)).sum()),
        "FP": int(((labels == 0) & (predictions == 1)).sum()),
        "TN": int(((labels == 0) & (predictions == 0)).sum()),
        "FN": int(((labels == 1) & (predictions == 0)).sum()),
    }


def _choose_threshold(labels, scores) -> tuple[float, dict]:
    scores = np.asarray(scores, dtype=np.float32)
    candidates = np.unique(
        np.concatenate([np.linspace(0.01, 0.99, 199), np.quantile(scores, np.linspace(0, 1, 201))])
    )
    best_threshold, best_metrics = 0.5, None
    for threshold in candidates:
        metrics = _metric_dict(labels, scores, float(threshold))
        key = _selection_key(metrics, labels)
        if best_metrics is None or key > _selection_key(best_metrics, labels):
            best_threshold, best_metrics = float(threshold), metrics
    return best_threshold, best_metrics


def _validation_selection(labels) -> dict[str, int | bool | str]:
    """Select Accuracy only for a truly 50/50 usable validation split."""
    labels = np.asarray(labels, dtype=np.int64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    balanced = positives > 0 and positives == negatives
    return {
        "balanced": balanced,
        "metric": "Accuracy" if balanced else "F1",
        "positives": positives,
        "negatives": negatives,
    }


def _selection_key(metrics: dict, labels) -> tuple[float, float]:
    context = _validation_selection(labels)
    if context["balanced"]:
        return float(metrics["Acc"]), float(metrics["F1"])
    return float(metrics["F1"]), float(metrics["BalancedAccuracy"])


def _limit(frame: pd.DataFrame, split: str, maximum: int | None, seed: int) -> pd.DataFrame:
    subset = frame[frame.split == split].copy()
    if maximum is not None and len(subset) > maximum:
        subset = subset.sample(maximum, random_state=seed)
    return subset.reset_index(drop=True)


def _adjacency(layer: dict) -> dict:
    return layer.get("adjacency", {}) if isinstance(layer, dict) else {}


def _load_raw_graphs(
    path: Path, wanted: set[str], required_layers: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """Load only graph layers required by the selected baseline.

    CodeNet has more than 135K endpoints and four stored graph views per
    endpoint. Keeping unused views as Python objects can exhaust Kaggle host
    RAM before a model is constructed.
    """
    allowed = set(required_layers) if required_layers is not None else None
    graphs = {}
    with _open_text(path) as source:
        for line in tqdm(source, desc="Loading exported program graphs", unit="code"):
            if not line.strip():
                continue
            row = json.loads(line)
            code_id = str(row.get("code_id"))
            if code_id in wanted:
                graphs[code_id] = {
                    key: _adjacency(value)
                    for key, value in row.get("graphs", {}).items()
                    if allowed is None or key in allowed
                }
    return graphs


def _load_deckard_vectors_streaming(
    path: Path, code_ids: list[str], train_ids: set[str],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], set[str]]:
    """Build DECKARD subtree vectors in two streaming passes.

    DECKARD keeps only 32 small uint8 vectors per endpoint, but the shared
    loader first materializes every exported AST as decoded Python objects.
    At CodeNet's scale that intermediate alone can exceed Kaggle host RAM, so
    the run dies or thrashes long before scoring.  Reading each line once and
    reducing it immediately keeps the peak at the retained vectors.
    """
    wanted = set(code_ids)
    frequency: dict[str, int] = {}
    with _open_text(path) as source:
        for line in tqdm(source, desc="DECKARD vocabulary pass", unit="code", leave=False):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("code_id")) not in train_ids:
                continue
            adjacency = _adjacency(row.get("graphs", {}).get("ast"))
            for node_type in adjacency.get("node_types", [])[:MAX_NODES]:
                value = str(node_type)
                frequency[value] = frequency.get(value, 0) + 1
    ordered = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    vocab = {"<pad>": 0, "<unk>": 1, **{value: index + 2 for index, (value, _) in enumerate(ordered)}}

    vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    usable: set[str] = set()
    with _open_text(path) as source:
        for line in tqdm(source, desc="DECKARD subtree vectors", unit="code"):
            if not line.strip():
                continue
            row = json.loads(line)
            code_id = str(row.get("code_id"))
            if code_id not in wanted:
                continue
            adjacency = _adjacency(row.get("graphs", {}).get("ast"))
            if not adjacency.get("node_types"):
                continue
            types, *_, children = _tree_arrays(adjacency, vocab)
            n = int((types != 0).sum())
            subtree_vectors, sizes = deckard_subtree_vectors(
                types[:n], children, len(vocab), min_nodes=3,
            )
            order = np.argsort(sizes)[-32:]
            # Category counts and subtree sizes are bounded by MAX_NODES, so
            # uint8 is lossless; deckard_pair_similarity promotes to float32.
            vectors[code_id] = (
                subtree_vectors[order].astype(np.uint8, copy=False),
                sizes[order].astype(np.uint8, copy=False),
            )
            usable.add(code_id)
    return vectors, usable


_TREE_ARRAY_SPEC = (
    ("node_types", MAX_NODES, np.int32),
    ("edge_parents", MAX_EDGES, np.int16),
    ("edge_children", MAX_EDGES, np.int16),
    ("depths", MAX_NODES, np.int16),
    ("binary_depths", MAX_NODES, np.int16),
    ("left_child", MAX_NODES, np.int16),
    ("right_sibling", MAX_NODES, np.int16),
    ("statements", MAX_STATEMENTS, np.int16),
)


def _load_tree_arrays_streaming(
    path: Path, code_ids: list[str], train_ids: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, int], set[str], dict[str, int]]:
    """Pack ASTNN/RtvNN/CDLH tensors without retaining decoded ASTs.

    These baselines need one AST view per endpoint, but the shared loader
    decodes every exported graph into Python objects and holds them all before
    packing.  At CodeNet's endpoint count that peak can exceed Kaggle host RAM
    even though the packed arrays themselves are modest.
    """
    wanted = set(code_ids)
    frequency: dict[str, int] = {}
    with _open_text(path) as source:
        for line in tqdm(source, desc="AST vocabulary pass", unit="code", leave=False):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("code_id")) not in train_ids:
                continue
            adjacency = _adjacency(row.get("graphs", {}).get("ast"))
            for node_type in adjacency.get("node_types", [])[:MAX_NODES]:
                value = str(node_type)
                frequency[value] = frequency.get(value, 0) + 1
    ordered = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    vocab = {"<pad>": 0, "<unk>": 1, **{value: index + 2 for index, (value, _) in enumerate(ordered)}}

    id_to_row = {code_id: index for index, code_id in enumerate(code_ids)}
    count = len(code_ids)
    arrays = {
        name: np.zeros((count, width), dtype=dtype)
        for name, width, dtype in _TREE_ARRAY_SPEC
    }
    for name in ("edge_parents", "edge_children", "left_child", "right_sibling", "statements"):
        arrays[name].fill(-1)
    usable: set[str] = set()
    with _open_text(path) as source:
        for line in tqdm(source, desc="Packing ASTs", unit="code"):
            if not line.strip():
                continue
            row = json.loads(line)
            code_id = str(row.get("code_id"))
            target = id_to_row.get(code_id)
            if target is None:
                continue
            adjacency = _adjacency(row.get("graphs", {}).get("ast"))
            if not adjacency.get("node_types"):
                continue
            packed = _tree_arrays(adjacency, vocab)[:8]
            for (name, _width, _dtype), value in zip(_TREE_ARRAY_SPEC, packed):
                arrays[name][target] = value
            usable.add(code_id)
    return arrays, vocab, usable, id_to_row


def _load_fa_arrays_streaming(
    path: Path, code_ids: list[str], train_ids: set[str], required_layers: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, int], set[str], dict[str, int]]:
    """Build FA-AST tensors in two streaming passes without retaining raw JSON.

    FA-AST needs four graph views per CodeNet endpoint.  Holding their decoded
    Python dictionaries before packing can exceed Kaggle host RAM.  Its type
    vocabulary is the only dependency between the two passes, so first count
    training CPG types and then fill the final compact arrays directly.
    """
    wanted = set(code_ids)
    frequency: dict[str, int] = {}
    with _open_text(path) as source:
        for line in tqdm(source, desc="FA-AST vocabulary pass", unit="code", leave=False):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("code_id")) not in train_ids:
                continue
            cpg = _adjacency(row.get("graphs", {}).get("cpg"))
            for node_type in cpg.get("node_types", [])[:MAX_NODES]:
                value = str(node_type)
                frequency[value] = frequency.get(value, 0) + 1
    ordered = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    vocab = {"<pad>": 0, "<unk>": 1, **{value: index + 2 for index, (value, _) in enumerate(ordered)}}
    id_to_row = {code_id: index for index, code_id in enumerate(code_ids)}
    count = len(code_ids)
    arrays = {
        "node_types": np.zeros((count, MAX_NODES), dtype=np.int32),
        "edge_src": np.full((count, len(RELATIONS), MAX_EDGES), -1, dtype=np.int16),
        "edge_dst": np.full((count, len(RELATIONS), MAX_EDGES), -1, dtype=np.int16),
    }
    usable: set[str] = set()
    with _open_text(path) as source:
        for line in tqdm(source, desc="FA-AST compact graph pass", unit="code"):
            if not line.strip():
                continue
            row = json.loads(line)
            code_id = str(row.get("code_id"))
            target = id_to_row.get(code_id)
            if target is None:
                continue
            graphs = {
                name: _adjacency(value)
                for name, value in row.get("graphs", {}).items()
                if name in required_layers
            }
            if not all(graphs.get(name, {}).get("node_types") for name in required_layers):
                continue
            node_types, edge_src, edge_dst = _flow_arrays(graphs, vocab)
            arrays["node_types"][target] = node_types
            arrays["edge_src"][target] = edge_src
            arrays["edge_dst"][target] = edge_dst
            usable.add(code_id)
    return arrays, vocab, usable, id_to_row


def _training_type_vocabulary(raw: dict[str, dict], train_ids: set[str], graph_view: str) -> dict[str, int]:
    frequency: dict[str, int] = {}
    for code_id in train_ids:
        adjacency = raw.get(code_id, {}).get(graph_view, {})
        for node_type in adjacency.get("node_types", [])[:MAX_NODES]:
            value = str(node_type)
            frequency[value] = frequency.get(value, 0) + 1
    ordered = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    return {"<pad>": 0, "<unk>": 1, **{value: index + 2 for index, (value, _) in enumerate(ordered)}}


def _statement_roots(types: list[str], children: list[list[int]]) -> list[int]:
    indegree = [0] * len(types)
    for child_list in children:
        for child in child_list:
            indegree[child] += 1
    roots = [index for index, degree in enumerate(indegree) if degree == 0]
    methods = [index for index in roots if types[index].upper() == "METHOD"] or roots[:1]
    statements = []
    for method in methods:
        direct = children[method]
        blocks = [node for node in direct if types[node].upper() == "BLOCK"]
        for parent in blocks or [method]:
            statements.extend(children[parent])
    return list(dict.fromkeys(statements or methods or [0]))[:MAX_STATEMENTS]


def _tree_arrays(adjacency: dict, vocab: dict[str, int]):
    raw_types = [str(value) for value in adjacency.get("node_types", [])]
    n = min(int(adjacency.get("num_nodes", 0) or 0), len(raw_types), MAX_NODES)
    rows, cols = [], []
    for parent, child in zip(adjacency.get("row", []), adjacency.get("col", [])):
        parent, child = int(parent), int(child)
        if parent < n and child < n and len(rows) < MAX_EDGES:
            rows.append(parent)
            cols.append(child)
    children = children_from_edges(n, rows, cols)
    depth = postorder_depths(children)
    left, right = left_child_right_sibling(children)
    binary_children = [list(filter(lambda value: value >= 0, (int(left[index]), int(right[index])))) for index in range(n)]
    binary_depth = postorder_depths(binary_children)
    types = np.zeros(MAX_NODES, dtype=np.int32)
    types[:n] = [vocab.get(value, 1) for value in raw_types[:n]]
    parents = np.full(MAX_EDGES, -1, dtype=np.int16)
    child_array = np.full(MAX_EDGES, -1, dtype=np.int16)
    parents[:len(rows)], child_array[:len(cols)] = rows, cols
    depths = np.zeros(MAX_NODES, dtype=np.int16)
    binary_depths = np.zeros(MAX_NODES, dtype=np.int16)
    left_array = np.full(MAX_NODES, -1, dtype=np.int16)
    right_array = np.full(MAX_NODES, -1, dtype=np.int16)
    depths[:n], binary_depths[:n], left_array[:n], right_array[:n] = depth, binary_depth, left, right
    statements = np.full(MAX_STATEMENTS, -1, dtype=np.int16)
    roots = _statement_roots(raw_types[:n], children) if n else []
    statements[:len(roots)] = roots
    return types, parents, child_array, depths, binary_depths, left_array, right_array, statements, children


def _map_relation(layer: dict, universe_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lookup = {str(node_id): index for index, node_id in enumerate(universe_ids)}
    layer_ids = [str(value) for value in layer.get("node_ids", [])]
    source = np.full(MAX_EDGES, -1, dtype=np.int16)
    target = np.full(MAX_EDGES, -1, dtype=np.int16)
    edges = []
    for row, col in zip(layer.get("row", []), layer.get("col", [])):
        row, col = int(row), int(col)
        if row < len(layer_ids) and col < len(layer_ids):
            u, v = lookup.get(layer_ids[row]), lookup.get(layer_ids[col])
            if u is not None and v is not None and u < MAX_NODES and v < MAX_NODES:
                edges.append((u, v))
                if len(edges) >= MAX_EDGES:
                    break
    if edges:
        source[:len(edges)], target[:len(edges)] = zip(*edges)
    return source, target


def _flow_universe(graphs: dict, cpg: dict) -> list[int]:
    """Rank CPG node positions by how many relation edges they carry.

    A graph larger than MAX_NODES must still be represented by a connected
    subgraph.  Keeping an arbitrary node prefix instead leaves nearly every
    edge with an endpoint outside the retained window, because each exported
    layer orders its nodes independently.  The surviving graph is then
    edgeless, message passing degenerates to a constant, and the model cannot
    train at all.
    """
    node_ids = [str(value) for value in cpg.get("node_ids", [])]
    if len(node_ids) <= MAX_NODES:
        return list(range(len(node_ids)))
    position = {node_id: index for index, node_id in enumerate(node_ids)}
    degree = np.zeros(len(node_ids), dtype=np.int64)
    neighbours: dict[int, list[int]] = {}
    for name in ("ast", "cfg", "ddg"):
        layer = graphs.get(name, {})
        layer_ids = [str(value) for value in layer.get("node_ids", [])]
        for row, col in zip(layer.get("row", []), layer.get("col", [])):
            row, col = int(row), int(col)
            if row >= len(layer_ids) or col >= len(layer_ids):
                continue
            source = position.get(layer_ids[row])
            target = position.get(layer_ids[col])
            if source is None or target is None or source == target:
                continue
            degree[source] += 1
            degree[target] += 1
            neighbours.setdefault(source, []).append(target)
            neighbours.setdefault(target, []).append(source)
    # Grow a connected region outward from the busiest node.  Selecting the
    # globally highest-degree nodes alone tends to pick mutually distant hubs,
    # whose connecting edges all fall outside the window.
    order = np.lexsort((np.arange(len(node_ids)), -degree))
    selected: set[int] = set()
    for seed in order:
        if len(selected) >= MAX_NODES:
            break
        seed = int(seed)
        if seed in selected:
            continue
        queue, selected = [seed], selected | {seed}
        while queue and len(selected) < MAX_NODES:
            current = queue.pop(0)
            ranked_neighbours = sorted(
                set(neighbours.get(current, ())), key=lambda node: (-degree[node], node)
            )
            for node in ranked_neighbours:
                if len(selected) >= MAX_NODES:
                    break
                if node not in selected:
                    selected.add(node)
                    queue.append(node)
    return sorted(selected)


def _flow_arrays(graphs: dict, vocab: dict[str, int]):
    cpg = graphs.get("cpg") or graphs.get("ast") or {}
    raw_ids = [str(value) for value in cpg.get("node_ids", [])]
    raw_types = [str(value) for value in cpg.get("node_types", [])]
    selected = _flow_universe(graphs, cpg)
    universe_ids = [raw_ids[index] for index in selected]
    universe_types = [raw_types[index] for index in selected if index < len(raw_types)]
    types = np.zeros(MAX_NODES, dtype=np.int32)
    types[:len(universe_types)] = [vocab.get(value, 1) for value in universe_types]
    sources = np.full((len(RELATIONS), MAX_EDGES), -1, dtype=np.int16)
    targets = np.full_like(sources, -1)
    for relation, (name, reverse) in enumerate((
        ("ast", False), ("ast", True),
        ("cfg", False), ("cfg", True),
        ("ddg", False), ("ddg", True),
    )):
        src, dst = _map_relation(graphs.get(name, {}), universe_ids)
        sources[relation], targets[relation] = (dst, src) if reverse else (src, dst)
    return types, sources, targets


def _deepsim_arrays(graphs: dict, vocab: dict[str, int]):
    cfg = graphs.get("cfg", {})
    cfg_ids = [str(value) for value in cfg.get("node_ids", [])][:MAX_NODES]
    cfg_types = [str(value) for value in cfg.get("node_types", [])][:len(cfg_ids)]
    types = np.zeros(MAX_NODES, dtype=np.int32)
    types[:len(cfg_types)] = [vocab.get(value, 1) for value in cfg_types]
    src, dst = _map_relation(cfg, cfg_ids)
    ddg_src, ddg_dst = _map_relation(graphs.get("ddg", {}), cfg_ids)
    # [CFG in/out/degree/branch/active/position,
    #  DDG in/out/degree/source/sink/active].  This is the portable analogue
    # of the original DeepSim control/data-flow semantic matrix.
    numeric = np.zeros((MAX_NODES, 12), dtype=np.float32)
    for u, v in zip(src[src >= 0], dst[dst >= 0]):
        numeric[int(u), 0] += 1
        numeric[int(v), 1] += 1
    numeric[:, 2] = numeric[:, 0] + numeric[:, 1]
    numeric[:, 3] = (numeric[:, 0] > 1).astype(np.float32)
    numeric[:, 4] = (numeric[:, 2] > 0).astype(np.float32)
    if cfg_ids:
        numeric[:len(cfg_ids), 5] = np.linspace(0, 1, len(cfg_ids), dtype=np.float32)
    for u, v in zip(ddg_src[ddg_src >= 0], ddg_dst[ddg_dst >= 0]):
        numeric[int(u), 6] += 1
        numeric[int(v), 7] += 1
    numeric[:, 8] = numeric[:, 6] + numeric[:, 7]
    numeric[:, 9] = (numeric[:, 6] > 0).astype(np.float32)
    numeric[:, 10] = (numeric[:, 7] > 0).astype(np.float32)
    numeric[:, 11] = (numeric[:, 8] > 0).astype(np.float32)
    numeric[:, (0, 1, 2, 6, 7, 8)] = np.log1p(numeric[:, (0, 1, 2, 6, 7, 8)])
    return types, numeric, src, dst


class _PairRows(Dataset):
    def __init__(self, frame: pd.DataFrame, id_to_row: dict[str, int]):
        self.left = frame.left_id.map(id_to_row).to_numpy(np.int64)
        self.right = frame.right_id.map(id_to_row).to_numpy(np.int64)
        self.labels = frame.label.to_numpy(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.left[index], self.right[index], self.labels[index]


class _DepthBucketBatchSampler:
    """Randomize batches while keeping similarly deep CDLH trees together.

    A Tree-LSTM batch executes up to its deepest tree level.  Ordinary random
    batching therefore makes a single deep AST force every shallow pair in the
    batch through extra padded levels.  This sampler retains every row and the
    configured logical batch size, randomizes both bucket order and row order
    each epoch, and only changes batch composition to reduce that wasted work.
    """

    def __init__(self, pair_depths: np.ndarray, batch_size: int, seed: int) -> None:
        self.order = np.argsort(np.asarray(pair_depths), kind="stable")
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return int(np.ceil(len(self.order) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        buckets = [
            self.order[start:start + self.batch_size]
            for start in range(0, len(self.order), self.batch_size)
        ]
        for bucket_index in rng.permutation(len(buckets)):
            # Preserve depth locality while avoiding a fixed order inside ties.
            yield rng.permutation(buckets[int(bucket_index)]).tolist()


def _pair_tree_depths(frame: pd.DataFrame, id_to_row: dict[str, int], tree_depths: np.ndarray) -> np.ndarray:
    left = frame.left_id.map(id_to_row).to_numpy(np.int64)
    right = frame.right_id.map(id_to_row).to_numpy(np.int64)
    return np.maximum(tree_depths[left], tree_depths[right])


def _sort_pairs_by_tree_depth(frame: pd.DataFrame, id_to_row: dict[str, int], tree_depths: np.ndarray) -> pd.DataFrame:
    """Reorder evaluation rows only; labels/languages remain aligned with scores."""
    order = np.argsort(_pair_tree_depths(frame, id_to_row, tree_depths), kind="stable")
    return frame.iloc[order].reset_index(drop=True)


def _pair_row_indices(frame: pd.DataFrame, id_to_row: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame.left_id.map(id_to_row).to_numpy(np.int64),
        frame.right_id.map(id_to_row).to_numpy(np.int64),
    )


def _endpoint_anchors(left_rows: np.ndarray, right_rows: np.ndarray) -> np.ndarray:
    """Choose the more frequently occurring endpoint as each pair's anchor."""
    counts = np.bincount(np.concatenate((left_rows, right_rows)))
    return np.where(counts[left_rows] >= counts[right_rows], left_rows, right_rows)


class _EndpointGroupedDepthBatchSampler:
    """Full-data CDLH batches with a repeated endpoint and comparable depth.

    The model deduplicates endpoint rows *within* each batch.  Keeping pairs
    with their most frequent endpoint together therefore avoids repeated
    Tree-LSTM calls while preserving every pair exactly once per epoch.
    """

    def __init__(
        self, left_rows: np.ndarray, right_rows: np.ndarray, pair_depths: np.ndarray,
        batch_size: int, seed: int,
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.epoch = 0
        anchors = _endpoint_anchors(left_rows, right_rows)
        # Stable depth sorting within an anchor keeps the max Tree-LSTM level
        # of each logical batch low in addition to enabling endpoint reuse.
        order = np.lexsort((np.asarray(pair_depths), anchors))
        sorted_anchors = anchors[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_anchors)) + 1, len(order)]
        self.groups = [order[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        self.row_count = len(order)

    def __len__(self) -> int:
        return int(np.ceil(self.row_count / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        full_batches: list[np.ndarray] = []
        remainder: list[np.ndarray] = []
        for group_index in rng.permutation(len(self.groups)):
            group = self.groups[int(group_index)]
            complete = len(group) - (len(group) % self.batch_size)
            for start in range(0, complete, self.batch_size):
                full_batches.append(group[start:start + self.batch_size])
            if complete < len(group):
                remainder.append(group[complete:])
        for batch_index in rng.permutation(len(full_batches)):
            yield rng.permutation(full_batches[int(batch_index)]).tolist()
        if remainder:
            tail = rng.permutation(np.concatenate(remainder))
            for start in range(0, len(tail), self.batch_size):
                yield tail[start:start + self.batch_size].tolist()


def _sort_pairs_by_endpoint_and_depth(
    frame: pd.DataFrame, id_to_row: dict[str, int], tree_depths: np.ndarray,
) -> pd.DataFrame:
    """Evaluation ordering that maximizes exact within-batch CDLH reuse."""
    left_rows, right_rows = _pair_row_indices(frame, id_to_row)
    order = np.lexsort((np.asarray(_pair_tree_depths(frame, id_to_row, tree_depths)), _endpoint_anchors(left_rows, right_rows)))
    return frame.iloc[order].reset_index(drop=True)


class _PairModel(nn.Module):
    def __init__(self, method: str, arrays: dict[str, np.ndarray], vocab_size: int):
        super().__init__()
        self.method = method
        for name, value in arrays.items():
            self.register_buffer(name, torch.from_numpy(value), persistent=False)
        if method == "astnn":
            self.encoder = ASTNNEncoder(vocab_size)
            code_dim = 200
        elif method == "rtvnn":
            self.encoder = RtvNNEncoder(vocab_size)
            code_dim = 192
        elif method == "cdlh":
            self.encoder = CDLHModel(vocab_size)
            code_dim = 64
        elif method == "deepsim":
            self.encoder = DeepSimEncoder(vocab_size)
            code_dim = 192
        elif method == "fa_ast_ggnn":
            self.encoder = FAASTGGNN(vocab_size)
            code_dim = 192
        elif method == "fa_ast_gmn":
            self.encoder = FAASTGMN(vocab_size)
            code_dim = 192
        else:
            raise ValueError(method)
        if method == "astnn":
            # The official clone model applies one symmetric linear decision
            # layer to the absolute difference between code vectors.
            self.classifier = nn.Linear(code_dim, 1)
        elif method != "cdlh":
            self.classifier = nn.Sequential(
                nn.Linear(code_dim * 2 + 2, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 1)
            )

    def _tree(self, rows: Tensor):
        args = (
            self.node_types[rows].long(), self.edge_parents[rows].long(),
            self.edge_children[rows].long(), self.depths[rows].long()
        )
        if self.method == "astnn":
            return self.encoder(*args, self.statements[rows].long()), args[0].new_zeros((), dtype=torch.float32)
        if self.method == "rtvnn":
            return self.encoder(*args)
        return self.encoder.encode(
            args[0], self.left_child[rows].long(), self.right_sibling[rows].long(),
            self.binary_depths[rows].long()
        ), args[0].new_zeros((), dtype=torch.float32)

    def _single(self, rows: Tensor):
        if self.method in {"astnn", "rtvnn", "cdlh"}:
            return self._tree(rows)
        if self.method == "deepsim":
            return self.encoder(
                self.node_types[rows].long(), self.numeric[rows].float(),
                self.edge_src[rows].long(), self.edge_dst[rows].long()
            ), self.node_types[rows].new_zeros((), dtype=torch.float32)
        return self.encoder(
            self.node_types[rows].long(), self.edge_src[rows].long(), self.edge_dst[rows].long()
        ), self.node_types[rows].new_zeros((), dtype=torch.float32)

    def _flow(self, rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the exact non-padding envelope for a flow-graph batch.

        Packing uses fixed 128-node / 512-edge tensors so the corpus can live
        in one compact buffer.  Running GGNN/GMN over their trailing padding
        is mathematically unnecessary: padded nodes are masked and padded
        edges are invalid.  Cropping therefore preserves the baseline's
        outputs while substantially reducing CodeNet's message passing and
        GMN attention work.
        """
        node_types = self.node_types[rows].long()
        edge_src = self.edge_src[rows].long()
        edge_dst = self.edge_dst[rows].long()
        nodes = max(1, int(node_types.ne(0).sum(1).max().item()))
        valid_edges = edge_src.ge(0) & edge_dst.ge(0)
        # Crop to the last occupied slot, not to the largest edge *count*.
        # Relations are packed in discovery order, so a sparse relation can
        # hold its edges beyond a denser relation's count; cropping by count
        # silently dropped those edges and left whole relations empty.
        if valid_edges.any():
            positions = valid_edges.nonzero()[:, -1]
            edges = int(positions.max().item()) + 1
        else:
            edges = 1
        return node_types[:, :nodes], edge_src[:, :, :edges], edge_dst[:, :, :edges]

    def forward(self, left_rows: Tensor, right_rows: Tensor):
        if self.method == "fa_ast_gmn":
            left_types, left_src, left_dst = self._flow(left_rows)
            right_types, right_src, right_dst = self._flow(right_rows)
            left, right = self.encoder.forward_pair(
                left_types, left_src, left_dst, right_types, right_src, right_dst,
            )
            auxiliary = left.new_zeros(())
        elif self.method == "cdlh":
            # CDLH has no stochastic encoder layer.  If a code occurs in more
            # than one pair of the logical batch, one Tree-LSTM evaluation and
            # multiple indexed uses have exactly the same loss and accumulated
            # gradient as evaluating that identical tree repeatedly.  Grouped
            # full-data batches below make this reuse substantial on AtCoder
            # and XGLUE without caching across optimizer steps.
            rows = torch.cat((left_rows, right_rows), dim=0)
            unique_rows, inverse = torch.unique(rows, sorted=False, return_inverse=True)
            unique_codes, auxiliary = self._single(unique_rows)
            left = unique_codes.index_select(0, inverse[:len(left_rows)])
            right = unique_codes.index_select(0, inverse[len(left_rows):])
        else:
            if self.method == "fa_ast_ggnn":
                left_types, left_src, left_dst = self._flow(left_rows)
                right_types, right_src, right_dst = self._flow(right_rows)
                left = self.encoder(left_types, left_src, left_dst)
                right = self.encoder(right_types, right_src, right_dst)
                left_aux = right_aux = left.new_zeros(())
            else:
                left, left_aux = self._single(left_rows)
                right, right_aux = self._single(right_rows)
            auxiliary = left_aux + right_aux
        if self.method == "cdlh":
            logits = 4.0 * (left * right).mean(-1)
        elif self.method == "astnn":
            logits = self.classifier(torch.abs(left - right)).squeeze(-1)
        else:
            features = torch.cat([
                torch.abs(left - right), left * right,
                (left * right).sum(-1, keepdim=True), torch.norm(left - right, dim=-1, keepdim=True)
            ], -1)
            logits = self.classifier(features).squeeze(-1)
        return logits, auxiliary, left, right


def _pack_method(method: str, raw: dict[str, dict], code_ids: list[str], train_ids: set[str]):
    graph_view = "cfg" if method == "deepsim" else "cpg" if method.startswith("fa_ast") else "ast"
    vocab = _training_type_vocabulary(raw, train_ids, graph_view)
    if method in {"astnn", "rtvnn", "cdlh"}:
        names = (
            "node_types", "edge_parents", "edge_children", "depths", "binary_depths",
            "left_child", "right_sibling", "statements"
        )
        rows = [_tree_arrays(raw[code_id]["ast"], vocab)[:8] for code_id in tqdm(code_ids, desc=f"Packing {method} ASTs")]
    elif method == "deepsim":
        names = ("node_types", "numeric", "edge_src", "edge_dst")
        rows = [_deepsim_arrays(raw[code_id], vocab) for code_id in tqdm(code_ids, desc="Packing CFG semantic matrices")]
    else:
        names = ("node_types", "edge_src", "edge_dst")
        rows = [_flow_arrays(raw[code_id], vocab) for code_id in tqdm(code_ids, desc="Packing typed FA-AST relations")]
    arrays = {name: np.stack([row[index] for row in rows]) for index, name in enumerate(names)}
    return arrays, vocab


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


@torch.no_grad()
def _predict_rows_adaptive(model, left, right, device):
    """Predict one batch, bisecting only a batch that does not fit the GPU."""
    try:
        logits, _, _, _ = model(left.to(device), right.to(device))
        return torch.sigmoid(logits).float().cpu()
    except RuntimeError as error:
        if not _is_cuda_oom(error) or len(left) <= 1:
            raise
        del error
        gc.collect()
        torch.cuda.empty_cache()
        middle = len(left) // 2
        return torch.cat((
            _predict_rows_adaptive(model, left[:middle], right[:middle], device),
            _predict_rows_adaptive(model, left[middle:], right[middle:], device),
        ))


@torch.no_grad()
def _predict(model, frame, id_to_row, batch_size, device):
    model.eval()
    output = []
    loader = DataLoader(_PairRows(frame, id_to_row), batch_size=batch_size, shuffle=False, num_workers=0)
    for left, right, _ in tqdm(loader, desc="Predict", leave=False):
        output.append(_predict_rows_adaptive(model, left, right, device).numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def _adaptive_train_step(
    model, method, optimizer, scaler, left, right, labels, device,
    maximum_microbatch=None,
):
    """Run a logical batch and halve only its microbatch after a CUDA OOM.

    Gradient weighting preserves the same mean loss as the unsplit logical
    batch.  Most batches therefore retain CDLH's fast batch size of 16, while
    an unusually deep AST can safely fall back to 8, 4, 2, or 1.
    """
    logical_rows = len(labels)
    microbatch = min(logical_rows, maximum_microbatch or logical_rows)
    while True:
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        chunk_left = chunk_right = chunk_labels = None
        logits = auxiliary = left_code = right_code = None
        chunk_loss = scaled_loss = None
        try:
            for start in range(0, logical_rows, microbatch):
                stop = min(logical_rows, start + microbatch)
                chunk_left = left[start:stop].to(device)
                chunk_right = right[start:stop].to(device)
                chunk_labels = labels[start:stop].to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits, auxiliary, left_code, right_code = model(chunk_left, chunk_right)
                    auxiliary = auxiliary.mean()
                    if method == "cdlh":
                        chunk_loss = CDLHModel.loss(left_code, right_code, chunk_labels)
                    else:
                        chunk_loss = nn.functional.binary_cross_entropy_with_logits(logits, chunk_labels)
                    if method == "rtvnn":
                        chunk_loss = chunk_loss + 0.05 * auxiliary
                    weight = len(chunk_labels) / logical_rows
                    scaled_loss = chunk_loss * weight
                scaler.scale(scaled_loss).backward()
                weighted_loss += float(chunk_loss.detach().cpu()) * len(chunk_labels)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            return weighted_loss, microbatch
        except RuntimeError as error:
            if not _is_cuda_oom(error) or microbatch <= 1:
                raise
            previous = microbatch
            microbatch = max(1, microbatch // 2)
            optimizer.zero_grad(set_to_none=True)
            chunk_left = chunk_right = chunk_labels = None
            logits = auxiliary = left_code = right_code = None
            chunk_loss = scaled_loss = None
            del error
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[CUDA OOM recovery] retrying logical batch with microbatch {previous} -> {microbatch}")


def _gpu_execution_metadata() -> dict[str, object]:
    """Describe the actual accelerator layout in every result manifest."""
    if not torch.cuda.is_available():
        return {
            "device": "cpu", "available_gpu_count": 0, "used_gpu_count": 0,
            "data_parallel": False, "gpus": [],
        }
    count = torch.cuda.device_count()
    return {
        "device": "cuda:0",
        "available_gpu_count": count,
        "used_gpu_count": 1,
        "data_parallel": False,
        "parallelism_note": (
            "Faithful graph baselines keep corpus tensors resident on one GPU; "
            "per-batch DataParallel replication would dominate runtime."
        ),
        "gpus": [torch.cuda.get_device_name(index) for index in range(count)],
    }


def _run_deckard(raw, train, valid, test, id_to_row, train_ids, vectors=None):
    if vectors is None:
        vocab = _training_type_vocabulary(raw, train_ids, "ast")
        vectors = {}
        total_codes = len(id_to_row)
        print(f"[Deckard] computing subtree vectors for {total_codes:,} code endpoints on CPU...", flush=True)
        for index, code_id in enumerate(tqdm(id_to_row, desc="DECKARD subtree vectors"), start=1):
            # The raw JSON graph is needed exactly once. Free it immediately:
            # CodeNet has >134k endpoints, and retaining raw graphs alongside every
            # code's subtree vectors can exceed Kaggle host memory before scoring.
            graph = raw.pop(code_id)
            types, *_, children = _tree_arrays(graph["ast"], vocab)
            n = int((types != 0).sum())
            subtree_vectors, sizes = deckard_subtree_vectors(types[:n], children, len(vocab), min_nodes=3)
            order = np.argsort(sizes)[-32:]
            # Category counts and subtree sizes are bounded by MAX_NODES (128 for
            # CodeNet), so uint8 is lossless. Cosine math promotes each retained
            # pair back to float32 in deckard_pair_similarity.
            vectors[code_id] = (
                subtree_vectors[order].astype(np.uint8, copy=False),
                sizes[order].astype(np.uint8, copy=False),
            )
            if index % 10_000 == 0 or index == total_codes:
                print(f"[Deckard] subtree vectors: {index:,}/{total_codes:,}", flush=True)
        del raw
        gc.collect()

    def scores(frame):
        print(f"[Deckard] scoring {len(frame):,} pairs...", flush=True)
        return np.asarray([
            deckard_pair_similarity(*vectors[left], *vectors[right])
            for left, right in tqdm(
                zip(frame.left_id, frame.right_id), total=len(frame),
                desc="DECKARD pair scoring", leave=False,
            )
        ], dtype=np.float32)

    valid_scores = scores(valid)
    threshold, valid_metrics = _choose_threshold(valid.label, valid_scores)
    return threshold, valid_metrics, scores(test), 0, [], None


def run_faithful_baseline(dataset_key: str, method: str) -> dict:
    if method not in METHOD_CONFIGS:
        raise ValueError(f"Unknown method {method!r}")
    config = dict(METHOD_CONFIGS[method])
    codenet = dataset_key.lower().replace("_", "-") == "codenet-4l"
    if codenet:
        # CodeNet has far more graph endpoints. CDLH evaluates two checkpointed
        # Tree-LSTMs per pair, so these logical batches are safe on one 16-GB T4.
        codenet_batches = {
            "astnn": 16, "rtvnn": 16, "cdlh": 4, "deepsim": 16,
            "fa_ast_ggnn": 4, "fa_ast_gmn": 2,
        }
        if method in codenet_batches:
            config["batch"] = min(int(config["batch"]), codenet_batches[method])
    # Older generated notebook headers contained only pair caps.  Keep the
    # shared runtime backward compatible while making the effective schedule
    # explicit in every output manifest.
    run_config = dict(globals().get("RUN_CONFIG", {}))
    max_train_pairs = run_config.get("max_train_pairs")
    max_valid_pairs = run_config.get("max_valid_pairs")
    max_test_pairs = run_config.get("max_test_pairs")
    epochs = max(1, int(run_config.get("epochs", 4)))
    patience = max(1, int(run_config.get("patience", 2)))
    run_profile = str(globals().get("RUN_PROFILE", "final_full"))
    seed = 42
    _seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    started = time.perf_counter()
    print(f"[{config['name']}] resolving input artifacts for {dataset_key}...", flush=True)
    pairs_path = _find_input(dataset_key, "pairs.csv.gz", "pairs.csv", "pairs.csv.gz.tmp")
    graphs_path = _find_input(
        dataset_key,
        "graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp",
    )
    # Detect compression from file bytes, not the extension: Kaggle sometimes
    # materializes a gzip payload as a `.tmp` file.
    with _open_text(pairs_path) as pairs_file:
        pairs = pd.read_csv(pairs_file, dtype={"left_id": str, "right_id": str, "split": str, "label": np.int64})
    print(f"[{config['name']}] loaded {len(pairs):,} benchmark pairs; selecting configured splits...", flush=True)
    # The CodeNet main-table release is deliberately a 50k-clone / 50k-
    # different-problem-nonclone benchmark.  Failing here prevents a Kaggle
    # attachment of the older 12k scope-study ZIP from silently producing a
    # non-comparable row in the CodeNet column.
    if dataset_key.lower().replace("_", "-") == "codenet-4l":
        observed = pairs.groupby("split").size().astype(int).to_dict()
        expected = {"train": 70_000, "valid": 15_000, "test": 15_000}
        if observed != expected:
            raise RuntimeError(
                "CodeNet baselines require the final clone50k_diff50k clean-data release "
                f"with split counts {expected}; attached input has {observed}."
            )
    if "gpt" in dataset_key.lower() and (pairs.left_id == pairs.right_id).any():
        raise RuntimeError("Identity/self pairs are forbidden in the rebuilt benchmark")
    train = _limit(pairs, "train", max_train_pairs, seed)
    valid = _limit(pairs, "valid", max_valid_pairs, seed + 1)
    test = _limit(pairs, "test", max_test_pairs, seed + 2)
    code_ids = sorted(set(train.left_id) | set(train.right_id) | set(valid.left_id) | set(valid.right_id) | set(test.left_id) | set(test.right_id))
    train_ids = set(train.left_id) | set(train.right_id)
    required = ("cfg", "ddg") if method == "deepsim" else ("ast", "cfg", "ddg", "cpg") if method.startswith("fa_ast") else ("ast",)
    print(
        f"[{config['name']}] loading {required} graph layer(s) for {len(code_ids):,} code endpoints...",
        flush=True,
    )
    prepacked_arrays = None
    prepacked_vocab = None
    prepacked_id_to_row = None
    deckard_vectors = None
    if method == "deckard":
        print(f"[{config['name']}] streaming ASTs directly into subtree vectors...", flush=True)
        deckard_vectors, usable = _load_deckard_vectors_streaming(
            graphs_path, code_ids, train_ids,
        )
        raw = None
    elif method.startswith("fa_ast"):
        print(f"[{config['name']}] streaming FA-AST graphs directly into compact arrays...", flush=True)
        prepacked_arrays, prepacked_vocab, usable, prepacked_id_to_row = _load_fa_arrays_streaming(
            graphs_path, code_ids, train_ids, required,
        )
        raw = None
    elif method in {"astnn", "rtvnn", "cdlh"}:
        print(f"[{config['name']}] streaming ASTs directly into compact arrays...", flush=True)
        prepacked_arrays, prepacked_vocab, usable, prepacked_id_to_row = _load_tree_arrays_streaming(
            graphs_path, code_ids, train_ids,
        )
        raw = None
    else:
        raw = _load_raw_graphs(graphs_path, set(code_ids), required)
        usable = {
            code_id for code_id in code_ids
            if all(raw.get(code_id, {}).get(view, {}).get("node_types") for view in required)
        }
    train = train[train.left_id.isin(usable) & train.right_id.isin(usable)].reset_index(drop=True)
    valid = valid[valid.left_id.isin(usable) & valid.right_id.isin(usable)].reset_index(drop=True)
    test = test[test.left_id.isin(usable) & test.right_id.isin(usable)].reset_index(drop=True)
    empty_splits = [name for name, frame in (("train", train), ("valid", valid), ("test", test)) if frame.empty]
    if empty_splits:
        raise RuntimeError(
            f"{config['name']} has no graph-evaluable rows in {empty_splits}. "
            f"Required graph layers: {required}; verify that the matching {dataset_key} clean-data ZIP is attached."
        )
    code_ids = sorted(set(train.left_id) | set(train.right_id) | set(valid.left_id) | set(valid.right_id) | set(test.left_id) | set(test.right_id))
    id_to_row = prepacked_id_to_row if prepacked_id_to_row is not None else {code_id: index for index, code_id in enumerate(code_ids)}
    print(
        f"[{config['name']}] usable pairs: train={len(train):,}, valid={len(valid):,}, test={len(test):,}; "
        f"packing {len(code_ids):,} graphs...",
        flush=True,
    )

    if method == "deckard":
        threshold, valid_metrics, test_scores, parameters, history, best_epoch = _run_deckard(
            raw, train, valid, test, id_to_row, train_ids, vectors=deckard_vectors,
        )
    else:
        if prepacked_arrays is not None:
            arrays, vocab = prepacked_arrays, prepacked_vocab
        else:
            arrays, vocab = _pack_method(method, raw, code_ids, train_ids)
        tree_depths = (
            arrays["binary_depths"].max(axis=1).astype(np.int16, copy=False)
            if method == "cdlh" else None
        )
        fa_train_pair_nodes = None
        if method.startswith("fa_ast"):
            train_left_rows, train_right_rows = _pair_row_indices(train, id_to_row)
            graph_nodes = arrays["node_types"].astype(bool).sum(1).astype(np.int16, copy=False)
            fa_train_pair_nodes = np.maximum(
                graph_nodes[train_left_rows], graph_nodes[train_right_rows],
            )
        # Neural baselines only require compact arrays after this point. Free
        # JSON-derived Python graphs before CUDA buffers are materialized.
        if raw is not None:
            del raw
        gc.collect()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _PairModel(method, arrays, len(vocab)).to(device)
        del arrays
        gc.collect()
        # Keep corpus buffers resident on one GPU.  ``DataParallel`` would
        # replicate those large buffers on every batch, making full-data runs
        # slower despite Kaggle exposing a second T4.
        print(f"[{config['name']}] using one GPU.", flush=True)
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        if method == "cdlh":
            assert tree_depths is not None
            train_left_rows, train_right_rows = _pair_row_indices(train, id_to_row)
            train_pair_depths = _pair_tree_depths(train, id_to_row, tree_depths)
            # Evaluation is order-invariant; grouping it by endpoint enables
            # exact within-batch encoder reuse and avoids deep ASTs inflating
            # otherwise shallow inference batches.
            valid = _sort_pairs_by_endpoint_and_depth(valid, id_to_row, tree_depths)
            test = _sort_pairs_by_endpoint_and_depth(test, id_to_row, tree_depths)
            loader = DataLoader(
                _PairRows(train, id_to_row),
                batch_sampler=_EndpointGroupedDepthBatchSampler(
                    train_left_rows, train_right_rows, train_pair_depths, config["batch"], seed,
                ),
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            unique_endpoints = len(np.unique(np.concatenate((train_left_rows, train_right_rows))))
            print(
                f"[{config['name']}] endpoint-grouped full-data batches: {len(train):,} pairs / "
                f"{unique_endpoints:,} train endpoints; pair depth "
                f"min/median/max={int(train_pair_depths.min())}/{int(np.median(train_pair_depths))}/{int(train_pair_depths.max())}.",
                flush=True,
            )
        else:
            dataset = _PairRows(train, id_to_row)
            if method.startswith("fa_ast"):
                assert fa_train_pair_nodes is not None
                # Graph-size buckets retain every pair and randomize their
                # order per epoch.  Combined with _flow's exact cropping this
                # prevents one large graph from making every FA-AST batch pay
                # for its 128x128 padded attention/message-passing envelope.
                loader = DataLoader(
                    dataset,
                    batch_sampler=_DepthBucketBatchSampler(fa_train_pair_nodes, config["batch"], seed),
                    num_workers=0, pin_memory=device.type == "cuda",
                )
                print(
                    f"[{config['name']}] size-bucketed FA-AST batches; graph nodes "
                    f"min/median/max={int(fa_train_pair_nodes.min())}/"
                    f"{int(np.median(fa_train_pair_nodes))}/{int(fa_train_pair_nodes.max())}.",
                    flush=True,
                )
            else:
                loader = DataLoader(
                    dataset, batch_size=config["batch"], shuffle=True,
                    num_workers=0, pin_memory=device.type == "cuda",
                )
        history, best = [], None
        training_microbatch = config["batch"]
        minimum_microbatch = config["batch"]
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            print(
                f"[{config['name']}] epoch {epoch}/{epochs} starting "
                f"(batch={config['batch']}, train pairs={len(train):,})...",
                flush=True,
            )
            model.train()
            total_loss, total_rows = 0.0, 0
            for left, right, labels in tqdm(loader, desc=f"{config['name']} epoch {epoch}"):
                batch_loss, used_microbatch = _adaptive_train_step(
                    model, method, optimizer, scaler, left, right, labels, device,
                    maximum_microbatch=training_microbatch,
                )
                training_microbatch = min(training_microbatch, used_microbatch)
                minimum_microbatch = min(minimum_microbatch, used_microbatch)
                total_loss += batch_loss
                total_rows += len(labels)
            valid_scores = _predict(model, valid, id_to_row, config["batch"], device)
            candidate_threshold, candidate_metrics = _choose_threshold(valid.label, valid_scores)
            row = {"epoch": epoch, "loss": total_loss / max(1, total_rows), **candidate_metrics, "threshold": candidate_threshold}
            history.append(row)
            key = _selection_key(candidate_metrics, valid.label)
            if best is None or key > best[0]:
                best = (key, copy.deepcopy(model.state_dict()), epoch, candidate_threshold, candidate_metrics)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        _, state, best_epoch, threshold, valid_metrics = best
        model.load_state_dict(state)
        print(f"[{config['name']}] evaluating validation-selected epoch {best_epoch} on test split...", flush=True)
        test_scores = _predict(model, test, id_to_row, config["batch"], device)

    test_metrics = _metric_dict(test.label, test_scores, threshold)
    runtime = time.perf_counter() - started
    result = {
        "Method": config["name"], "Implementation": "paper-faithful adaptation on repository-exported graphs",
        "BestEpoch": best_epoch, "BestValidF1": valid_metrics["F1"], "BestValidAcc": valid_metrics["Acc"],
        "ValidationSelectionMetric": _validation_selection(valid.label)["metric"],
        "ValidationBalanced": _validation_selection(valid.label)["balanced"],
        "ValidationPositives": _validation_selection(valid.label)["positives"],
        "ValidationNegatives": _validation_selection(valid.label)["negatives"], **test_metrics,
        "Threshold": threshold, "TrainPairs": len(train), "ValidPairs": len(valid), "TestPairs": len(test),
        "TrainableParameters": parameters, "RuntimeSeconds": runtime, "RuntimeMinutes": runtime / 60,
        "RunProfile": run_profile, "Seed": seed, "Execution": _gpu_execution_metadata(),
    }
    working = Path(globals().get("WORK_DIR_OVERRIDE", "/kaggle/working"))
    working.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_key}_{method}_faithful"
    pd.DataFrame([result]).to_csv(working / f"{stem}_results.csv", index=False)
    if history:
        pd.DataFrame(history).to_csv(working / f"{stem}_history.csv", index=False)
    manifest = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_key": dataset_key, "method": config["name"], "implementation": result["Implementation"],
        "input_graphs": list(required), "self_pair_count": 0, "hyperparameters": config,
        "profile": run_profile, "configured_epochs": epochs, "patience": patience, "seed": seed,
        "execution": _gpu_execution_metadata(), "global_batch_size": config["batch"],
        "minimum_training_microbatch": minimum_microbatch if method != "deckard" else None,
        "selected_epoch": best_epoch, "selected_threshold": threshold, "result": result,
    }
    (working / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    record_language_breakdown(test, test_scores, threshold, dataset=dataset_key, method=config["name"])
    print(pd.DataFrame([result]))
    return result
