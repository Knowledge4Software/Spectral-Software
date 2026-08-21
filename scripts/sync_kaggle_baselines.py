"""Synchronize baseline notebooks with their one canonical implementation.

The four dataset folders must differ only in dataset attachment/name.  This
script embeds the shared faithful-baseline runtime and applies the same
validation-selection policy to the native GNN and spectral-SNN controls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# CodeNet uses the same baseline families and the same generated runtime as the
# four established benchmarks.  Its notebooks are materialized below from the
# canonical AtCoder templates so all five folders stay behaviorally aligned.
DATASETS = ("atcoder_v3", "codexglue_v3", "gptclonebench_v3", "semanticclonebench_v3", "codenet_4l")
FAITHFUL = (
    "astnn_baseline.ipynb", "cdlh_baseline.ipynb", "deckard_baseline.ipynb",
    "deepsim_baseline.ipynb", "fa_ast_ggnn_baseline.ipynb", "fa_ast_gmn_baseline.ipynb",
    "rtvnn_baseline.ipynb",
)
FAITHFUL_METHODS = {
    "astnn_baseline.ipynb": "astnn",
    "cdlh_baseline.ipynb": "cdlh",
    "deckard_baseline.ipynb": "deckard",
    "deepsim_baseline.ipynb": "deepsim",
    "fa_ast_ggnn_baseline.ipynb": "fa_ast_ggnn",
    "fa_ast_gmn_baseline.ipynb": "fa_ast_gmn",
    "rtvnn_baseline.ipynb": "rtvnn",
}
NATIVE = ("gnn_baselines.ipynb", "snn_baselines.ipynb")
ALL_BASELINE_NOTEBOOKS = (*FAITHFUL, *NATIVE)


def _ensure_codenet_baseline_templates() -> int:
    """Create the CodeNet baseline suite from the canonical V3 templates.

    The generated content is normalized later in ``main`` exactly as it is for
    every other dataset.  Keeping only the dataset key/header different avoids
    silent baseline drift between CodeNet and the paper's established datasets.
    """
    source_dir = ROOT / "kaggle" / "atcoder_v3" / "baselines"
    target_dir = ROOT / "kaggle" / "codenet_4l" / "baselines"
    target_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for filename in ALL_BASELINE_NOTEBOOKS:
        target = target_dir / filename
        if target.exists():
            continue
        notebook = json.loads((source_dir / filename).read_text(encoding="utf-8"))
        header = notebook["cells"][0]
        source = _cell_source(header)
        source = source.replace("ATCoder V3", "CodeNet 4L 50k")
        source = source.replace("ATCoder", "CodeNet")
        source = source.replace('DATASET_KEYS = ("at-coder",)', 'DATASET_KEYS = ("codenet-4l",)')
        _set_cell_source(header, source)
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        created += 1
    return created


def _embed_source(path: Path) -> str:
    """Remove a module docstring/future import, which cannot follow a cell header."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith('"""'):
        opening = lines.pop(0)
        if opening.count('"""') < 2:
            while lines and lines.pop(0).strip() != '"""':
                pass
    lines = [line for line in lines if line != "from __future__ import annotations"]
    return "\n".join(lines).lstrip()


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_cell_source(cell: dict, source: str) -> None:
    cell["source"] = [line + "\n" for line in source.splitlines()]


def _faithful_cell() -> str:
    core = _embed_source(ROOT / "research/faithful_graph_baselines/core.py")
    runtime = _embed_source(ROOT / "research/faithful_graph_baselines/notebook_runtime.py")
    return "# Generated from research/faithful_graph_baselines; do not edit this cell by hand.\n\n" + core + "\n\n" + runtime


def _faithful_run_cell(method: str) -> str:
    return f'''# Execute the configured baseline. This cell is intentionally separate from
# the embedded runtime so a Kaggle "Run All" performs training rather than
# merely defining functions.
FAITHFUL_RESULTS = [run_faithful_baseline(dataset_key, "{method}") for dataset_key in DATASET_KEYS]
FAITHFUL_RESULTS_DF = pd.DataFrame(FAITHFUL_RESULTS)
display(FAITHFUL_RESULTS_DF)
''' 


def _normalise_faithful_header(source: str, *, codenet: bool = False) -> str:
    """Give every generated faithful run an explicit training schedule."""
    schedule_marker = '# Shared faithful-baseline schedule; consumed by notebook_runtime.py.\n'
    if schedule_marker not in source:
        old = "RUN_CONFIG = RUN_PRESETS[RUN_PROFILE]\n"
        new = (
            "RUN_CONFIG = dict(RUN_PRESETS[RUN_PROFILE])\n"
            + schedule_marker
            + 'RUN_CONFIG["epochs"] = 4\n'
            + 'RUN_CONFIG.setdefault("patience", 2)\n'
        )
        if old not in source:
            raise RuntimeError("Could not find the faithful baseline RUN_CONFIG declaration")
        source = source.replace(old, new, 1)
    # Normalize prior generated schedules to the four-epoch paper protocol.
    source = source.replace('RUN_CONFIG.setdefault("epochs", 4)\n', 'RUN_CONFIG["epochs"] = 4\n')
    source = source.replace('RUN_CONFIG.setdefault("epochs", 5)\n', 'RUN_CONFIG["epochs"] = 4\n')
    source = source.replace('RUN_CONFIG["epochs"] = 5\n', 'RUN_CONFIG["epochs"] = 4\n')
    memory_marker = '# Prevent CUDA allocator fragmentation on long tree-baseline runs.\n'
    if memory_marker not in source:
        old = "import torch\n"
        new = (
            "import os\n"
            + memory_marker
            + 'os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")\n'
            + "import torch\n"
        )
        if old not in source:
            raise RuntimeError("Could not find the faithful baseline torch import")
        source = source.replace(old, new, 1)
    resource_marker = "# CodeNet 50k resource guard; consumed by notebook_runtime.py.\n"
    if codenet:
        resource_block = (
            resource_marker
            +
            "# A 128-node cap keeps all CodeNet graph baselines viable on one 16-GB T4.\n"
            "MAX_NODES = 128\n"
            "MAX_EDGES = 256\n"
            "MAX_STATEMENTS = 32\n"
            "GNN_BATCH_SIZE = 16\n"
            "SNN_BATCH_SIZE = 2048\n"
        )
        if resource_marker not in source:
            source += "\n" + resource_block
        else:
            source = re.sub(
                rf"{re.escape(resource_marker)}(?:.*\n){{0,8}}",
                resource_block,
                source,
                count=1,
            )
    return source


def _gnn_helpers() -> str:
    return '''    def validation_selection_context(labels: np.ndarray) -> dict:
        labels = np.asarray(labels, dtype=np.int64)
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        balanced = positives > 0 and positives == negatives
        return {"balanced": balanced, "metric": "Accuracy" if balanced else "F1", "positives": positives, "negatives": negatives}


    def validation_selection_key(metrics: dict, labels: np.ndarray) -> tuple[float, float]:
        context = validation_selection_context(labels)
        return (metrics["Acc"], metrics["F1"]) if context["balanced"] else (metrics["F1"], metrics["BalancedAccuracy"])


    def best_validation_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict]:
        if probs.size == 0:
            return 0.5, binary_metrics(labels, probs, threshold=0.5)
        candidates = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, 101)))
        candidates = np.unique(np.concatenate([candidates, np.asarray([0.5], dtype=np.float32)]))
        best_threshold = 0.5
        best_metrics = None
        for threshold in candidates:
            metrics = binary_metrics(labels, probs, threshold=float(threshold))
            if best_metrics is None or validation_selection_key(metrics, labels) > validation_selection_key(best_metrics, labels):
                best_metrics = metrics
                best_threshold = float(threshold)
        return best_threshold, best_metrics or binary_metrics(labels, probs, threshold=0.5)
'''


def _snn_helpers() -> str:
    return '''    def validation_selection_context(labels: np.ndarray) -> dict:
        labels = np.asarray(labels, dtype=np.int64)
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        balanced = positives > 0 and positives == negatives
        return {"balanced": balanced, "metric": "Accuracy" if balanced else "F1", "positives": positives, "negatives": negatives}


    def validation_selection_key(metrics: dict, labels: np.ndarray) -> tuple[float, float]:
        context = validation_selection_context(labels)
        return (metrics["Acc"], metrics["F1"]) if context["balanced"] else (metrics["F1"], metrics["BalancedAccuracy"])


    def best_validation_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
        candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 501)))
        candidates = np.unique(np.concatenate([candidates, np.asarray([0.5], dtype=np.float32)]))
        best_thr = 0.5
        best_metrics = None
        for thr in candidates:
            current = metrics_at_threshold(labels, scores, float(thr))
            if best_metrics is None or validation_selection_key(current, labels) > validation_selection_key(best_metrics, labels):
                best_thr = float(thr)
                best_metrics = current
        return best_thr
'''


def _replace_function(source: str, start: str, end: str, replacement: str) -> str:
    left = source.index(start)
    right = source.index(end, left)
    return source[:left] + replacement + source[right:]


def _patch_gnn(source: str) -> str:
    source = _replace_function(source, "    def best_f1_threshold", "\n\n    def train_one_graph_type", _gnn_helpers())
    source = source.replace("best_f1_threshold(", "best_validation_threshold(")
    source = source.replace("        best_valid_f1 = -1.0\n", "        best_valid_f1 = -1.0\n        best_selection_key = None\n")
    source = source.replace("            scheduler.step(valid_metrics[\"F1\"])", "            current_selection_key = validation_selection_key(valid_metrics, valid_labels)\n            scheduler.step(current_selection_key[0])")
    source = source.replace(
        "            if valid_metrics[\"F1\"] > best_valid_f1 + 1e-5:\n                best_valid_f1 = valid_metrics[\"F1\"]",
        "            if best_selection_key is None or current_selection_key > best_selection_key:\n                best_selection_key = current_selection_key\n                best_valid_f1 = valid_metrics[\"F1\"]",
    )
    source = source.replace(
        '            "BestValidF1": best_valid_f1,\n',
        '            "BestValidF1": best_valid_f1,\n            "BestValidAcc": selected_valid_metrics["Acc"],\n            "ValidationSelectionMetric": validation_selection_context(valid_labels)["metric"],\n            "ValidationBalanced": validation_selection_context(valid_labels)["balanced"],\n            "ValidationPositives": validation_selection_context(valid_labels)["positives"],\n            "ValidationNegatives": validation_selection_context(valid_labels)["negatives"],\n',
    )
    source = source.replace('            "TrainPairs": len(tr),\n            "TestPairs": len(te),', '            "TrainPairs": len(tr),\n            "ValidPairs": len(eval_df),\n            "TestPairs": len(te),')
    return source


def _patch_snn(source: str) -> str:
    source = _replace_function(source, "    def best_f1_threshold", "\n\n    @torch.no_grad()", _snn_helpers())
    source = source.replace("best_f1_threshold(", "best_validation_threshold(")
    source = source.replace("        best_valid_f1 = -1.0\n", "        best_valid_f1 = -1.0\n        best_valid_acc = -1.0\n        best_selection_key = None\n")
    source = source.replace("            scheduler.step(valid_metrics[\"F1\"])", "            current_selection_key = validation_selection_key(valid_metrics, y_valid)\n            scheduler.step(current_selection_key[0])")
    source = source.replace(
        "            if valid_metrics[\"F1\"] > best_valid_f1:\n                best_valid_f1 = valid_metrics[\"F1\"]",
        "            if best_selection_key is None or current_selection_key > best_selection_key:\n                best_selection_key = current_selection_key\n                best_valid_f1 = valid_metrics[\"F1\"]\n                best_valid_acc = valid_metrics[\"Acc\"]",
    )
    source = source.replace(
        '            "BestValidF1": best_valid_f1,\n            **test_metrics,',
        '            "BestValidF1": best_valid_f1,\n            "BestValidAcc": best_valid_acc,\n            "ValidationSelectionMetric": validation_selection_context(y_valid)["metric"],\n            "ValidationBalanced": validation_selection_context(y_valid)["balanced"],\n            "ValidationPositives": validation_selection_context(y_valid)["positives"],\n            "ValidationNegatives": validation_selection_context(y_valid)["negatives"],\n            **test_metrics,',
    )
    source = source.replace('            "TrainPairs": int(len(split_data["train"].labels)),\n            "TestPairs": int(len(split_data["test"].labels)),', '            "TrainPairs": int(len(split_data["train"].labels)),\n            "ValidPairs": int(len(split_data["valid"].labels)),\n            "TestPairs": int(len(split_data["test"].labels)),')
    return source


def _normalise_native_runtime(source: str, filename: str) -> str:
    """Apply persistent Kaggle safety fixes to native GNN/SNN notebooks."""
    if filename == "gnn_baselines.ipynb":
        source = source.replace(
            "- If CUDA runs out of memory, reduce `BATCH_SIZE` from `2048` to `1024`.",
            "- T4-safe adaptive batching starts at `64` and automatically halves an outlier batch after CUDA OOM.",
        )
        # A pair batch can contain twice as many distinct, variable-size
        # graphs.  Keep the fast path at 64 and bisect only an outlier batch.
        source = source.replace("    BATCH_SIZE = 2048\n", "    BATCH_SIZE = 64\n")
        source = source.replace("    BATCH_SIZE = 512\n", "    BATCH_SIZE = 64\n")
        source = source.replace(
            "    BATCH_SIZE = 64\n",
            '    BATCH_SIZE = int(globals().get("GNN_BATCH_SIZE", 64))\n',
        )
        if (
            "def layer_to_graph" in source
            and "    MAX_GRAPH_NODES = int(globals().get(\"MAX_NODES\", 256))\n" not in source
        ):
            config_marker = "    # =========================\n    # Load and sample pairs\n"
            source = source.replace(
                config_marker,
                "    # CodeNet injects MAX_NODES=128 in its header; other datasets use 256.\n"
                "    MAX_GRAPH_NODES = int(globals().get(\"MAX_NODES\", 256))\n\n"
                + config_marker,
                1,
            )
            old_graph_cap = '''        n = int(adjacency.get("num_nodes", 0) or 0)
        if n <= 0:
            n = 1
        row = np.asarray(adjacency.get("row", []), dtype=np.int64)
        col = np.asarray(adjacency.get("col", []), dtype=np.int64)
        raw_edge_count = int(min(row.size, col.size))

        if row.size and col.size:
            row = np.clip(row, 0, n - 1)
            col = np.clip(col, 0, n - 1)
'''
            new_graph_cap = '''        raw_nodes = int(adjacency.get("num_nodes", 0) or 0)
        n = max(1, min(raw_nodes, MAX_GRAPH_NODES))
        row = np.asarray(adjacency.get("row", []), dtype=np.int64)
        col = np.asarray(adjacency.get("col", []), dtype=np.int64)
        valid_edges = (row >= 0) & (row < n) & (col >= 0) & (col < n)
        row, col = row[valid_edges], col[valid_edges]
        raw_edge_count = int(min(row.size, col.size))

        if row.size and col.size:
'''
            if old_graph_cap not in source:
                raise RuntimeError("Could not install GNN node cap")
            source = source.replace(old_graph_cap, new_graph_cap, 1)
        if "    def gnn_is_cuda_oom" not in source and "    def predict_probs" in source:
            source = source.replace("    import time\n", "    import time\n    import gc\n", 1)
            old_predict = '''    @torch.no_grad()
    def predict_probs(model: PairGNN, df: pd.DataFrame, graphs: Dict[str, GraphData], device: str) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_probs = []
        all_labels = []
        for batch in batches(df, BATCH_SIZE, shuffle=False):
            left_ids = batch.left_id.tolist()
            right_ids = batch.right_id.tolist()
            with maybe_autocast(device):
                logits = model.forward_pairs(left_ids, right_ids, graphs, device)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch.label.to_numpy(dtype=np.int64))
        probs = np.concatenate(all_probs) if all_probs else np.asarray([], dtype=np.float32)
        labels = np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.int64)
        return probs, labels
'''
            new_predict = '''    def gnn_is_cuda_oom(error: BaseException) -> bool:
        return isinstance(error, torch.OutOfMemoryError) or (
            isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
        )


    @torch.no_grad()
    def predict_batch_adaptive(model: PairGNN, batch: pd.DataFrame, graphs: Dict[str, GraphData], device: str) -> np.ndarray:
        logits = None
        try:
            with maybe_autocast(device):
                logits = model.forward_pairs(batch.left_id.tolist(), batch.right_id.tolist(), graphs, device)
            return torch.sigmoid(logits).float().cpu().numpy()
        except RuntimeError as error:
            if not gnn_is_cuda_oom(error) or len(batch) <= 1:
                raise
            logits = None
            del error
            gc.collect()
            torch.cuda.empty_cache()
            middle = len(batch) // 2
            return np.concatenate((
                predict_batch_adaptive(model, batch.iloc[:middle], graphs, device),
                predict_batch_adaptive(model, batch.iloc[middle:], graphs, device),
            ))


    @torch.no_grad()
    def predict_probs(model: PairGNN, df: pd.DataFrame, graphs: Dict[str, GraphData], device: str) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_probs = []
        all_labels = []
        for batch in batches(df, BATCH_SIZE, shuffle=False):
            all_probs.append(predict_batch_adaptive(model, batch, graphs, device))
            all_labels.append(batch.label.to_numpy(dtype=np.int64))
        probs = np.concatenate(all_probs) if all_probs else np.asarray([], dtype=np.float32)
        labels = np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.int64)
        return probs, labels
'''
            if old_predict not in source:
                raise RuntimeError("Could not install adaptive GNN prediction")
            source = source.replace(old_predict, new_predict, 1)

            old_training = '''        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            train_batches = batches(tr, BATCH_SIZE, shuffle=True)
            total_batches = math.ceil(len(tr) / BATCH_SIZE)
            train_progress = tqdm(
                train_batches,
                total=total_batches,
                desc=f"{graph_type.upper()} epoch {epoch:02d}/{EPOCHS}",
                unit="batch",
                leave=False,
            )
            for batch in train_progress:
                left_ids = batch.left_id.tolist()
                right_ids = batch.right_id.tolist()
                labels = torch.tensor(batch.label.to_numpy(dtype=np.float32), device=DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with maybe_autocast(DEVICE):
                    logits = model.forward_pairs(left_ids, right_ids, graphs, DEVICE)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
                if losses:
                    train_progress.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
'''
            new_training = '''        training_microbatch = BATCH_SIZE
        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            train_batches = batches(tr, BATCH_SIZE, shuffle=True)
            total_batches = math.ceil(len(tr) / BATCH_SIZE)
            train_progress = tqdm(
                train_batches,
                total=total_batches,
                desc=f"{graph_type.upper()} epoch {epoch:02d}/{EPOCHS}",
                unit="batch",
                leave=False,
            )
            for batch in train_progress:
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    weighted_loss = 0.0
                    micro = labels = logits = loss = scaled_loss = None
                    try:
                        for start in range(0, len(batch), training_microbatch):
                            micro = batch.iloc[start:start + training_microbatch]
                            labels = torch.tensor(micro.label.to_numpy(dtype=np.float32), device=DEVICE)
                            with maybe_autocast(DEVICE):
                                logits = model.forward_pairs(
                                    micro.left_id.tolist(), micro.right_id.tolist(), graphs, DEVICE
                                )
                                loss = criterion(logits, labels)
                                scaled_loss = loss * (len(micro) / len(batch))
                            scaler.scale(scaled_loss).backward()
                            weighted_loss += float(loss.detach().cpu()) * len(micro)
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                        scaler.step(optimizer)
                        scaler.update()
                        losses.append(weighted_loss / max(1, len(batch)))
                        break
                    except RuntimeError as error:
                        if not gnn_is_cuda_oom(error) or training_microbatch <= 1:
                            raise
                        previous = training_microbatch
                        training_microbatch = max(1, training_microbatch // 2)
                        optimizer.zero_grad(set_to_none=True)
                        micro = labels = logits = loss = scaled_loss = None
                        del error
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(
                            f"[CUDA OOM recovery] {graph_type.upper()} microbatch "
                            f"{previous} -> {training_microbatch}"
                        )
                if losses:
                    train_progress.set_postfix(
                        loss=f"{np.mean(losses[-50:]):.4f}", microbatch=training_microbatch
                    )
'''
            if old_training not in source:
                raise RuntimeError("Could not install adaptive GNN training")
            source = source.replace(old_training, new_training, 1)
            source = source.replace(
                '            "TrainableParameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),\n',
                '            "TrainableParameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),\n'
                '            "MinimumTrainingMicrobatch": int(training_microbatch),\n',
                1,
            )
        if '"MacroF1": test_metrics["MacroF1"]' not in source and '"P": test_metrics["P"]' in source:
            source = source.replace(
                '            "Acc": test_metrics["Acc"],\n            "Threshold": selected_threshold,',
                '            "Acc": test_metrics["Acc"],\n'
                '            "MacroF1": test_metrics["MacroF1"],\n'
                '            "BalancedAccuracy": test_metrics["BalancedAccuracy"],\n'
                '            "ROC_AUC": test_metrics["ROC_AUC"],\n'
                '            "Threshold": selected_threshold,',
                1,
            )
        empty_result = 'return {"Method": graph_type.upper(), "P": 0.0, "R": 0.0, "F1": 0.0, "Acc": 0.0}'
        if empty_result in source:
            source = source.replace(
                empty_result,
                'return {\n'
                '                "Method": graph_type.upper(), "P": 0.0, "R": 0.0, "F1": 0.0, "Acc": 0.0,\n'
                '                "MacroF1": 0.0, "BalancedAccuracy": 0.0, "ROC_AUC": float("nan"),\n'
                '                "TrainableParameters": 0, "RuntimeSeconds": float(time.perf_counter() - graph_started),\n'
                '                "RuntimeMinutes": float(time.perf_counter() - graph_started) / 60.0,\n'
                '            }',
                1,
            )
    if filename == "snn_baselines.ipynb" and "def make_loader" in source and "batchnorm_singleton_guard" not in source:
        old = '''        kwargs = {
            "batch_size": BATCH_SIZE,
            "shuffle": shuffle,
            "num_workers": NUM_WORKERS,
            "pin_memory": DEVICE == "cuda",
        }
'''
        new = '''        row_count = len(arrays.labels)
        effective_batch = min(BATCH_SIZE, max(1, row_count))
        # batchnorm_singleton_guard: BatchNorm cannot train on a final batch of one.
        drop_singleton = bool(shuffle and row_count > effective_batch and row_count % effective_batch == 1)
        kwargs = {
            "batch_size": effective_batch,
            "shuffle": shuffle,
            "num_workers": NUM_WORKERS,
            "pin_memory": DEVICE == "cuda",
            "drop_last": drop_singleton,
        }
'''
        if old not in source:
            raise RuntimeError("Could not install the SNN singleton-batch guard")
        source = source.replace(old, new, 1)
    if filename == "snn_baselines.ipynb":
        source = source.replace(
            "    BATCH_SIZE = 8192\n",
            '    BATCH_SIZE = int(globals().get("SNN_BATCH_SIZE", 8192))\n',
        )
    return source


def main() -> None:
    faithful = _faithful_cell()
    changed = _ensure_codenet_baseline_templates()
    for dataset in DATASETS:
        directory = ROOT / "kaggle" / dataset / "baselines"
        for filename in FAITHFUL:
            path = directory / filename
            notebook = json.loads(path.read_text(encoding="utf-8"))
            header = notebook["cells"][0]
            normalized_header = _normalise_faithful_header(
                _cell_source(header), codenet=(dataset == "codenet_4l")
            )
            notebook_changed = False
            if normalized_header != _cell_source(header):
                _set_cell_source(header, normalized_header)
                notebook_changed = True
            target = next(cell for cell in notebook["cells"] if "# Generated from research/faithful_graph_baselines" in _cell_source(cell))
            if _cell_source(target).rstrip("\n") != faithful.rstrip("\n"):
                _set_cell_source(target, faithful)
                notebook_changed = True
            run_cell = _faithful_run_cell(FAITHFUL_METHODS[filename])
            existing_runs = [cell for cell in notebook["cells"] if "# Execute the configured baseline." in _cell_source(cell)]
            if len(existing_runs) != 1 or _cell_source(existing_runs[0]).rstrip("\n") != run_cell.rstrip("\n"):
                for cell in existing_runs:
                    notebook["cells"].remove(cell)
                notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []})
                _set_cell_source(notebook["cells"][-1], run_cell)
                notebook_changed = True
            if notebook_changed:
                path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                changed += 1
        for filename, patcher in (("gnn_baselines.ipynb", _patch_gnn), ("snn_baselines.ipynb", _patch_snn)):
            path = directory / filename
            notebook = json.loads(path.read_text(encoding="utf-8"))
            notebook_changed = False
            header = notebook["cells"][0]
            normalized_header = _normalise_faithful_header(
                _cell_source(header), codenet=(dataset == "codenet_4l")
            )
            if normalized_header != _cell_source(header):
                _set_cell_source(header, normalized_header)
                notebook_changed = True
            for cell in notebook["cells"]:
                old_source = _cell_source(cell)
                new_source = patcher(old_source) if "def best_f1_threshold" in old_source else old_source
                new_source = _normalise_native_runtime(new_source, filename)
                if new_source != old_source:
                    _set_cell_source(cell, new_source)
                    notebook_changed = True
            if notebook_changed:
                path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                changed += 1
    print(f"Synchronized {changed} notebook(s).")


if __name__ == "__main__":
    main()
