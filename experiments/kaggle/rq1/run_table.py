"""Produce every RQ1 table row after downloading latent-graph Kaggle exports.

For each dataset and graph view this runner reports P/R/F1/Acc for PSS
(``No Train``), Random Forest, Logistic Regression, and the reference SNN.
It uses official train/valid/test pairs and never uses test for selection.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kaggle.rq1.evaluate_pss import (
    ALL_GRAPH_TYPES,
    CONVENTIONAL_GRAPH_TYPES,
    common_support,
    load_conventional_spectra,
    load_latent_export,
)
from spectral_code.similarity.pss import PSSSimilarity


K_EIGEN = 128
OUTPUTS_ROOT = ROOT.parent / "outputs"
DEFAULT_RQ1_ROOT = OUTPUTS_ROOT / "kaggle" / "RQ1"
DEFAULT_OUTPUT_DIR = DEFAULT_RQ1_ROOT / "table_results"
DEFAULT_DATASETS = {
    "atcoder": {
        "clean_data": OUTPUTS_ROOT / "atcoder_v3" / "clean_data",
        "latent_export": DEFAULT_RQ1_ROOT / "atcoder_export_latent_graphs.zip",
    },
    "xglue": {
        "clean_data": OUTPUTS_ROOT / "codexglue_v3" / "clean_data",
        "latent_export": DEFAULT_RQ1_ROOT / "xglue_export_latent_graphs.zip",
    },
}
METHOD_ORDER = (
    "SPECTRA-Siam",
    "SPECTRA-Siam Spectrum + No Train",
    "SPECTRA-Siam Spectrum + RF",
    "SPECTRA-Siam Spectrum + LR",
    "SPECTRA-Siam Spectrum + SNN",
    "AST + No Train", "AST + RF", "AST + LR", "AST + SNN",
    "CFG + No Train", "CFG + RF", "CFG + LR", "CFG + SNN",
    "DDG + No Train", "DDG + RF", "DDG + LR", "DDG + SNN",
    "CPG + No Train", "CPG + RF", "CPG + LR", "CPG + SNN",
    "Predict All Clone",
)
DISPLAY_DATASETS = {"atcoder": "AtCoder", "xglue": "BigCloneBench"}
CURRENT_MAIN_METHOD_SIGNATURE = {
    "input_ablation": "lex",
    "use_node_lexical": True,
    "use_source_lexical": False,
}


@dataclass(frozen=True)
class SplitArrays:
    left: np.ndarray
    right: np.ndarray
    labels: np.ndarray


def require_current_main_method_export(manifest: dict) -> None:
    """Reject RQ1 archives made by an older SPECTRA-Siam input variant.

    The RQ1 direct-method row and learned latent spectra must come from the
    same lexical-node-input method reported in the main paper table.  In
    particular, a prior export with a source-token lexical residual enabled is
    not comparable to the current fixed-band spectral method, even though both
    archives may be named ``input_only_lex``.
    """
    if str(manifest.get("dataset", "")).lower() not in {"atcoder_v3", "codexglue_v3"}:
        return
    signature = dict(manifest.get("method_input_signature") or {})
    config = dict(manifest.get("model_config") or {})
    observed = {
        "input_ablation": signature.get("input_ablation"),
        "use_node_lexical": signature.get("use_node_lexical", config.get("use_node_lexical")),
        "use_source_lexical": signature.get("use_source_lexical", config.get("use_source_lexical")),
    }
    if observed != CURRENT_MAIN_METHOD_SIGNATURE:
        raise RuntimeError(
            "Stale/incompatible RQ1 latent export: expected the current main "
            f"SPECTRA-Siam signature {CURRENT_MAIN_METHOD_SIGNATURE}, found {observed}. "
            "Re-run the current RQ1 Kaggle export notebook and replace this ZIP; "
            "do not combine its direct or latent-spectrum rows with the current main method."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_metric(labels: np.ndarray) -> str:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError("Each official split must contain clones and non-clones")
    return "Accuracy" if positives == negatives else "F1"


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int8)
    return {
        "P": float(precision_score(labels, predictions, zero_division=0)),
        "R": float(recall_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "Acc": float(accuracy_score(labels, predictions)),
        "MacroF1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "BalancedAccuracy": float(balanced_accuracy_score(labels, predictions)),
        "Threshold": float(threshold),
        "TP": int(((predictions == 1) & (labels == 1)).sum()),
        "FP": int(((predictions == 1) & (labels == 0)).sum()),
        "TN": int(((predictions == 0) & (labels == 0)).sum()),
        "FN": int(((predictions == 0) & (labels == 1)).sum()),
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray, primary: str) -> float:
    """Use validation Accuracy on exact 50/50 data, otherwise validation F1."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    if primary == "F1":
        objective = f1
    elif primary == "Accuracy":
        positives = float(labels.sum())
        negatives = float(len(labels) - labels.sum())
        true_positives = recall[:-1] * positives
        false_positives = np.divide(
            true_positives,
            precision[:-1],
            out=np.full_like(true_positives, negatives),
            where=precision[:-1] > 0,
        ) - true_positives
        objective = (true_positives + negatives - false_positives) / len(labels)
    else:
        raise ValueError(primary)
    index = max(range(len(thresholds)), key=lambda value: (objective[value], f1[value], thresholds[value]))
    return float(thresholds[index])


def primary_value(metric_values: dict, primary: str) -> float:
    return float(metric_values["Acc" if primary == "Accuracy" else "F1"])


def eigen_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.zeros(8, dtype=np.float32)
    q25, q50, q75 = np.percentile(values, (25, 50, 75)).astype(np.float32)
    return np.asarray(
        (min(len(values) / 2000.0, 10.0), values.mean(), values.std(), values.min(), values.max(), q25, q50, q75),
        dtype=np.float32,
    )


def vector_from_spectrum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    values.sort()
    padded = np.zeros(K_EIGEN, dtype=np.float32)
    padded[: min(K_EIGEN, len(values))] = values[:K_EIGEN]
    return np.concatenate((eigen_stats(values), padded)).astype(np.float32, copy=False)


def pair_features(
    matrix: np.ndarray, left: np.ndarray, right: np.ndarray, *, chunk: int = 100_000, desc: str | None = None,
) -> np.ndarray:
    dimension = matrix.shape[1]
    result = np.empty((len(left), 2 * dimension + 2), dtype=np.float32)
    starts = range(0, len(left), chunk)
    if desc:
        starts = tqdm(starts, desc=desc, unit="chunk", leave=False, total=-(-len(left) // chunk))
    for start in starts:
        end = min(len(left), start + chunk)
        lhs, rhs = matrix[left[start:end]], matrix[right[start:end]]
        diff = np.abs(lhs - rhs)
        product = lhs * rhs
        cosine = (lhs * rhs).sum(axis=1, keepdims=True) / (
            np.linalg.norm(lhs, axis=1, keepdims=True) * np.linalg.norm(rhs, axis=1, keepdims=True) + 1e-8
        )
        l2 = np.linalg.norm(lhs - rhs, axis=1, keepdims=True)
        result[start:end] = np.concatenate((diff, product, cosine, l2), axis=1)
    return result


def matrices_and_splits(
    pairs: pd.DataFrame, spectra: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, SplitArrays]]:
    code_ids = sorted({str(value) for value in pairs.left_id} | {str(value) for value in pairs.right_id})
    index = {code_id: position for position, code_id in enumerate(code_ids)}
    matrix = np.stack([vector_from_spectrum(spectra[code_id]) for code_id in code_ids])
    splits = {}
    for split in ("train", "valid", "test"):
        frame = pairs[pairs.split.eq(split)]
        splits[split] = SplitArrays(
            frame.left_id.map(index).to_numpy(dtype=np.int64),
            frame.right_id.map(index).to_numpy(dtype=np.int64),
            frame.label.to_numpy(dtype=np.int8),
        )
    return matrix, splits


class PairIndexDataset(Dataset):
    def __init__(self, split: SplitArrays) -> None:
        self.left = torch.from_numpy(np.array(split.left, copy=True)).long()
        self.right = torch.from_numpy(np.array(split.right, copy=True)).long()
        self.labels = torch.from_numpy(np.array(split.labels, copy=True)).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.left[index], self.right[index], self.labels[index]


class SiameseSpectralNet(nn.Module):
    """The shared project SNN architecture used by the spectral baselines."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, embed_dim: int = 128, dropout: float = 0.10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = F.normalize(self.encoder(left), dim=-1)
        right = F.normalize(self.encoder(right), dim=-1)
        difference = (left - right).abs()
        product = left * right
        cosine = product.sum(dim=-1, keepdim=True)
        l2 = torch.linalg.vector_norm(left - right, dim=-1, keepdim=True)
        return self.head(torch.cat((difference, product, cosine, l2), dim=-1)).squeeze(-1)


def loaders(splits: dict[str, SplitArrays], batch_size: int, device: str) -> dict[str, DataLoader]:
    output = {}
    for name, split in splits.items():
        effective_batch = min(batch_size, max(1, len(split.labels)))
        # BatchNorm cannot train on a final singleton. Evaluation is safe.
        drop_singleton = name == "train" and len(split.labels) > effective_batch and len(split.labels) % effective_batch == 1
        output[name] = DataLoader(
            PairIndexDataset(split), batch_size=effective_batch, shuffle=name == "train",
            num_workers=0, pin_memory=device == "cuda", drop_last=drop_singleton,
        )
    return output


@torch.no_grad()
def snn_scores(model: nn.Module, codes: torch.Tensor, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all, scores_all = [], []
    for left, right, labels in loader:
        logits = model(codes[left.to(device, non_blocking=True)], codes[right.to(device, non_blocking=True)])
        labels_all.append(labels.numpy())
        scores_all.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(labels_all).astype(np.int8), np.concatenate(scores_all).astype(np.float64)


def run_snn(matrix: np.ndarray, splits: dict[str, SplitArrays], args: argparse.Namespace, seed: int) -> tuple[np.ndarray, np.ndarray, float, dict]:
    set_seed(seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    train_endpoints = np.unique(np.concatenate((splits["train"].left, splits["train"].right)))
    mean = matrix[train_endpoints].mean(axis=0, keepdims=True)
    std = matrix[train_endpoints].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    codes = torch.from_numpy((matrix - mean) / std).float().to(device)
    data = loaders(splits, args.snn_batch_size, device)
    model = SiameseSpectralNet(codes.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    primary = select_metric(splits["valid"].labels)
    best_state = None
    best_score = None
    best_threshold = 0.5
    stale = 0
    for epoch in range(1, args.snn_epochs + 1):
        model.train()
        batches = tqdm(
            data["train"],
            desc=f"SNN epoch {epoch}/{args.snn_epochs}",
            unit="batch",
            leave=False,
        )
        for left, right, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(codes[left.to(device, non_blocking=True)], codes[right.to(device, non_blocking=True)])
            loss = criterion(logits, labels.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        valid_labels, valid_scores = snn_scores(model, codes, data["valid"], device)
        threshold = choose_threshold(valid_labels, valid_scores, primary)
        valid = metrics(valid_labels, valid_scores, threshold)
        ranking = (primary_value(valid, primary), valid["F1"], valid["BalancedAccuracy"], valid["Acc"])
        scheduler.step(primary_value(valid, primary))
        if best_score is None or ranking > best_score:
            best_score, best_threshold, stale = ranking, threshold, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.snn_patience:
                break
    model.load_state_dict({name: tensor.to(device) for name, tensor in best_state.items()})
    _, test_scores = snn_scores(model, codes, data["test"], device)
    _, valid_scores = snn_scores(model, codes, data["valid"], device)
    details = {
        "BestEpoch": epoch - stale,
        "ValidationSelectionMetric": primary,
        "TrainableParameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "Device": device,
    }
    del model, optimizer, scheduler, codes, data
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return valid_scores, test_scores, best_threshold, details


def pss_scores(pairs: pd.DataFrame, spectra: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    metric = PSSSimilarity()
    output = {}
    # No-Train PSS has no fitted state, so train scores are intentionally not
    # computed. Validation calibrates its threshold and test is held out.
    for split in ("valid", "test"):
        frame = pairs[pairs.split.eq(split)]
        left = [spectra[str(code_id)] for code_id in frame.left_id]
        right = [spectra[str(code_id)] for code_id in frame.right_id]
        output[split] = metric.compute_many(left, right)
    return output


def row(dataset: str, graph: str, learner: str, test: dict, valid: dict, details: dict, seconds: float) -> dict:
    prefix = "SPECTRA-Siam Spectrum" if graph == "latent" else graph.upper()
    return {
        "Dataset": DISPLAY_DATASETS[dataset], "Graph": graph.upper(), "Method": f"{prefix} + {learner}",
        "P": test["P"], "R": test["R"], "F1": test["F1"], "Acc": test["Acc"],
        "BestValidP": valid["P"], "BestValidR": valid["R"], "BestValidF1": valid["F1"], "BestValidAcc": valid["Acc"],
        "Threshold": test["Threshold"], "ValidationSelectionMetric": details["ValidationSelectionMetric"],
        "TrainPairs": details["TrainPairs"], "ValidPairs": details["ValidPairs"], "TestPairs": details["TestPairs"],
        "TrainableParameters": details.get("TrainableParameters", np.nan), "Device": details.get("Device", "cpu"),
        "RuntimeSeconds": seconds, "RuntimeMinutes": seconds / 60.0,
    }


def _find_checkpoint_bytes(archive: zipfile.ZipFile) -> bytes | None:
    """Return the trained checkpoint from an export archive.

    A Kaggle export that trained the model keeps ``*_final.pt`` beside the
    latent-graph folder, but an export produced from an existing checkpoint
    carries it only inside the nested latent-graph ZIP. Both layouts hold the
    same file, so look one level down before giving up.
    """
    names = archive.namelist()
    direct = [name for name in names if name.endswith("_final.pt")]
    if direct:
        return archive.read(direct[0])
    for nested_name in (name for name in names if name.endswith(".zip")):
        with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested:
            inner = [name for name in nested.namelist() if name.endswith("_final.pt")]
            if inner:
                return nested.read(inner[0])
    return None


def direct_spectra_siam_row(dataset: str, archive_path: Path) -> dict | None:
    with zipfile.ZipFile(archive_path) as archive:
        payload = _find_checkpoint_bytes(archive)
    if payload is None:
        return None
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    test = checkpoint.get("test")
    if not test:
        return None
    return {
        "Dataset": DISPLAY_DATASETS[dataset], "Graph": "LATENT", "Method": "SPECTRA-Siam",
        "P": float(test["Precision"]), "R": float(test["Recall"]), "F1": float(test["F1"]), "Acc": float(test["Accuracy"]),
        "BestValidP": np.nan, "BestValidR": np.nan, "BestValidF1": float(checkpoint["best"]["valid"]["F1"]), "BestValidAcc": float(checkpoint["best"]["valid"]["Accuracy"]),
        "Threshold": float(checkpoint["best"]["threshold"]), "ValidationSelectionMetric": checkpoint["best"].get("selection_metric", "unknown"),
        "TrainPairs": np.nan, "ValidPairs": np.nan, "TestPairs": np.nan, "TrainableParameters": np.nan, "Device": "Kaggle", "RuntimeSeconds": np.nan, "RuntimeMinutes": np.nan,
    }


def run_dataset(
    dataset: str, clean_data: Path, latent_export: Path, output: Path, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest, pairs, latent = load_latent_export(latent_export)
    require_current_main_method_export(manifest)
    required_ids = set(pairs.left_id.astype(str)) | set(pairs.right_id.astype(str))
    conventional = load_conventional_spectra(clean_data, required_ids)
    pairs, coverage = common_support(pairs, conventional, latent)
    for split in ("train", "valid", "test"):
        labels = pairs[pairs.split.eq(split)].label.to_numpy(dtype=np.int8)
        select_metric(labels)
    output.mkdir(parents=True, exist_ok=True)
    coverage_path = output / "rq1_common_support.json"
    if getattr(args, "refresh_latent", False) and coverage_path.is_file():
        previous_coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        comparable_fields = (
            "common_code_count", "common_pairs_by_split", "validation_balance_dropped",
        )
        if any(previous_coverage.get(field) != coverage.get(field) for field in comparable_fields):
            raise RuntimeError(
                "Cannot reuse conventional RQ1 rows: the replacement latent export changed "
                "the common-support pair universe. Use a new --output-dir and rerun all rows."
            )
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    # ``conventional`` intentionally retains every raw record so that
    # ``common_support`` can report graph coverage.  Some records lack a view
    # (for example an empty CFG), therefore never materialize a view dictionary
    # from all raw records.  Restrict it to the selected common-support IDs
    # first; every remaining code is guaranteed to have every required view.
    common_ids = set(pairs.left_id.astype(str)) | set(pairs.right_id.astype(str))
    conventional_common = {
        code_id: views for code_id, views in conventional.items() if code_id in common_ids
    }
    spectra_by_graph = {"latent": {code_id: values for code_id, values in latent.items() if code_id in common_ids}}
    spectra_by_graph.update({
        graph: {code_id: views[graph] for code_id, views in conventional_common.items()}
        for graph in CONVENTIONAL_GRAPH_TYPES
    })
    result_path = output / "rq1_table_rows.csv"
    completed = pd.read_csv(result_path) if result_path.is_file() and args.resume else pd.DataFrame()
    if not completed.empty and "Method" in completed:
        completed = completed.drop_duplicates(subset=["Method"], keep="last").reset_index(drop=True)
        if getattr(args, "refresh_latent", False):
            # The direct classifier row and every learned-latent spectrum row
            # are tied to the archive fingerprint. Fixed-graph rows are safe
            # to retain only after the common-support equality check above.
            completed = completed.loc[~completed.Method.str.startswith("SPECTRA-Siam")].reset_index(drop=True)
    rows = [] if completed.empty else completed.to_dict("records")
    completed_methods = set(completed.Method) if not completed.empty else set()
    direct = direct_spectra_siam_row(dataset, latent_export)
    if direct is not None and direct["Method"] not in completed_methods:
        rows.append(direct)

    selected_graphs = tuple(getattr(args, "graphs", None) or ALL_GRAPH_TYPES)
    selected_learners = tuple(getattr(args, "learners", None) or ("No Train", "RF", "LR", "SNN"))
    skip_snn_graphs = set(getattr(args, "skip_snn_graphs", None) or ())
    scheduled = [
        (graph, learner)
        for graph in ALL_GRAPH_TYPES if graph in selected_graphs
        for learner in ("No Train", "RF", "LR", "SNN")
        if learner in selected_learners and not (learner == "SNN" and graph in skip_snn_graphs)
    ]
    progress = tqdm(total=len(scheduled), desc=f"{DISPLAY_DATASETS[dataset]} RQ1", unit="row")
    for graph_index, graph in enumerate(ALL_GRAPH_TYPES):
        if graph not in selected_graphs:
            continue
        matrix, splits = matrices_and_splits(pairs, spectra_by_graph[graph])
        primary = select_metric(splits["valid"].labels)
        shared = {
            "ValidationSelectionMetric": primary,
            "TrainPairs": len(splits["train"].labels), "ValidPairs": len(splits["valid"].labels), "TestPairs": len(splits["test"].labels),
        }
        for learner in ("No Train", "RF", "LR", "SNN"):
            if learner not in selected_learners:
                continue
            if learner == "SNN" and graph in skip_snn_graphs:
                continue
            method = f"{'SPECTRA-Siam Spectrum' if graph == 'latent' else graph.upper()} + {learner}"
            if method in completed_methods:
                progress.update(1)
                continue
            progress.set_description(f"{DISPLAY_DATASETS[dataset]}: {method}")
            started = time.perf_counter()
            if learner == "No Train":
                scores = pss_scores(pairs, spectra_by_graph[graph])
                threshold = choose_threshold(splits["valid"].labels, scores["valid"], primary)
                valid, test = metrics(splits["valid"].labels, scores["valid"], threshold), metrics(splits["test"].labels, scores["test"], threshold)
                details = {**shared, "TrainableParameters": 0, "Device": "cpu"}
            elif learner in {"RF", "LR"}:
                tag = f"{DISPLAY_DATASETS[dataset]}: {method}"
                train_x = pair_features(matrix, splits["train"].left, splits["train"].right, desc=f"{tag} [train features]")
                valid_x = pair_features(matrix, splits["valid"].left, splits["valid"].right, desc=f"{tag} [valid features]")
                test_x = pair_features(matrix, splits["test"].left, splits["test"].right, desc=f"{tag} [test features]")
                if learner == "RF":
                    tqdm.write(f"{tag}: fitting {args.rf_trees} trees on {len(train_x):,} pairs ({args.rf_jobs} workers)...")
                    classifier = RandomForestClassifier(n_estimators=args.rf_trees, max_depth=args.rf_max_depth, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=args.rf_jobs, random_state=args.seed + graph_index, verbose=2)
                else:
                    scaler = StandardScaler()
                    train_x, valid_x, test_x = scaler.fit_transform(train_x), scaler.transform(valid_x), scaler.transform(test_x)
                    classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed + graph_index)
                classifier.fit(train_x, splits["train"].labels)
                if learner == "RF":
                    tqdm.write(f"{tag}: fit done, scoring {len(valid_x) + len(test_x):,} pairs...")
                valid_scores, test_scores = classifier.predict_proba(valid_x)[:, 1], classifier.predict_proba(test_x)[:, 1]
                threshold = choose_threshold(splits["valid"].labels, valid_scores, primary)
                valid, test = metrics(splits["valid"].labels, valid_scores, threshold), metrics(splits["test"].labels, test_scores, threshold)
                details = {**shared, "TrainableParameters": np.nan, "Device": "cpu"}
                del train_x, valid_x, test_x, classifier
            else:
                valid_scores, test_scores, threshold, snn_details = run_snn(matrix, splits, args, args.seed + graph_index)
                valid, test = metrics(splits["valid"].labels, valid_scores, threshold), metrics(splits["test"].labels, test_scores, threshold)
                details = {**shared, **snn_details}
            rows.append(row(dataset, graph, learner, test, valid, details, time.perf_counter() - started))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            completed_methods.add(method)
            progress.update(1)
            gc.collect()
    progress.close()
    results = pd.DataFrame(rows)
    results["_order"] = pd.Categorical(results.Method, categories=METHOD_ORDER, ordered=True)
    results = results.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    results.to_csv(result_path, index=False)
    return results, pairs


def add_predict_all_clone(dataset: str, results: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    # Resume-safe: an interrupted/restarted aggregate must never duplicate the
    # deterministic control row.
    results = results[results.Method.ne("Predict All Clone")].copy()
    test = pairs[pairs.split.eq("test")]
    labels = test.label.to_numpy(dtype=np.int8)
    baseline = metrics(labels, np.ones(len(labels), dtype=np.float64), 0.5)
    item = {
        "Dataset": DISPLAY_DATASETS[dataset], "Graph": "-", "Method": "Predict All Clone",
        "P": baseline["P"], "R": baseline["R"], "F1": baseline["F1"], "Acc": baseline["Acc"],
        "BestValidP": np.nan, "BestValidR": np.nan, "BestValidF1": np.nan, "BestValidAcc": np.nan, "Threshold": 0.5,
        "ValidationSelectionMetric": "-", "TrainPairs": np.nan, "ValidPairs": np.nan, "TestPairs": len(test), "TrainableParameters": 0, "Device": "cpu", "RuntimeSeconds": 0.0, "RuntimeMinutes": 0.0,
    }
    return pd.concat((results, pd.DataFrame([item])), ignore_index=True)


def write_paper_files(all_results: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    all_results = all_results.drop_duplicates(subset=["Dataset", "Method"], keep="last").reset_index(drop=True)
    all_results.to_csv(output / "rq1_all_table_rows.csv", index=False, float_format="%.6f")
    values = all_results.pivot(index="Method", columns="Dataset", values=["P", "R", "F1", "Acc"])
    values = values.reindex(METHOD_ORDER).dropna(how="all")
    values.to_csv(output / "rq1_paper_table_values.csv", float_format="%.6f")
    lines = ["% Method & BigCloneBench P R F1 Acc & AtCoder P R F1 Acc"]
    for method, values_row in values.iterrows():
        cells = [method]
        for dataset in ("BigCloneBench", "AtCoder"):
            for metric in ("P", "R", "F1", "Acc"):
                value = values_row.get((metric, dataset), np.nan)
                cells.append("--" if pd.isna(value) else f"{value:.4f}")
        lines.append(" & ".join(cells) + r" \\")
    (output / "rq1_paper_table_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_paper_figure(all_results, output / "rq1_paper_table.png")


def write_paper_figure(all_results: pd.DataFrame, path: Path) -> None:
    """Render the exact paper rows as a compact reviewable PNG."""
    import matplotlib.pyplot as plt

    indexed = all_results.set_index(["Method", "Dataset"])
    rows = []
    for method in METHOD_ORDER:
        if method not in set(all_results.Method):
            continue
        values = [method]
        for dataset in ("BigCloneBench", "AtCoder"):
            for metric in ("P", "R", "F1", "Acc"):
                try:
                    value = indexed.loc[(method, dataset), metric]
                    if isinstance(value, pd.Series):
                        value = value.iloc[-1]
                except KeyError:
                    value = np.nan
                values.append("--" if pd.isna(value) else f"{float(value):.3f}")
        rows.append(values)

    columns = [
        "Method", "BCB P", "BCB R", "BCB F1", "BCB Acc",
        "AtCoder P", "AtCoder R", "AtCoder F1", "AtCoder Acc",
    ]
    figure, axis = plt.subplots(figsize=(16, max(7, 0.44 * len(rows) + 1.6)))
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.4)
        if row_index == 0:
            cell.set_facecolor("#dedede")
            cell.set_text_props(weight="bold")
        elif column_index == 0:
            cell.set_text_props(ha="left")
        if row_index > 0 and rows[row_index - 1][0] in {
            "SPECTRA-Siam", "SPECTRA-Siam Spectrum + No Train", "AST + No Train",
            "CFG + No Train", "DDG + No Train", "CPG + No Train", "Predict All Clone",
        }:
            cell.set_linewidth(0.9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def default_run_args(*, resume: bool = True, device: str = "auto") -> argparse.Namespace:
    return argparse.Namespace(
        resume=resume, seed=42, rf_trees=200, rf_max_depth=16, rf_jobs=2,
        snn_epochs=4, snn_patience=3, snn_batch_size=8192, device=device,
        graphs=None, learners=None, skip_snn_graphs=None, refresh_latent=False,
    )


def run_default_dataset(
    dataset: str,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    resume: bool = True,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every RQ1 row for one dataset using the repository's standard paths."""
    if dataset not in DEFAULT_DATASETS:
        raise ValueError(f"Unknown RQ1 dataset {dataset!r}; choose one of {tuple(DEFAULT_DATASETS)}")
    paths = DEFAULT_DATASETS[dataset]
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Default {dataset} {label} path does not exist: {path}")
    args = default_run_args(resume=resume, device=device)
    dataset_output = output_dir / dataset
    results, pairs = run_dataset(
        dataset, paths["clean_data"], paths["latent_export"], dataset_output, args,
    )
    results = add_predict_all_clone(dataset, results, pairs)
    results.to_csv(dataset_output / "rq1_table_rows.csv", index=False, float_format="%.6f")
    return results, pairs


def run_default_all(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR, resume: bool = True, device: str = "auto"
) -> pd.DataFrame:
    """Run AtCoder and XGLUE and write the complete two-dataset paper table."""
    frames = []
    for dataset in ("atcoder", "xglue"):
        print("\n" + "=" * 100)
        print(f"RQ1 complete table: {DISPLAY_DATASETS[dataset]}")
        print("=" * 100)
        result, _ = run_default_dataset(
            dataset, output_dir=output_dir, resume=resume, device=device,
        )
        frames.append(result)
    combined = pd.concat(frames, ignore_index=True)
    write_paper_files(combined, output_dir)
    print(combined[["Dataset", "Method", "P", "R", "F1", "Acc"]].to_string(index=False))
    print("\nComplete RQ1 outputs:", output_dir)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atcoder-clean-data", type=Path)
    parser.add_argument("--atcoder-latent-export", type=Path)
    parser.add_argument("--xglue-clean-data", type=Path)
    parser.add_argument("--xglue-latent-export", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=16)
    parser.add_argument("--rf-jobs", type=int, default=2)
    parser.add_argument("--snn-epochs", type=int, default=4)
    parser.add_argument("--snn-patience", type=int, default=3)
    parser.add_argument("--snn-batch-size", type=int, default=8192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--graphs", nargs="+", choices=tuple(ALL_GRAPH_TYPES),
        help="Run only these graph representations (default: all).",
    )
    parser.add_argument(
        "--learners", nargs="+", choices=("No Train", "RF", "LR", "SNN"),
        help="Run only these downstream learners (default: all).",
    )
    parser.add_argument(
        "--skip-snn-graphs", nargs="+", choices=tuple(ALL_GRAPH_TYPES), default=(),
        help="Do not run SNN for these graph views; useful when compatible SNN rows come from Kaggle.",
    )
    parser.add_argument(
        "--refresh-latent", action="store_true",
        help="With --resume, replace only direct/latent SPECTRA-Siam rows after a fresh compatible latent export.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    requested = ((args.atcoder_clean_data, args.atcoder_latent_export), (args.xglue_clean_data, args.xglue_latent_export))
    if not any(clean and export for clean, export in requested):
        parser.error("Supply a matching clean-data directory and latent export for AtCoder, XGLUE, or both")
    for clean, export in requested:
        if (clean is None) != (export is None):
            parser.error("Each selected dataset needs both --*-clean-data and --*-latent-export")
    return args


def main() -> None:
    args = parse_args()
    all_results = []
    configurations = (
        ("atcoder", args.atcoder_clean_data, args.atcoder_latent_export),
        ("xglue", args.xglue_clean_data, args.xglue_latent_export),
    )
    for dataset, clean_data, latent_export in configurations:
        if clean_data is None:
            continue
        result, pairs = run_dataset(dataset, clean_data, latent_export, args.output_dir / dataset, args)
        result = add_predict_all_clone(dataset, result, pairs)
        result.to_csv(args.output_dir / dataset / "rq1_table_rows.csv", index=False, float_format="%.6f")
        all_results.append(result)
    combined = pd.concat(all_results, ignore_index=True)
    write_paper_files(combined, args.output_dir)
    print(combined[["Dataset", "Method", "P", "R", "F1", "Acc"]].to_string(index=False))
    print("Paper-ready files:", args.output_dir / "rq1_paper_table_values.csv")


if __name__ == "__main__":
    main()
