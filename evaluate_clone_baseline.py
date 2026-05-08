from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectral_code.config import PipelineConfig
from spectral_code.evaluation import (
    BigCloneBenchLoader,
    SpectralCloneBaseline,
    evaluate_binary,
    evaluate_by_type,
)
from spectral_code.evaluation.baseline import save_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spectral clone-detection baseline on BCB")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--type-labels", type=str, default=None,
                   help="Optional official type-label file (jsonl/tsv).")
    p.add_argument("--graph-type", type=str, choices=["ast", "cfg"], default="ast")
    p.add_argument("--spectral-mode", type=str, default="normalized_laplacian",
                   choices=["laplacian", "normalized_laplacian"])
    p.add_argument("--lang", type=str, default="java")
    p.add_argument("--eigen-solver", type=str, default="auto")
    p.add_argument("--k-eigen", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--tune-threshold", action="store_true",
                   help="Tune threshold on validation split using F1.")
    p.add_argument("--sample-per-type", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="baseline_report.json")
    return p


def main() -> int:
    args = build_parser().parse_args()

    loader = BigCloneBenchLoader(args.data_dir, type_labels_path=args.type_labels)

    config = PipelineConfig(
        graph_type=args.graph_type,
        spectral_mode=args.spectral_mode,
        eigen_solver=args.eigen_solver,
        k_eigen=args.k_eigen,
    )

    try:
        model = SpectralCloneBaseline(config=config, lang=args.lang, threshold=args.threshold)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 2

    # train_pairs = loader.get_pairs("train")
    valid_pairs = loader.sample_pairs("valid", n=2000, seed=args.seed)
    test_pairs = loader.sample_pairs("test", n=5000, seed=args.seed)

    tuned = None
    if args.tune_threshold:
        best_t, valid_best = model.tune_threshold(valid_pairs)
        tuned = {
            "best_threshold": best_t,
            "valid_metrics": valid_best,
        }

    binary_test = evaluate_binary(model, test_pairs)
    sampled = loader.stratified_sample("test", per_type=args.sample_per_type, seed=args.seed)
    per_type = evaluate_by_type(model, sampled)

    payload = {
        "config": {
            "data_dir": args.data_dir,
            "graph_type": args.graph_type,
            "spectral_mode": args.spectral_mode,
            "lang": args.lang,
            "eigen_solver": args.eigen_solver,
            "k_eigen": args.k_eigen,
            "threshold": model.threshold,
            "tune_threshold": args.tune_threshold,
            "sample_per_type": args.sample_per_type,
            "seed": args.seed,
        },
        "tuning": tuned,
        "binary_test": binary_test,
        "per_type": per_type,
    }

    print("\n=== Spectral Clone Baseline Report ===")
    print(f"Graph type      : {args.graph_type}")
    print(f"Spectral mode   : {args.spectral_mode}")
    print(f"Language        : {args.lang}")
    print(f"Threshold       : {model.threshold:.4f}")

    if tuned is not None:
        print("\n[Validation tuning]")
        print(json.dumps(tuned, indent=2, ensure_ascii=False))

    print("\n[Binary test metrics]")
    print(json.dumps(binary_test, indent=2, ensure_ascii=False))

    print("\n[Per-type success rates on sampled positive test pairs]")
    print(json.dumps(per_type, indent=2, ensure_ascii=False))

    save_report(args.output, payload)
    print(f"\nSaved JSON report to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
