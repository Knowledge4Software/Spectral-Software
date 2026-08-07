import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (
    DEFAULT_BASELINE_OUTPUT_DIR,
    DEFAULT_CLEAN_DATA_DIR,
    GRAPH_TYPES,
    best_f1_threshold,
    load_code_vectors,
    load_pairs,
    maybe_sample_train,
    metrics_at_threshold,
    result_row,
    save_results,
    split_pair_matrices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGLUE spectral Logistic Regression baseline.")
    parser.add_argument("--clean-data-dir", type=Path, default=DEFAULT_CLEAN_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASELINE_OUTPUT_DIR / "lr")
    parser.add_argument("--graph-types", default=",".join(GRAPH_TYPES))
    parser.add_argument("--k-eigen", type=int, default=64)
    parser.add_argument("--include-graph-stats", action="store_true")
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_types = [item.strip().lower() for item in args.graph_types.split(",") if item.strip()]
    pairs = load_pairs(args.clean_data_dir)
    print("All pairs:", len(pairs))
    print(pairs.groupby(["split", "label"]).size())

    rows = []
    for graph_type in graph_types:
        print("\n" + "=" * 80)
        print(f"Graph: {graph_type.upper()}")
        vectors = load_code_vectors(args.clean_data_dir, graph_type, args.k_eigen, args.include_graph_stats)
        data = split_pair_matrices(pairs, vectors, ("train", "valid", "test"))
        x_train, y_train = data["train"]
        x_valid, y_valid = data["valid"]
        x_test, y_test = data["test"]
        x_fit, y_fit = maybe_sample_train(x_train, y_train, args.max_train_pairs, args.seed)

        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_fit)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
            random_state=args.seed,
        )
        model.fit(x_fit, y_fit)
        valid_scores = model.predict_proba(x_valid)[:, 1]
        test_scores = model.predict_proba(x_test)[:, 1]
        threshold = best_f1_threshold(y_valid, valid_scores)
        valid_metrics = metrics_at_threshold(y_valid, valid_scores, threshold)
        test_metrics = metrics_at_threshold(y_test, test_scores, threshold)
        rows.append(result_row(graph_type, "LR", test_metrics, valid_metrics))

    save_results(rows, args.output_dir / "xglue_lr_results.csv")


if __name__ == "__main__":
    main()
