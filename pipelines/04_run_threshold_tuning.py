import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.project_paths import ensure_dirs
from spectral_code.evaluation.tuning import run_fast_grid_search
from spectral_code.utils.dataset_paths import bcb_type_dir, output_root_for


@dataclass(frozen=True)
class TuningExperiment:
    name: str
    graph_types: list[str]
    metrics: list[str]
    k_values: list[int | None]
    optimize_for: str = "accuracy"


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    if raw.strip().lower() == "full":
        return None
    return int(raw)


def _parse_csv(raw: str | None, default: list[str]) -> list[str]:
    if raw is None or raw.strip() == "":
        return default
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    return values or default


def _parse_k_values(raw: str | None, default: list[int | None]) -> list[int | None]:
    if raw is None or raw.strip() == "":
        return default
    values = [_parse_optional_int(value) for value in raw.split(",") if value.strip()]
    return values or default


def main():
    ensure_dirs()

    clone_type = os.getenv("BCB_CLONE_TYPE", "1").strip()
    features_db_path = Path(
        os.getenv(
            "FEATURES_DB_PATH",
            str(output_root_for("bcb", clone_type) / "spectral_features" / "spectral_features_manifest.json"),
        )
    )
    bcb_data_dir = Path(
        os.getenv(
            "BCB_DATA_DIR",
            str(bcb_type_dir(clone_type))
        )
    )
    n_samples = _parse_optional_int(os.getenv("TUNING_N_SAMPLES", "full"))
    optimize_for = os.getenv("TUNING_OPTIMIZE_FOR", "f1").strip().lower()
    graph_types = _parse_csv(os.getenv("TUNING_GRAPH_TYPES"), ["ast", "cfg", "ddg", "pdg", "cpg"])
    metrics = _parse_csv(os.getenv("TUNING_METRICS"), ["pss"])
    k_values = _parse_k_values(os.getenv("TUNING_K_VALUES"), [None])
    metric_label = "-".join(metrics)
    k_label = "-".join("full" if k is None else str(k) for k in k_values)

    experiment = TuningExperiment(
        name=f"type{clone_type}_all_graphs_k{k_label}_{metric_label}_{optimize_for}",
        graph_types=graph_types,
        metrics=metrics,
        k_values=k_values,
        optimize_for=optimize_for,
    )

    # Optional experiments. Uncomment one of these if you want a narrower or wider run.
    # experiment = TuningExperiment(
    #     name="type1_ast_full_pss",
    #     graph_types=["ast"],
    #     metrics=["pss"],
    #     k_values=[None],
    # )

    # experiment = TuningExperiment(
    #     name="type1_ast_full_all_metrics",
    #     graph_types=["ast"],
    #     metrics=["pss", "heat_kernel", "wasserstein", "jensenshannon"],
    #     k_values=[None],
    # )

    # experiment = TuningExperiment(
    #     name="type1_ast_pss_k_sweep",
    #     graph_types=["ast"],
    #     metrics=["pss"],
    #     k_values=[25, 50, 100, None],
    # )

    print("[*] Starting threshold tuning...")
    print(f"[*] Features: {features_db_path}")
    print(f"[*] Data Dir: {bcb_data_dir}")
    print(f"[*] Sample Count: {'Full' if n_samples is None else n_samples}")
    print(f"[*] Experiment: {experiment.name}")

    results = run_fast_grid_search(
        str(features_db_path),
        str(bcb_data_dir),
        n_samples=n_samples,
        optimize_for=experiment.optimize_for,
        k_values=experiment.k_values,
        metrics=experiment.metrics,
        graph_types=experiment.graph_types,
        out_filename=f"trained_{experiment.name}.json",
    )

    if results:
        print("\n[+] Summary:")
        print(f"{'Graph':<8} {'Threshold':>12} {'Accuracy':>10} {'F1':>10}")
        print("-" * 44)
        for row in results:
            print(
                f"{row['graph_type'].upper():<8} "
                f"{row['best_threshold']:>12.6f} "
                f"{row.get('train_accuracy', row['best_metric']):>10.4f} "
                f"{row.get('train_f1', 0.0):>10.4f}"
            )

    print(f"\n[+] Tuning completed. Results saved to trained_{experiment.name}.json.")

if __name__ == "__main__":
    main()
