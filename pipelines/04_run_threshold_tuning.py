import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.project_paths import SPECTRAL_FEATURES_DIR, ensure_dirs
from spectral_code.evaluation.tuning import run_fast_grid_search


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


def main():
    ensure_dirs()

    features_db_path = Path(
        os.getenv(
            "FEATURES_DB_PATH",
            str(SPECTRAL_FEATURES_DIR / "spectral_features_manifest.json"),
        )
    )
    bcb_data_dir = Path(
        os.getenv("BCB_DATA_DIR", str(PROJECT_ROOT / "bench_data" / "bcb_full_type1"))
    )
    n_samples = _parse_optional_int(os.getenv("TUNING_N_SAMPLES", "full"))

    # Default run: compare all graph representations with Full eigenvalues + PSS.
    experiment = TuningExperiment(
        name="type1_all_graphs_full_pss",
        graph_types=["ast", "cfg", "ddg", "pdg", "cpg"],
        metrics=["pss"],
        k_values=[None],
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
