import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[7]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.run_pipeline_helpers import run_bcb_metric_tuning


CLONE_TYPE = "3"
VARIANT = "3/strong"
METRICS = [
    "pss",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune BCB Type-3 Strong spectral metric thresholds.")
    parser.add_argument("--k-values", default=os.environ.get("TUNING_K_VALUES", "full"))
    parser.add_argument("--n-samples", default=os.environ.get("TUNING_N_SAMPLES", ""))
    parser.add_argument("--optimize-for", default=os.environ.get("TUNING_OPTIMIZE_FOR", "f1"), choices=["accuracy", "precision", "recall", "f1"])
    parser.add_argument("--no-pair-score-csv", action="store_true")
    args = parser.parse_args()

    os.environ["TUNING_K_VALUES"] = args.k_values
    os.environ["TUNING_OPTIMIZE_FOR"] = args.optimize_for
    os.environ["TUNING_SAVE_PAIR_SCORES"] = "0" if args.no_pair_score_csv else "1"
    if args.n_samples.strip():
        os.environ["TUNING_N_SAMPLES"] = args.n_samples.strip()

    print("[*] Metrics:", ", ".join(METRICS))
    print("[*] Output folder: outputs/bcb/type3/strong")
    print("[*] Result JSON includes thresholds, accuracy, precision, recall, f1, AUC, and balanced-fold variance.")
    print("[*] Pair-level score CSV:", "disabled" if args.no_pair_score_csv else "enabled")
    run_bcb_metric_tuning(CLONE_TYPE, metrics=METRICS, stage_name="04_tune_pss_wasserstein", variant=VARIANT)
