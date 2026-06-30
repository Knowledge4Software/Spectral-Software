import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.bcb_cross_type_generalization import run_from_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune thresholds on BCB Types 1+2+3/all and evaluate the fixed thresholds on Type 4."
    )
    parser.add_argument(
        "--graph-types",
        default=os.environ.get("BCB_CROSS_GRAPH_TYPES", "ast,cfg,ddg,pdg,cpg"),
        help="Comma-separated graph layers, e.g. ast,cfg,ddg,pdg,cpg.",
    )
    parser.add_argument(
        "--metrics",
        default=os.environ.get("BCB_CROSS_METRICS", "pss"),
        help="Comma-separated spectral similarity metrics.",
    )
    parser.add_argument(
        "--k-values",
        default=os.environ.get("BCB_CROSS_K_VALUES", os.environ.get("TUNING_K_VALUES", "full")),
        help="Comma-separated eigenvalue counts. Use 'full' for the complete spectrum.",
    )
    parser.add_argument(
        "--optimize-for",
        default=os.environ.get("BCB_CROSS_OPTIMIZE_FOR", os.environ.get("TUNING_OPTIMIZE_FOR", "f1")),
        choices=["accuracy", "precision", "recall", "f1"],
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("BCB_CROSS_SEED", "42")))
    args = parser.parse_args()

    os.environ["BCB_CROSS_GRAPH_TYPES"] = args.graph_types
    os.environ["BCB_CROSS_METRICS"] = args.metrics
    os.environ["BCB_CROSS_K_VALUES"] = args.k_values
    os.environ["BCB_CROSS_OPTIMIZE_FOR"] = args.optimize_for
    os.environ["BCB_CROSS_SEED"] = str(args.seed)

    print("[*] Train threshold on: BCB Type1 + Type2 + Type3/all")
    print("[*] Test fixed threshold on: BCB Type4")
    print("[*] Graph types:", args.graph_types)
    print("[*] Metrics:", args.metrics)
    print("[*] K values:", args.k_values)
    run_from_env()
