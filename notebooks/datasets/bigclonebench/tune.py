"""Run repeatable spectral-threshold tuning for a built BigCloneBench variant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.run_pipeline_helpers import run_bcb_metric_tuning
from spectral_code.utils.pipeline_cli import run_spectral_tuning_cli


VARIANTS = {
    "type1": ("1", "1"),
    "type2": ("2", "2"),
    "type3_moderate": ("3", "3/moderate"),
    "type3_strong": ("3", "3/strong"),
    "type3_very_strong": ("3", "3/very_strong"),
    "type4": ("4", "4"),
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="type4")
    parser.add_argument("-h", "--help", action="store_true")
    args, tuning_args = parser.parse_known_args()
    if args.help:
        print("BCB variant: --variant {" + ",".join(VARIANTS) + "} (default: type4)\n")
        tuning_args = ["--help"]
    clone_type, output_variant = VARIANTS[args.variant]
    sys.argv = [sys.argv[0], *tuning_args]
    run_spectral_tuning_cli(
        description=f"Tune BCB {args.variant} spectral metric thresholds.",
        output_folder=f"outputs/bcb/{output_variant}",
        runner=lambda: run_bcb_metric_tuning(
            clone_type,
            metrics=["pss"],
            stage_name="tune_pss_wasserstein",
            variant=output_variant,
        ),
    )


if __name__ == "__main__":
    main()
