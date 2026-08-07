"""Build durable BigCloneBench data, graph, and spectral artifacts once.

The legacy per-variant ``run_pipeline`` folders remain available, but this is
the preferred entry point for normal work. It deliberately preserves
``clean_graphs`` and ``spectral_features`` for later tuning and clean exports.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.run_pipeline_helpers import (
    bcb_non_clone_defaults,
    bcb_positive_only_defaults,
    run_bcb_data_extraction,
    run_bcb_graph_extraction,
    run_bcb_spectral_feature_extraction,
)


@dataclass(frozen=True)
class VariantSpec:
    clone_type: str
    output_variant: str
    defaults: Callable[[], list[str]]


VARIANTS = {
    "type1": VariantSpec("1", "1", bcb_positive_only_defaults),
    "type2": VariantSpec("2", "2", bcb_positive_only_defaults),
    "type3_moderate": VariantSpec(
        "3", "3/moderate", lambda: bcb_positive_only_defaults(type3_min_similarity=0.50, type3_max_similarity=0.70)
    ),
    "type3_strong": VariantSpec(
        "3", "3/strong", lambda: bcb_positive_only_defaults(type3_min_similarity=0.70, type3_max_similarity=0.90)
    ),
    "type3_very_strong": VariantSpec(
        "3", "3/very_strong", lambda: bcb_positive_only_defaults(type3_min_similarity=0.90, type3_max_similarity=1.00)
    ),
    "type4": VariantSpec("4", "4", bcb_positive_only_defaults),
    "non_clone": VariantSpec("1", "non_clone", bcb_non_clone_defaults),
}
STAGES = ("data", "graphs", "spectra")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build durable BigCloneBench pipeline artifacts.")
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=tuple(VARIANTS),
        default=["type4", "non_clone"],
        help="Variants to build. Default: Type-4 plus shared non-clones.",
    )
    parser.add_argument(
        "--start-at",
        choices=STAGES,
        default="graphs",
        help="First stage. Use data only for a first-time dataset preparation.",
    )
    parser.add_argument("--stop-after", choices=STAGES, default="spectra")
    args = parser.parse_args()

    start = STAGES.index(args.start_at)
    stop = STAGES.index(args.stop_after)
    if start > stop:
        parser.error("--start-at must not be after --stop-after.")

    for variant_name in args.variant:
        spec = VARIANTS[variant_name]
        print(f"\n{'=' * 80}\nBCB build: {variant_name}\n{'=' * 80}")
        if start <= STAGES.index("data") <= stop:
            run_bcb_data_extraction(spec.clone_type, [], defaults=spec.defaults(), variant=spec.output_variant)
        if start <= STAGES.index("graphs") <= stop:
            run_bcb_graph_extraction(spec.clone_type, variant=spec.output_variant)
        if start <= STAGES.index("spectra") <= stop:
            run_bcb_spectral_feature_extraction(spec.clone_type, variant=spec.output_variant)


if __name__ == "__main__":
    main()
