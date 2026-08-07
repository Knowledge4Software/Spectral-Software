"""Small shared command-line helpers for dataset pipeline entry points."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PipelineStages = Sequence[tuple[str, str]]


def run_dataset_section(
    dataset: str,
    variant: str | None,
    section: str,
    run_dir: Path,
) -> None:
    """Run a shared dataset section from a thin dataset-specific entry point."""
    # Section imports load the graph/spectral stack. Keep CLI-only commands,
    # including ``run_all.py --help``, responsive until a section is selected.
    from spectral_code.evaluation.pipeline_section_runner import (
        SectionConfig,
        run_clean_data_export_section,
        run_dataset_preparation_section,
        run_graph_extraction_section,
        run_spectral_feature_section,
    )

    section_runners: dict[str, Callable[[SectionConfig], None]] = {
        "prepare": run_dataset_preparation_section,
        "graphs": run_graph_extraction_section,
        "spectra": run_spectral_feature_section,
        "export": run_clean_data_export_section,
    }
    try:
        runner = section_runners[section]
    except KeyError as exc:
        available = ", ".join(sorted(section_runners))
        raise ValueError(f"Unknown section '{section}'. Choose one of: {available}.") from exc
    runner(SectionConfig(dataset=dataset, variant=variant, run_dir=run_dir))


def run_numbered_pipeline(
    stages: PipelineStages,
    *,
    description: str,
    completion_message: str,
    run_dir: Path,
    argv: Sequence[str] | None = None,
) -> None:
    """Run an inclusive range of numbered scripts from a pipeline directory."""
    parser = argparse.ArgumentParser(description=description)
    stage_ids = [stage for stage, _ in stages]
    parser.add_argument("--start-at", choices=stage_ids, default=stage_ids[0])
    parser.add_argument("--stop-after", choices=stage_ids, default=stage_ids[-1])
    args = parser.parse_args(argv)

    selected = [
        (stage, script)
        for stage, script in stages
        if int(args.start_at) <= int(stage) <= int(args.stop_after)
    ]
    if not selected:
        raise ValueError("--start-at must be earlier than or equal to --stop-after.")

    for stage, script in selected:
        script_path = run_dir / script
        print(f"\n[*] Running stage {stage}: {script_path}")
        subprocess.run([sys.executable, str(script_path)], cwd=str(run_dir), check=True)

    print(f"\n[+] {completion_message}")


def run_spectral_tuning_cli(
    *,
    description: str,
    output_folder: str,
    runner: Callable[[], None],
    metrics: Sequence[str] = ("pss",),
    allow_pair_score_csv: bool = True,
    argv: Sequence[str] | None = None,
) -> None:
    """Parse common tuning options, configure the environment, and run tuning."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--k-values",
        default=os.environ.get("TUNING_K_VALUES", "full"),
        help="Comma-separated eigenvalue counts, for example: full or 25,50,full.",
    )
    parser.add_argument(
        "--n-samples",
        default=os.environ.get("TUNING_N_SAMPLES", ""),
        help="Optional pair sample size. Leave empty to use all available pairs.",
    )
    parser.add_argument(
        "--optimize-for",
        default=os.environ.get("TUNING_OPTIMIZE_FOR", "f1"),
        choices=("accuracy", "precision", "recall", "f1"),
    )
    if allow_pair_score_csv:
        parser.add_argument(
            "--no-pair-score-csv",
            action="store_true",
            help="Disable the large pair-level score CSV.",
        )
    args = parser.parse_args(argv)

    os.environ["TUNING_K_VALUES"] = args.k_values
    os.environ["TUNING_OPTIMIZE_FOR"] = args.optimize_for
    if allow_pair_score_csv:
        os.environ["TUNING_SAVE_PAIR_SCORES"] = "0" if args.no_pair_score_csv else "1"
    if args.n_samples.strip():
        os.environ["TUNING_N_SAMPLES"] = args.n_samples.strip()

    print("[*] Metrics:", ", ".join(metrics))
    print("[*] Output folder:", output_folder)
    if allow_pair_score_csv:
        print("[*] Pair-level score CSV:", "disabled" if args.no_pair_score_csv else "enabled")
    runner()
