from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import json
import random

import pandas as pd

from spectral_code.evaluation.bcb_dataset import ClonePair
from spectral_code.evaluation.notebook_helpers import (
    bcb_spec,
    load_pairs_for_spec,
    run_tuning_for_spec,
    save_dataframe_report,
    save_json_report,
    pair_stats_dataframe,
    semantic_spec,
    xglue_spec,
)
from spectral_code.evaluation.semantic_preparation import (
    default_semantic_prepared_dir,
    prepare_semantic_dataset,
    prepared_dataset_summary,
)
from spectral_code.utils.dataset_paths import bcb_type_dir, output_root_for, xglue_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PREPARED_FILES = ("data.jsonl", "train.txt")
JOERN_LANGUAGE_BY_SECTION = {
    ("bcb", None): "javasrc",
    ("semantic", "c"): "c",
    ("semantic", "cs"): "csharpsrc",
    ("semantic", "csharp"): "csharpsrc",
    ("semantic", "java"): "javasrc",
    ("semantic", "python"): "pythonsrc",
    ("xglue", None): "javasrc",
}
GRAPH_TYPES_BY_SECTION = {
    ("bcb", None): ["ast", "cfg", "ddg", "pdg", "cpg"],
    ("semantic", "c"): ["cfg", "ddg", "cpg"],
    ("semantic", "cs"): ["cfg", "ddg", "cpg"],
    ("semantic", "csharp"): ["cfg", "ddg", "cpg"],
    ("semantic", "java"): ["ast", "cfg", "ddg", "pdg", "cpg"],
    ("semantic", "python"): ["cfg", "ddg", "cpg"],
    ("xglue", None): ["ast", "cfg", "ddg", "pdg", "cpg"],
}


@dataclass(frozen=True)
class SectionConfig:
    dataset: str
    variant: str | None
    run_dir: Path

    @property
    def slug(self) -> str:
        if self.dataset == "bcb":
            variant = str(self.variant).strip().replace("\\", "/").strip("/")
            return "bcb_type" + variant.replace("/", "_")
        if self.dataset == "semantic":
            return f"semantic_{str(self.variant).strip().lower()}"
        return self.dataset

    @property
    def display_name(self) -> str:
        if self.dataset == "bcb":
            return f"BigCloneBench Type {self.variant}"
        if self.dataset == "semantic":
            return f"Semantic Benchmark ({self.variant})"
        return "XGLUE"

    @property
    def output_root(self) -> Path:
        if self.dataset == "bcb":
            return output_root_for("bcb", self.variant)
        if self.dataset == "semantic":
            return output_root_for("semantic_benchmark", str(self.variant).strip().lower())
        return output_root_for("xglue")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw.strip())


def _env_k_values(name: str, default: list[int | None] | None = None) -> list[int | None]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default or [None]
    values = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        values.append(None if item in {"full", "none", "all"} else int(item))
    return values or (default or [None])


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw.strip())


def _select_balanced_bcb_tuning_pairs(
    pairs: list[ClonePair],
    *,
    spec_key: str,
    seed: int,
) -> tuple[list[ClonePair], dict]:
    positives = [pair for pair in pairs if pair.label == 1]
    negatives = [pair for pair in pairs if pair.label == 0]
    metadata = {
        "enabled": False,
        "reason": "",
        "seed": seed,
        "input_pairs": len(pairs),
        "input_positive_pairs": len(positives),
        "input_negative_pairs": len(negatives),
        "selected_pairs": len(pairs),
        "selected_positive_pairs": len(positives),
        "selected_negative_pairs": len(negatives),
        "chunks": [],
    }

    if not spec_key.startswith("bcb_") or spec_key == "bcb_non_clone":
        metadata["reason"] = "not_a_bcb_clone_type"
        return pairs, metadata

    if not positives or not negatives:
        metadata["reason"] = "requires_both_positive_and_negative_pairs"
        return pairs, metadata

    rng = random.Random(seed)
    shuffled_positives = list(positives)
    shuffled_negatives = list(negatives)
    rng.shuffle(shuffled_positives)
    rng.shuffle(shuffled_negatives)

    chunks = []
    selected_pairs: list[ClonePair] = []

    if len(negatives) >= len(positives):
        pos_chunk = shuffled_positives
        neg_chunk = shuffled_negatives[:len(pos_chunk)]
        chunk_pairs = pos_chunk + neg_chunk
        rng.shuffle(chunk_pairs)
        selected_pairs.extend(chunk_pairs)
        chunks.append(
            {
                "chunk": 1,
                "positive_pairs": len(pos_chunk),
                "negative_pairs": len(neg_chunk),
                "unique_negative_pairs": len({(pair.left_id, pair.right_id) for pair in neg_chunk}),
                "total_pairs": len(chunk_pairs),
            }
        )
        strategy = "sample_non_clones_to_clone_count"
        selected_positive_pairs = len(pos_chunk)
        selected_negative_pair_evaluations = len(neg_chunk)
        unique_selected_negative_pairs = len(neg_chunk)
    else:
        chunk_size = len(shuffled_negatives)
        for chunk_index, start in enumerate(range(0, len(shuffled_positives), chunk_size), start=1):
            stop = min(start + chunk_size, len(shuffled_positives))
            pos_chunk = shuffled_positives[start:stop]
            if len(pos_chunk) == len(shuffled_negatives):
                neg_chunk = list(shuffled_negatives)
            else:
                neg_chunk = rng.sample(shuffled_negatives, len(pos_chunk))
            chunk_pairs = pos_chunk + neg_chunk
            rng.shuffle(chunk_pairs)
            selected_pairs.extend(chunk_pairs)
            chunks.append(
                {
                    "chunk": chunk_index,
                    "positive_pairs": len(pos_chunk),
                    "negative_pairs": len(neg_chunk),
                    "unique_negative_pairs": len({(pair.left_id, pair.right_id) for pair in neg_chunk}),
                    "total_pairs": len(chunk_pairs),
                }
            )
        strategy = "chunk_clones_against_reused_non_clone_pool"
        selected_positive_pairs = len(shuffled_positives)
        selected_negative_pair_evaluations = sum(chunk["negative_pairs"] for chunk in chunks)
        unique_selected_negative_pairs = len(shuffled_negatives)

    metadata.update(
        {
            "enabled": True,
            "reason": strategy,
            "selected_pairs": len(selected_pairs),
            "selected_positive_pairs": selected_positive_pairs,
            "selected_negative_pairs": selected_negative_pair_evaluations,
            "unique_selected_negative_pairs": unique_selected_negative_pairs,
            "dropped_positive_pairs": len(positives) - selected_positive_pairs,
            "dropped_negative_pairs": len(negatives) - unique_selected_negative_pairs,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
    )
    return selected_pairs, metadata


def _require_prepared_files(data_dir: Path) -> None:
    missing = [name for name in REQUIRED_PREPARED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Prepared dataset is missing required files in {data_dir}: {', '.join(missing)}")


def _run_python_script(
    relative_script_path: str,
    script_args: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> None:
    command = [sys.executable, str(PROJECT_ROOT / relative_script_path)]
    if script_args:
        command.extend(script_args)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else os.pathsep.join([str(PROJECT_ROOT), existing_pythonpath])
    )
    if env_overrides:
        env.update(env_overrides)

    print(f"\n[*] Running {relative_script_path}")
    subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=True)


def _prepare_bcb_dataset(config: SectionConfig) -> Path:
    clone_type = str(config.variant).strip()
    output_dir = bcb_type_dir(clone_type)
    _require_prepared_files(output_dir)
    print(f"[*] Using prepared BCB dataset: {output_dir}")
    return output_dir


def _prepare_semantic_dataset(config: SectionConfig) -> Path:
    def _is_current_semantic_prepared_dir(path: Path) -> bool:
        data_jsonl = path / "data.jsonl"
        if not data_jsonl.exists():
            return False
        try:
            with data_jsonl.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                return False
            record = json.loads(first_line)
        except Exception:
            return False
        return "lang" in record

    language = str(config.variant).strip()
    output_dir = default_semantic_prepared_dir(language)
    if output_dir.exists() and not _env_flag("SECTION_FORCE_PREPARE") and not _env_flag("SEMANTIC_FORCE_PREPARE"):
        try:
            _require_prepared_files(output_dir)
            if _is_current_semantic_prepared_dir(output_dir):
                print(f"[*] Reusing prepared semantic dataset: {output_dir}")
                return output_dir
            print(f"[*] Semantic prepared dataset is outdated, regenerating: {output_dir}")
        except FileNotFoundError:
            pass

    prepared = prepare_semantic_dataset(
        language=language,
        output_dir=output_dir,
        negative_ratio=_env_float("SEMANTIC_NEGATIVE_RATIO", 1.0),
        seed=_env_int("SEMANTIC_SEED", 42) or 42,
    )
    save_json_report(output_dir / "prepared_summary.json", prepared_dataset_summary(prepared))
    _require_prepared_files(output_dir)
    return output_dir


def _prepare_xglue_dataset() -> Path:
    data_dir = xglue_dir()
    _require_prepared_files(data_dir)
    print(f"[*] Using XGLUE dataset directory: {data_dir}")
    return data_dir


def prepare_dataset(config: SectionConfig) -> Path:
    if config.dataset == "bcb":
        return _prepare_bcb_dataset(config)
    if config.dataset == "semantic":
        return _prepare_semantic_dataset(config)
    if config.dataset == "xglue":
        return _prepare_xglue_dataset()
    raise ValueError(f"Unsupported dataset kind: {config.dataset}")


def run_full_pipeline_section(config: SectionConfig) -> None:
    data_dir = prepare_dataset(config)
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_dir = output_root / "reports"
    summary_dir.mkdir(parents=True, exist_ok=True)

    pipeline_env = {
        "BCB_DATA_FILE": str(data_dir / "data.jsonl"),
        "BCB_DATA_DIR": str(data_dir),
        "OUTPUT_DIR": str(output_root),
    }
    variant_key = str(config.variant).strip().lower() if config.variant is not None else None
    graph_types = GRAPH_TYPES_BY_SECTION.get((config.dataset, variant_key))
    if graph_types is None:
        graph_types = GRAPH_TYPES_BY_SECTION.get((config.dataset, None), ["ast", "cfg", "ddg", "pdg", "cpg"])
    base_layers = [g for g in graph_types if g != "cpg"]
    pipeline_env["PIPELINE_GRAPH_TYPES"] = ",".join(base_layers)
    pipeline_env["PIPELINE_BASE_LAYERS"] = ",".join(base_layers)
    pipeline_env["SPECTRAL_GRAPH_TYPES"] = ",".join(graph_types)
    if config.dataset == "semantic":
        joern_language = JOERN_LANGUAGE_BY_SECTION.get((config.dataset, variant_key))
        if joern_language:
            pipeline_env["JOERN_LANGUAGE"] = joern_language
    elif config.dataset == "bcb":
        pipeline_env["JOERN_LANGUAGE"] = "javasrc"
        pipeline_env["JOERN_USE_DIRECT_FRONTEND"] = os.getenv("JOERN_USE_DIRECT_FRONTEND", "1")
        pipeline_env["BCB_MAX_METHOD_LINES"] = os.getenv("BCB_MAX_METHOD_LINES", "2000")
        pipeline_env["JOERN_PARSE_CHUNK_SIZE"] = os.getenv("JOERN_PARSE_CHUNK_SIZE", "0")
        pipeline_env["JOERN_PARSE_MIN_CHUNK_SIZE"] = os.getenv("JOERN_PARSE_MIN_CHUNK_SIZE", "50")
        pipeline_env["JOERN_PARSE_INACTIVITY_TIMEOUT_SECONDS"] = os.getenv(
            "JOERN_PARSE_INACTIVITY_TIMEOUT_SECONDS",
            "0",
        )
    elif config.dataset == "xglue":
        pipeline_env["JOERN_LANGUAGE"] = "javasrc"
    if config.dataset == "bcb" and config.variant is not None:
        pipeline_env["BCB_CLONE_TYPE"] = str(config.variant)

    print(f"[*] Dataset: {config.display_name}")
    print(f"[*] Prepared Data: {data_dir}")
    print(f"[*] Output Root: {output_root}")

    spec = _dataset_spec(config)
    pairs = load_pairs_for_spec(
        spec,
        negative_ratio=_env_float("SEMANTIC_NEGATIVE_RATIO", 1.0),
    )
    pair_stats = pair_stats_dataframe(pairs)
    pair_stats_summary = {
        "dataset": config.display_name,
        "slug": config.slug,
        "total_pairs": int(len(pair_stats)),
        "positive_pairs": int((pair_stats["label"] == 1).sum()),
        "negative_pairs": int((pair_stats["label"] == 0).sum()),
        "avg_left_lines": float(pair_stats["left_lines"].mean()) if not pair_stats.empty else 0.0,
        "avg_right_lines": float(pair_stats["right_lines"].mean()) if not pair_stats.empty else 0.0,
        "avg_left_chars": float(pair_stats["left_chars"].mean()) if not pair_stats.empty else 0.0,
        "avg_right_chars": float(pair_stats["right_chars"].mean()) if not pair_stats.empty else 0.0,
    }
    save_dataframe_report(summary_dir / f"{config.slug}_pair_stats.csv", pair_stats)
    save_json_report(summary_dir / f"{config.slug}_pair_stats_summary.json", pair_stats_summary)

    _run_python_script("pipelines/01_extract_dataset.py", env_overrides=pipeline_env)
    _run_python_script("pipelines/02_build_graph_db.py", env_overrides=pipeline_env)
    _run_python_script("pipelines/03_extract_spectral_features.py", env_overrides=pipeline_env)

    summary = {
        "dataset": config.display_name,
        "slug": config.slug,
        "prepared_data_dir": str(data_dir),
        "output_root": str(output_root),
        "features_manifest": str(output_root / "spectral_features" / "spectral_features_manifest.json"),
        "timing_stats": str(output_root / "timing_stats.json"),
        "pair_stats_csv": str(summary_dir / f"{config.slug}_pair_stats.csv"),
        "pair_stats_summary_json": str(summary_dir / f"{config.slug}_pair_stats_summary.json"),
        "graph_types": graph_types,
        "pipeline_env": pipeline_env,
    }
    save_json_report(summary_dir / f"{config.slug}_pipeline_summary.json", summary)
    print(f"\n[+] Pipeline section finished for {config.display_name}.")


def _dataset_spec(config: SectionConfig):
    if config.dataset == "bcb":
        return bcb_spec(str(config.variant))
    if config.dataset == "semantic":
        return semantic_spec(str(config.variant))
    if config.dataset == "xglue":
        return xglue_spec()
    raise ValueError(f"Unsupported dataset kind: {config.dataset}")


def run_pss_wasserstein_tuning(config: SectionConfig) -> None:
    spec = _dataset_spec(config)
    summary_dir = spec.output_root / "reports"
    summary_dir.mkdir(parents=True, exist_ok=True)
    optimize_for = os.getenv("TUNING_OPTIMIZE_FOR", "f1").strip().lower()
    requested_n_samples = _env_int("TUNING_N_SAMPLES")
    n_samples = requested_n_samples
    graph_types_raw = os.getenv("TUNING_GRAPH_TYPES", "").strip()
    variant_key = str(config.variant).strip().lower() if config.variant is not None else None
    default_graph_types = GRAPH_TYPES_BY_SECTION.get((config.dataset, variant_key))
    if default_graph_types is None:
        default_graph_types = GRAPH_TYPES_BY_SECTION.get((config.dataset, None), ["ast", "cfg", "ddg", "pdg", "cpg"])
    graph_types = [item.strip().lower() for item in graph_types_raw.split(",") if item.strip()] or default_graph_types
    default_metrics = ["pss"]
    metrics_raw = os.getenv("TUNING_METRICS", ",".join(default_metrics)).strip()
    metrics = [item.strip().lower() for item in metrics_raw.split(",") if item.strip()] or default_metrics
    k_values = _env_k_values("TUNING_K_VALUES", [None])

    pairs = load_pairs_for_spec(
        spec,
        negative_ratio=_env_float("SEMANTIC_NEGATIVE_RATIO", 1.0),
    )
    bcb_pair_selection = {"enabled": False, "reason": "not_attempted"}
    if _env_flag("TUNING_BCB_BALANCED_NON_CLONE_SAMPLE", True):
        pairs, bcb_pair_selection = _select_balanced_bcb_tuning_pairs(
            pairs,
            spec_key=spec.key,
            seed=_env_int("TUNING_RANDOM_SEED", 42) or 42,
        )
        if bcb_pair_selection.get("enabled"):
            n_samples = None
            print(
                "[*] BCB tuning pair selection: "
                f"{bcb_pair_selection['selected_positive_pairs']:,} clone pairs + "
                f"{bcb_pair_selection['selected_negative_pairs']:,} non-clone evaluations "
                f"({bcb_pair_selection.get('unique_selected_negative_pairs', 0):,} unique) "
                f"across {bcb_pair_selection['chunk_count']:,} chunk(s)."
            )
            if requested_n_samples is not None:
                print(
                    "[!] TUNING_N_SAMPLES is ignored for balanced BCB clone-vs-non-clone tuning; "
                    "set TUNING_BCB_BALANCED_NON_CLONE_SAMPLE=0 to use raw sampling."
                )
    else:
        bcb_pair_selection = {
            "enabled": False,
            "reason": "disabled_by_TUNING_BCB_BALANCED_NON_CLONE_SAMPLE",
            "input_pairs": len(pairs),
        }
    save_json_report(summary_dir / f"{config.slug}_tuning_pair_selection.json", bcb_pair_selection)
    results = run_tuning_for_spec(
        spec=spec,
        pairs=pairs,
        metrics=metrics,
        graph_types=graph_types,
        k_values=k_values,
        n_samples=n_samples,
        optimize_for=optimize_for,
        out_filename=f"trained_{config.slug}_{optimize_for}_pss_wasserstein.json",
    )
    results_df = pd.DataFrame(results or [])
    if not results_df.empty:
        save_dataframe_report(summary_dir / f"{config.slug}_tuning_results.csv", results_df)
        printable_columns = [
            "graph_type",
            "metric",
            "best_threshold",
            "train_accuracy",
            "train_precision",
            "train_recall",
            "train_f1",
            "train_auc",
        ]
        printable = results_df[[col for col in printable_columns if col in results_df.columns]].copy()
        print("\n[+] Tuning metrics:")
        print(printable.to_string(index=False))

    summary = {
        "dataset": config.display_name,
        "slug": config.slug,
        "features_manifest": str(spec.features_manifest),
        "output_root": str(spec.output_root),
        "tuning_results_csv": str(summary_dir / f"{config.slug}_tuning_results.csv"),
        "n_samples": n_samples,
        "requested_n_samples": requested_n_samples,
        "bcb_pair_selection": bcb_pair_selection,
        "optimize_for": optimize_for,
        "graph_types": graph_types,
        "k_values": ["full" if value is None else value for value in k_values],
        "metrics": metrics,
        "results": results or [],
    }
    save_json_report(summary_dir / f"{config.slug}_tuning_summary.json", summary)
    print(f"\n[+] Spectral metric tuning finished for {config.display_name}.")
