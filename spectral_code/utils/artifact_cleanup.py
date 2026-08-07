from __future__ import annotations

import os
import shutil
from pathlib import Path


INTERMEDIATE_DIR_NAMES = (
    "dataset_features",
    "java_files",
    "cpg",
    "dot",
)

JOERN_INTERMEDIATE_PATTERNS = (
    "batch_src_*",
    "joern_raw_graphs_*",
    "batch_cpg_*.bin",
    "batch_cpg_*.bin.chunks.json",
    "batch_cpg_*_chunks",
    "batch_cpg_*.parse.log",
)

POST_SPECTRAL_DIAGNOSTIC_PATTERNS = (
    "skipped_methods_pipeline01.jsonl",
    "skipped_graph_parse_pipeline01.jsonl",
    "rebuild_graph_spectral.stdout.log",
    "rebuild_graph_spectral.stderr.log",
)

LEGACY_SPECTRAL_FILES = (
    "spectral_vectors_full.pkl",
    "spectral_vectors_full.pkl.tmp",
)

FINALIZED_PIPELINE_DIR_NAMES = (
    "dataset_features",
    "java_files",
    "cpg",
    "dot",
    "clean_graphs",
    "spectral_features",
)

FINALIZED_PIPELINE_FILE_NAMES = (
    "timing_stats.json",
    "pipeline_timings.json",
    "skipped_methods_pipeline01.jsonl",
    "skipped_graph_parse_pipeline01.jsonl",
    "rebuild_graph_spectral.stdout.log",
    "rebuild_graph_spectral.stderr.log",
)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _safe_child(path: Path, parent: Path) -> Path:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if resolved_path == resolved_parent or resolved_parent not in resolved_path.parents:
        raise ValueError(f"Refusing to remove path outside expected output root: {path}")
    return resolved_path


def remove_path(
    path: Path,
    parent: Path,
    *,
    dry_run: bool = False,
    compute_size: bool = False,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    safe_path = _safe_child(path, parent)
    size = path_size(safe_path) if compute_size else 0
    kind = "dir" if safe_path.is_dir() else "file"
    record = {
        "path": str(safe_path),
        "kind": kind,
        "bytes": size,
        "size": _format_bytes(size) if compute_size else "not measured",
        "dry_run": dry_run,
    }
    if dry_run:
        return record
    try:
        if safe_path.is_dir():
            shutil.rmtree(safe_path, ignore_errors=True)
        else:
            safe_path.unlink()
    except FileNotFoundError:
        pass
    return record


def cleanup_intermediate_artifacts(
    output_root: Path,
    *,
    include_dataset_features: bool = True,
    include_legacy_dirs: bool = True,
    include_post_spectral_diagnostics: bool = False,
    dry_run: bool = False,
    compute_size: bool = False,
) -> dict[str, object]:
    output_root = output_root.resolve()
    removed: list[dict[str, object]] = []
    seen: set[Path] = set()

    dir_names = list(INTERMEDIATE_DIR_NAMES if include_legacy_dirs else ())
    if not include_dataset_features and "dataset_features" in dir_names:
        dir_names.remove("dataset_features")

    for name in dir_names:
        target = output_root / name
        if target.exists() and target.resolve() not in seen:
            result = remove_path(target, output_root, dry_run=dry_run, compute_size=compute_size)
            if result:
                removed.append(result)
                seen.add(target.resolve())

    patterns = list(JOERN_INTERMEDIATE_PATTERNS)
    if include_post_spectral_diagnostics:
        patterns.extend(POST_SPECTRAL_DIAGNOSTIC_PATTERNS)

    for pattern in patterns:
        for target in output_root.glob(pattern):
            if target.exists() and target.resolve() not in seen:
                result = remove_path(target, output_root, dry_run=dry_run, compute_size=compute_size)
                if result:
                    removed.append(result)
                    seen.add(target.resolve())

    # Pipelines create ``models/`` up front, even when no model is trained.
    # Keep trained models, but avoid leaving an empty placeholder in durable
    # extraction-only outputs.
    models_dir = output_root / "models"
    if include_post_spectral_diagnostics and models_dir.is_dir() and not any(models_dir.iterdir()):
        result = remove_path(models_dir, output_root, dry_run=dry_run, compute_size=compute_size)
        if result:
            removed.append(result)

    total_bytes = sum(int(item["bytes"]) for item in removed)
    return {
        "output_root": str(output_root),
        "dry_run": dry_run,
        "size_measured": compute_size,
        "removed_count": len(removed),
        "bytes": total_bytes,
        "size": _format_bytes(total_bytes) if compute_size else "not measured",
        "removed": removed,
    }


def cleanup_legacy_spectral_artifacts(
    spectral_features_dir: Path,
    *,
    dry_run: bool = False,
    compute_size: bool = False,
) -> dict[str, object]:
    spectral_features_dir = spectral_features_dir.resolve()
    removed = []
    for name in LEGACY_SPECTRAL_FILES:
        target = spectral_features_dir / name
        result = remove_path(target, spectral_features_dir, dry_run=dry_run, compute_size=compute_size)
        if result:
            removed.append(result)

    total_bytes = sum(int(item["bytes"]) for item in removed)
    return {
        "spectral_features_dir": str(spectral_features_dir),
        "dry_run": dry_run,
        "size_measured": compute_size,
        "removed_count": len(removed),
        "bytes": total_bytes,
        "size": _format_bytes(total_bytes) if compute_size else "not measured",
        "removed": removed,
    }


def cleanup_finalized_pipeline_artifacts(
    output_root: Path,
    *,
    dry_run: bool = False,
    compute_size: bool = False,
) -> dict[str, object]:
    """Remove graph-extraction artefacts after a clean-data export succeeds.

    Reports, trained models, baselines, and the final ``clean_data`` directory
    are intentionally preserved.  This function only removes outputs that can
    be regenerated from the prepared dataset.
    """
    output_root = output_root.resolve()
    removed: list[dict[str, object]] = []
    for name in (*FINALIZED_PIPELINE_DIR_NAMES, *FINALIZED_PIPELINE_FILE_NAMES):
        result = remove_path(
            output_root / name,
            output_root,
            dry_run=dry_run,
            compute_size=compute_size,
        )
        if result:
            removed.append(result)

    total_bytes = sum(int(item["bytes"]) for item in removed)
    return {
        "output_root": str(output_root),
        "dry_run": dry_run,
        "size_measured": compute_size,
        "removed_count": len(removed),
        "bytes": total_bytes,
        "size": _format_bytes(total_bytes) if compute_size else "not measured",
        "removed": removed,
    }


def print_cleanup_summary(summary: dict[str, object], title: str = "Cleanup") -> None:
    count = int(summary.get("removed_count", 0) or 0)
    size = summary.get("size", "0 B")
    dry_run = bool(summary.get("dry_run", False))
    action = "Would remove" if dry_run else "Removed"
    print(f"[*] {title}: {action} {count:,} path(s), {size}.")
    for item in summary.get("removed", []):
        print(f"    - {item['size']} {item['path']}")
