from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_project_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_project_env()


def path_from_env(name: str, default: str | Path, base: Path = PROJECT_ROOT) -> Path:
    raw = os.getenv(name)
    path = Path(raw).expanduser() if raw else Path(default).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


DATA_ROOT = path_from_env("DATA_ROOT", PROJECT_ROOT.parent / "data")
OUTPUTS_ROOT = path_from_env("OUTPUT_BASE_DIR", PROJECT_ROOT.parent / "outputs")


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _match_child(parent: Path, predicate) -> Path | None:
    if not parent.exists():
        return None

    for child in parent.iterdir():
        if predicate(child):
            return child
    return None


def bcb_dump_path() -> Path:
    env_path = os.getenv("BCB_DUMP_PATH")
    if env_path:
        return path_from_env("BCB_DUMP_PATH", DATA_ROOT / "bcb" / "dump")

    candidates = [
        DATA_ROOT / "bcb" / "dump",
        DATA_ROOT / "bcb.dump",
        DATA_ROOT / "bcb",
    ]
    return _first_existing(candidates)


def bcb_prepared_root() -> Path:
    env_root = os.getenv("BCB_PREPARED_ROOT")
    if env_root:
        return path_from_env("BCB_PREPARED_ROOT", DATA_ROOT / "bcb_prepared")

    candidates = [
        DATA_ROOT / "bcb_prepared",
        DATA_ROOT / "bigclonebench_prepared",
    ]
    return _first_existing(candidates)


def bcb_type_dir(clone_type: str | int) -> Path:
    clone_type = str(clone_type).strip()
    env_dir = os.getenv("BCB_DATA_DIR")
    if env_dir:
        return path_from_env("BCB_DATA_DIR", bcb_prepared_root() / f"bcb_full_type{clone_type}")

    if clone_type.lower().replace("-", "_") in {"non_clone", "nonclone", "false_positives"}:
        return DATA_ROOT / "bcb" / "non_clone"

    candidates = [
        DATA_ROOT / "bcb" / f"type{clone_type}",
        bcb_prepared_root() / f"bcb_full_type{clone_type}",
        DATA_ROOT / f"bcb_full_type{clone_type}",
    ]
    return _first_existing(candidates)


def xglue_dir() -> Path:
    return path_from_env("XGLUE_DATA_DIR", DATA_ROOT / "xglue")


def semantic_benchmark_root() -> Path:
    env_root = os.getenv("SEMANTIC_BENCHMARK_DIR")
    if env_root:
        return path_from_env("SEMANTIC_BENCHMARK_DIR", DATA_ROOT / "semantic_benchmark")

    candidates = [
        DATA_ROOT / "Semantic Benchmark",
        DATA_ROOT / "semantic benchmark",
        DATA_ROOT / "semantic_benchmark",
    ]
    return _first_existing(candidates)


def semantic_dump_path() -> Path:
    env_path = os.getenv("SEMANTIC_DUMP_PATH")
    if env_path:
        return path_from_env("SEMANTIC_DUMP_PATH", DATA_ROOT / "semantic_clonebench.dump")

    candidates = [
        DATA_ROOT / "semantic_clonebench.dump",
        DATA_ROOT / "semantic_clone_bench.dump",
    ]
    return _first_existing(candidates)


def semantic_prepared_root() -> Path:
    return path_from_env("SEMANTIC_PREPARED_ROOT", DATA_ROOT / "semantic_benchmark_prepared")


def semantic_prepared_dir(language: str) -> Path:
    return semantic_prepared_root() / language.strip().lower()


def semantic_language_dir(language: str) -> Path:
    root = semantic_benchmark_root()
    normalized = language.strip().lower()
    direct = root / language
    if direct.exists():
        return direct

    matched = _match_child(root, lambda child: child.is_dir() and child.name.strip().lower() == normalized)
    if matched is None:
        raise FileNotFoundError(f"Semantic Benchmark language folder not found for: {language}")
    return matched


def semantic_injected_dir(language: str) -> Path:
    language_dir = semantic_language_dir(language)
    matched = _match_child(
        language_dir,
        lambda child: child.is_dir() and "inject" in child.name.lower() and "system" in child.name.lower(),
    )
    if matched is None:
        raise FileNotFoundError(f"Injected-in-system folder not found for semantic language: {language}")
    return matched


def semantic_standalone_dir(language: str) -> Path:
    language_dir = semantic_language_dir(language)
    matched = _match_child(
        language_dir,
        lambda child: child.is_dir() and "stand" in child.name.lower() and "clone" in child.name.lower(),
    )
    if matched is None:
        raise FileNotFoundError(f"Standalone-clones folder not found for semantic language: {language}")
    return matched


def semantic_reference_file(language: str) -> Path:
    injected_dir = semantic_injected_dir(language)
    matched = _match_child(
        injected_dir,
        lambda child: child.is_file() and child.name.lower().endswith("references.txt"),
    )
    if matched is None:
        raise FileNotFoundError(f"Reference file not found for semantic language: {language}")
    return matched


def output_root_for(dataset: str, variant: str | int | None = None) -> Path:
    dataset_key = dataset.strip().lower().replace(" ", "_")

    if dataset_key in {"bcb", "bigclonebench", "big_clone_bench"}:
        clone_type = str(variant if variant is not None else os.getenv("BCB_CLONE_TYPE", "1")).strip()
        if clone_type.lower().replace("-", "_") in {"non_clone", "nonclone", "false_positives"}:
            return OUTPUTS_ROOT / "bcb" / "non_clone"
        return OUTPUTS_ROOT / "bcb" / f"type{clone_type}"

    if dataset_key == "xglue":
        return OUTPUTS_ROOT / "xglue"

    if dataset_key in {"semantic", "semantic_benchmark", "semanticbenchmark"}:
        if variant is None:
            return OUTPUTS_ROOT / "semantic_benchmark"
        return OUTPUTS_ROOT / "semantic_benchmark" / str(variant).strip().lower()

    return OUTPUTS_ROOT / dataset_key
