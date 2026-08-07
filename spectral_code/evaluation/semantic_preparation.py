from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from spectral_code.evaluation.bcb_dataset import ClonePair
from spectral_code.evaluation.semantic_dataset import SemanticBenchmarkLoader
from spectral_code.utils.dataset_paths import semantic_dump_path, semantic_prepared_dir


@dataclass(frozen=True)
class PreparedSemanticDataset:
    language: str
    output_dir: Path
    data_jsonl: Path
    train_txt: Path
    metadata_json: Path
    type_labels_tsv: Path
    pair_count: int
    positive_pairs: int
    negative_pairs: int
    written_functions: int


def default_semantic_prepared_dir(language: str) -> Path:
    return semantic_prepared_dir(language)


def _unique_records_from_pairs(pairs: list[ClonePair], language: str) -> list[dict]:
    records: dict[int, dict] = {}
    normalized_language = language.strip().lower()
    for pair in pairs:
        records[pair.left_id] = {
            "idx": int(pair.left_id),
            "func": pair.left_code,
            "lang": normalized_language,
        }
        records[pair.right_id] = {
            "idx": int(pair.right_id),
            "func": pair.right_code,
            "lang": normalized_language,
        }
    return [records[key] for key in sorted(records)]


def prepare_semantic_dataset(
    language: str,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> PreparedSemanticDataset:
    loader = SemanticBenchmarkLoader(language=language, seed=seed)
    output_dir = Path(output_dir) if output_dir is not None else default_semantic_prepared_dir(language)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = loader.get_pairs()
    records = _unique_records_from_pairs(pairs, language)

    data_jsonl = output_dir / "data.jsonl"
    train_txt = output_dir / "train.txt"
    metadata_json = output_dir / "metadata.json"
    type_labels_tsv = output_dir / "type_labels.tsv"

    with data_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with train_txt.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(f"{pair.left_id}\t{pair.right_id}\t{pair.label}\n")

    with type_labels_tsv.open("w", encoding="utf-8") as f:
        for pair in pairs:
            if pair.label == 1:
                f.write(f"{pair.left_id}\t{pair.right_id}\tsemantic_{language.strip().lower()}\n")

    positive_pairs = sum(1 for pair in pairs if pair.label == 1)
    negative_pairs = len(pairs) - positive_pairs

    metadata = {
        "dataset": "semantic_benchmark",
        "language": language,
        "dump_path": str(semantic_dump_path()),
        "seed": seed,
        "total_pairs": len(pairs),
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "written_functions": len(records),
        "data_jsonl_semantics": "Functions/snippets extracted from semantic_clone.code_snippet for the selected language.",
        "train_txt_semantics": "Rows come from semantic_clone.clone_pair with label 1. No label-0 pairs are generated automatically.",
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreparedSemanticDataset(
        language=language,
        output_dir=output_dir,
        data_jsonl=data_jsonl,
        train_txt=train_txt,
        metadata_json=metadata_json,
        type_labels_tsv=type_labels_tsv,
        pair_count=len(pairs),
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        written_functions=len(records),
    )


def prepared_dataset_summary(prepared: PreparedSemanticDataset) -> dict:
    payload = asdict(prepared)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload
