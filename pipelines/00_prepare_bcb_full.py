import argparse
import json
import random
import sys
import os
import re
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DUMP_PATH = Path(r"C:\Users\koush\PyProjects\bcb")
DEFAULT_CLONE_TYPE = int(os.getenv("BCB_CLONE_TYPE", "1"))
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "bench_data" / f"bcb_full_type{DEFAULT_CLONE_TYPE}"


def _copy_unescape(value: str) -> str:
    if value == r"\N":
        return ""

    result: list[str] = []
    i = 0
    while i < len(value):
        char = value[i]
        if char != "\\" or i + 1 >= len(value):
            result.append(char)
            i += 1
            continue

        escaped = value[i + 1]
        replacements = {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "\\": "\\",
        }
        result.append(replacements.get(escaped, escaped))
        i += 2

    return "".join(result)


def _normalize_pair(left_id: int, right_id: int) -> tuple[int, int]:
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


def _iter_copy_table(dump_path: Path, table_name: str, desc: str):
    dump_size = dump_path.stat().st_size
    target_prefix = f"COPY {table_name} "
    in_table = False

    with dump_path.open("rb") as f, tqdm(
        total=dump_size,
        unit="B",
        unit_scale=True,
        desc=desc,
    ) as pbar:
        for raw in f:
            pbar.update(len(raw))
            line = raw.decode("utf-8", errors="replace").rstrip("\n")

            if not in_table:
                if line.startswith(target_prefix):
                    in_table = True
                continue

            if line == r"\.":
                break

            yield line


def _parse_optional_float(value: str) -> float | None:
    if value == r"\N" or value.strip() == "":
        return None
    return float(value)


def _strip_comments_and_whitespace(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    code = re.sub(r"//.*?$", " ", code, flags=re.M)
    code = re.sub(r"#.*?$", " ", code, flags=re.M)
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def _normalize_identifiers_and_literals(code: str) -> str:
    code = _strip_comments_and_whitespace(code)
    code = re.sub(r'"(?:\\.|[^"\\])*"', " $ ", code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", " $ ", code)
    code = re.sub(r"\b\d+(?:\.\d+)?\b", " # ", code)

    keywords = {
        "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
        "return", "new", "class", "public", "private", "protected", "static", "final",
        "void", "int", "long", "double", "float", "boolean", "char", "byte", "short",
        "true", "false", "null", "try", "catch", "finally", "throw", "throws", "extends",
        "implements", "import", "package", "this", "super", "var", "def", "lambda",
        "and", "or", "not", "in", "is", "with", "as", "from", "pass", "yield", "async",
        "await", "elif",
    }

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token in keywords else "@"

    code = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl, code)
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def _lecture_type(left_code: str, right_code: str) -> int:
    left_raw = _strip_comments_and_whitespace(left_code)
    right_raw = _strip_comments_and_whitespace(right_code)
    if left_raw == right_raw:
        return 1

    left_norm = _normalize_identifiers_and_literals(left_code)
    right_norm = _normalize_identifiers_and_literals(right_code)
    if left_norm == right_norm:
        return 2

    return 3


def _is_requested_clone_type(
    parts: list[str],
    clone_type: int,
    type3_min_similarity: float,
    type3_max_similarity: float,
    type4_max_similarity: float,
) -> bool:
    syntactic_type = parts[4]
    if clone_type in (1, 2):
        return syntactic_type == str(clone_type)

    if clone_type in (3, 4):
        if syntactic_type != "3" or len(parts) < 7:
            return False

        similarity_line = _parse_optional_float(parts[5])
        similarity_token = _parse_optional_float(parts[6])
        if similarity_line is None or similarity_token is None:
            return False

        min_similarity = min(similarity_line, similarity_token)
        if clone_type == 4:
            return min_similarity < type4_max_similarity

        # BigCloneBench stores Type-4 candidates as syntactic_type=3 with low similarity.
        return type3_min_similarity <= min_similarity < type3_max_similarity

    raise ValueError(f"Unsupported clone type: {clone_type}")


def collect_clone_type_pairs(
    dump_path: Path,
    clone_type: int,
    type3_min_similarity: float = 0.50,
    type3_max_similarity: float = 0.95,
    type4_max_similarity: float = 0.50,
) -> list[tuple[int, int, int]]:
    pairs: set[tuple[int, int]] = set()

    for line in _iter_copy_table(dump_path, "public.clones", f"Pass 1/5: Type-{clone_type} clone pairs"):
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        if not _is_requested_clone_type(
            parts,
            clone_type,
            type3_min_similarity,
            type3_max_similarity,
            type4_max_similarity,
        ):
            continue

        left_id = int(parts[0])
        right_id = int(parts[1])
        pairs.add(_normalize_pair(left_id, right_id))

    return [(left_id, right_id, 1) for left_id, right_id in sorted(pairs)]


def collect_code_ids(
    dump_path: Path,
    code_ids_to_keep: set[int] | None = None,
) -> tuple[list[int], dict[int, str]]:
    ids: list[int] = []
    kept_codes: dict[int, str] = {}

    for line in _iter_copy_table(dump_path, "public.pretty_printed_functions", "Pass 2/5: Function code ids"):
        function_id_raw, code_raw = line.split("\t", 1)
        function_id = int(function_id_raw)
        ids.append(function_id)
        if code_ids_to_keep is not None and function_id in code_ids_to_keep:
            kept_codes[function_id] = _copy_unescape(code_raw)

    return ids, kept_codes


def filter_lecture_type_positives(
    positives: list[tuple[int, int, int]],
    clone_type: int,
    code_map: dict[int, str],
    enabled: bool,
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    stats = {
        "strict_lecture_filter_enabled": int(enabled and clone_type != 4),
        "strict_lecture_candidates": len(positives),
        "strict_lecture_missing_code": 0,
        "strict_lecture_rejected": 0,
    }
    if not enabled or clone_type == 4:
        return positives, stats

    filtered: list[tuple[int, int, int]] = []
    for left_id, right_id, label in positives:
        left_code = code_map.get(left_id)
        right_code = code_map.get(right_id)
        if left_code is None or right_code is None:
            stats["strict_lecture_missing_code"] += 1
            stats["strict_lecture_rejected"] += 1
            continue

        if _lecture_type(left_code, right_code) != clone_type:
            stats["strict_lecture_rejected"] += 1
            continue

        filtered.append((left_id, right_id, label))

    return filtered, stats


def sample_candidate_nonclones(
    code_ids: list[int],
    excluded_pairs: set[tuple[int, int]],
    wanted: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    max_attempts = max(wanted * 100, 100_000)
    attempts = 0

    with tqdm(total=wanted, desc="Sampling candidate non-clones", unit="pair") as pbar:
        while len(candidates) < wanted and attempts < max_attempts:
            attempts += 1
            left_id = rng.choice(code_ids)
            right_id = rng.choice(code_ids)
            if left_id == right_id:
                continue

            key = _normalize_pair(left_id, right_id)
            if key in excluded_pairs or key in candidates:
                continue

            candidates.add(key)
            pbar.update(1)

    if len(candidates) < wanted:
        raise RuntimeError(f"Could only sample {len(candidates)} non-clone candidates out of {wanted}.")

    return candidates


def remove_known_clones(
    dump_path: Path,
    candidate_nonclones: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    if not candidate_nonclones:
        return candidate_nonclones

    validated = set(candidate_nonclones)

    for line in _iter_copy_table(dump_path, "public.clones", "Pass 3/5: Remove known clones"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        key = _normalize_pair(int(parts[0]), int(parts[1]))
        validated.discard(key)

        if len(validated) == len(candidate_nonclones):
            # Keep scanning: a future row might still match a candidate.
            pass

    return validated


def extract_codes(
    dump_path: Path,
    needed_ids: set[int],
    output_path: Path,
) -> int:
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for line in _iter_copy_table(dump_path, "public.pretty_printed_functions", "Pass 4/5: Extract code"):
            function_id_raw, code_raw = line.split("\t", 1)
            function_id = int(function_id_raw)
            if function_id not in needed_ids:
                continue

            out.write(json.dumps({"idx": str(function_id), "func": _copy_unescape(code_raw)}, ensure_ascii=False))
            out.write("\n")
            written += 1

            if written == len(needed_ids):
                break

    return written


def write_pairs(path: Path, rows: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for left_id, right_id, label in rows:
            f.write(f"{left_id}\t{right_id}\t{label}\n")


def write_type_labels(path: Path, positives: list[tuple[int, int, int]], clone_type: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for left_id, right_id, _ in positives:
            f.write(f"{left_id}\t{right_id}\ttype_{clone_type}\n")


def write_positive_train(path: Path, positives: list[tuple[int, int, int]]) -> None:
    """Write every known positive clone pair for this clone type."""
    write_pairs(path, positives)


def _parse_positive_fraction(value: str) -> float:
    fraction = float(value)
    if fraction <= 0.0 or fraction > 1.0:
        raise argparse.ArgumentTypeError("--positive-fraction must be in the range (0, 1].")
    return fraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-clone-type BCB benchmark from the full BCB dump.")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clone-type", type=int, choices=[1, 2, 3, 4], default=DEFAULT_CLONE_TYPE)
    parser.add_argument(
        "--type3-min-similarity",
        type=float,
        default=0.50,
        help=(
            "Minimum LEAST(similarity_line, similarity_token) for Type-3 positives. "
            "BCB rows with syntactic_type=3 below this threshold are Type-4-like and excluded."
        ),
    )
    parser.add_argument(
        "--type3-max-similarity",
        type=float,
        default=0.95,
        help=(
            "Exclusive upper bound for LEAST(similarity_line, similarity_token) in Type-3 positives. "
            "This removes near-Type-2 template clones from the lecture-clean Type-3 benchmark."
        ),
    )
    parser.add_argument(
        "--type4-max-similarity",
        type=float,
        default=0.50,
        help=(
            "Exclusive upper bound for LEAST(similarity_line, similarity_token) in Type-4 positives. "
            "BCB represents Type-4-like pairs as syntactic_type=3 below this threshold."
        ),
    )
    parser.add_argument(
        "--no-strict-lecture-filter",
        action="store_false",
        dest="strict_lecture_filter",
        help=(
            "Disable code-text lecture filtering and keep only the BCB syntactic_type/similarity filters."
        ),
    )
    parser.add_argument("--target-pairs", type=int, default=100_000)
    parser.add_argument("--max-positive-pairs", type=int, default=0)
    parser.add_argument(
        "--positive-fraction",
        type=_parse_positive_fraction,
        default=0.50,
        help=(
            "Maximum fraction of train.txt that may be positive clones. "
            "Use 1.0 to keep only positives when positives exceed --target-pairs."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dump.exists():
        raise FileNotFoundError(f"BCB dump not found: {args.dump}")
    if args.target_pairs < 1:
        raise ValueError("--target-pairs must be at least 1.")
    if args.clone_type == 3 and args.type3_min_similarity >= args.type3_max_similarity:
        raise ValueError("--type3-min-similarity must be lower than --type3-max-similarity.")
    if args.clone_type == 4 and args.type4_max_similarity <= 0:
        raise ValueError("--type4-max-similarity must be greater than 0.")
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = PROJECT_ROOT / "bench_data" / f"bcb_full_type{args.clone_type}"

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positives = collect_clone_type_pairs(
        args.dump,
        args.clone_type,
        args.type3_min_similarity,
        args.type3_max_similarity,
        args.type4_max_similarity,
    )
    positive_code_ids = {left_id for left_id, _, _ in positives} | {right_id for _, right_id, _ in positives}
    code_ids, positive_code_map = collect_code_ids(args.dump, positive_code_ids)
    code_id_set = set(code_ids)

    positives = [row for row in positives if row[0] in code_id_set and row[1] in code_id_set]
    positives, strict_filter_stats = filter_lecture_type_positives(
        positives,
        args.clone_type,
        positive_code_map,
        args.strict_lecture_filter,
    )
    total_available_positives = len(positives)
    max_positive_pairs = max(1, int(args.target_pairs * args.positive_fraction))
    if args.max_positive_pairs > 0:
        max_positive_pairs = min(max_positive_pairs, args.max_positive_pairs)

    if len(positives) > max_positive_pairs:
        positives = rng.sample(positives, max_positive_pairs)
        positives.sort()

    positive_keys = {_normalize_pair(left_id, right_id) for left_id, right_id, _ in positives}
    wanted_negatives = max(0, args.target_pairs - len(positives))

    candidate_count = max(wanted_negatives + 10_000, int(wanted_negatives * 1.25))
    candidate_nonclones = sample_candidate_nonclones(code_ids, positive_keys, candidate_count, rng)
    validated_nonclones = remove_known_clones(args.dump, candidate_nonclones)

    if len(validated_nonclones) < wanted_negatives:
        raise RuntimeError(
            f"Only {len(validated_nonclones)} validated non-clones remained; "
            f"need {wanted_negatives}. Re-run with a larger candidate multiplier."
        )

    negatives = [(left_id, right_id, 0) for left_id, right_id in sorted(validated_nonclones)]
    negatives = rng.sample(negatives, wanted_negatives)

    rows = positives + negatives
    rng.shuffle(rows)

    needed_ids = {left_id for left_id, _, _ in rows} | {right_id for _, right_id, _ in rows}
    written_functions = extract_codes(args.dump, needed_ids, args.output_dir / "data.jsonl")

    write_pairs(args.output_dir / "train.txt", rows)
    write_positive_train(args.output_dir / "train_positives.txt", positives)
    write_type_labels(args.output_dir / "type_labels.tsv", positives, args.clone_type)

    metadata = {
        "dump_path": str(args.dump.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "target_pairs": args.target_pairs,
        "seed": args.seed,
        "source": "public.pretty_printed_functions + public.clones",
        "train_txt_semantics": (
            "Mixed benchmark split: all selected positive clone pairs for this type "
            "plus sampled validated non-clone pairs up to target_pairs."
        ),
        "train_positives_txt_semantics": "Complete list of known positive clone pairs for this clone type.",
        "data_jsonl_semantics": "Only functions referenced by train.txt rows, not all functions in the BCB dump.",
        "available_code_ids": len(code_ids),
        "clone_type": args.clone_type,
        "available_positive_clones": total_available_positives,
        "positive_fraction": args.positive_fraction,
        "max_positive_pairs": max_positive_pairs,
        "type3_min_similarity": args.type3_min_similarity if args.clone_type == 3 else None,
        "type3_max_similarity": args.type3_max_similarity if args.clone_type == 3 else None,
        "type4_max_similarity": args.type4_max_similarity if args.clone_type == 4 else None,
        "type3_semantics": (
            "syntactic_type=3 and type3_min_similarity <= "
            "LEAST(similarity_line, similarity_token) < type3_max_similarity; "
            "lower-similarity rows are Type-4-like and very high-similarity rows are Type-2-like."
            if args.clone_type == 3
            else None
        ),
        "type4_semantics": (
            "syntactic_type=3 and LEAST(similarity_line, similarity_token) < type4_max_similarity."
            if args.clone_type == 4
            else None
        ),
        "strict_lecture_filter": bool(args.strict_lecture_filter and args.clone_type != 4),
        **strict_filter_stats,
        f"type_{args.clone_type}_clones": len(positives),
        "positive_clones": len(positives),
        "train_positives_pairs": len(positives),
        "non_clones": len(negatives),
        "total_pairs": len(rows),
        "needed_function_ids": len(needed_ids),
        "written_functions": written_functions,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[+] Full BCB Type-{args.clone_type} benchmark created.")
    print(f"    Output Dir: {args.output_dir}")
    print(f"    Functions written: {written_functions}")
    print(f"    Type-{args.clone_type} clones: {len(positives)}")
    if strict_filter_stats["strict_lecture_filter_enabled"]:
        print(f"    Strict lecture rejected: {strict_filter_stats['strict_lecture_rejected']}")
    print(f"    Non-clones: {len(negatives)}")
    print(f"    Total pairs: {len(rows)}")


if __name__ == "__main__":
    main()
