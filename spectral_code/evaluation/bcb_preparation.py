import argparse
import json
import random
import sys
import os
import re
import time
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.dataset_paths import bcb_dump_path, bcb_type_dir, output_root_for
from spectral_code.utils.pipeline_timings import record_pipeline_timing


DEFAULT_DUMP_PATH = bcb_dump_path()
DEFAULT_CLONE_TYPE = int(os.getenv("BCB_CLONE_TYPE", "1"))
DEFAULT_OUTPUT_DIR = bcb_type_dir(DEFAULT_CLONE_TYPE)


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


def is_getter_setter(code: str) -> bool:
    compact = _strip_comments_and_whitespace(code)
    if "{" not in compact or "}" not in compact:
        return False

    signature = compact.split("{", 1)[0]
    body = compact.split("{", 1)[1].rsplit("}", 1)[0].strip()
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)\s*$", signature)
    if not match:
        return False

    method_name = match.group(1)
    params = match.group(2).strip()
    simple_body = re.sub(r"\s+", " ", body).strip()
    simple_body = simple_body.strip("{} ")

    if method_name.startswith("get") or method_name.startswith("is"):
        if params:
            return False
        return bool(re.fullmatch(r"return\s+.+;", simple_body)) and not re.search(
            r"\b(if|for|while|switch|try|catch|throw)\b",
            simple_body,
        )

    if method_name.startswith("set"):
        if not params:
            return False
        if not re.search(r"\b(if|for|while|switch|try|catch|throw)\b", simple_body):
            return True
        assignment = r"(?:this\.)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*[A-Za-z_][A-Za-z0-9_]*\s*;"
        return bool(
            re.fullmatch(assignment, simple_body)
            or re.fullmatch(assignment + r"\s*return\s+this\s*;", simple_body)
        )

    return False


def has_unsupported_nested_java_method(code: str) -> bool:
    """Detect Java constructs whose nested method bodies are not represented in the outer method graph."""
    compact = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    compact = re.sub(r"//.*", " ", compact)

    nested_method_signature = (
        r"\b(?:public|protected|private|static|final|synchronized|abstract|native|strictfp|\s)*"
        r"(?:[A-Za-z_$][\w$]*(?:\s*<[^;{}()]*>)?(?:\s*\[\])?|\?)"
        r"(?:\s+[A-Za-z_$][\w$]*(?:\s*<[^;{}()]*>)?(?:\s*\[\])?)*"
        r"\s+[A-Za-z_$][\w$]*\s*\([^;{}]*\)\s*(?:throws\s+[A-Za-z0-9_.$,\s]+)?\{"
    )

    anonymous_class_with_method = re.compile(
        r"\bnew\s+[A-Za-z_$][\w$.$]*(?:\s*<[^;{}()]*>)?\s*\([^;{}]*\)\s*\{"
        r"(?:(?!\bnew\s+[A-Za-z_$][\w$.$]*(?:\s*<[^;{}()]*>)?\s*\([^;{}]*\)\s*\{).)*?"
        + nested_method_signature,
        flags=re.DOTALL,
    )
    local_class_with_method = re.compile(
        r"\bclass\s+[A-Za-z_$][\w$]*[^{]*\{(?:(?!\bclass\s+[A-Za-z_$][\w$]*[^{]*\{).)*?"
        + nested_method_signature,
        flags=re.DOTALL,
    )

    return bool(anonymous_class_with_method.search(compact) or local_class_with_method.search(compact))


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
    type3_max_similarity: float = 1.00,
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


def collect_false_positive_pairs(dump_path: Path) -> list[tuple[int, int, int]]:
    pairs: set[tuple[int, int]] = set()

    for line in _iter_copy_table(dump_path, "public.false_positives", "Pass 1b/5: Curated false-positive non-clones"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        left_id = int(parts[0])
        right_id = int(parts[1])
        pairs.add(_normalize_pair(left_id, right_id))

    return [(left_id, right_id, 0) for left_id, right_id in sorted(pairs)]


def _nonempty_line_count(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())


def filter_false_positive_nonclone_rows(
    false_positive_nonclones: list[tuple[int, int, int]],
    code_id_set: set[int],
    positive_keys: set[tuple[int, int]],
    getter_setter_code_ids: set[int],
    three_line_code_ids: set[int] | None = None,
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    three_line_code_ids = three_line_code_ids or set()
    rows: list[tuple[int, int, int]] = []
    stats = {
        "false_positive_nonclone_pairs_before_code_filter": 0,
        "false_positive_nonclone_pairs_removed_by_getter_setter": 0,
        "false_positive_nonclone_pairs_removed_by_both_three_line": 0,
        "false_positive_nonclone_pairs_removed_by_other_code_filters": 0,
    }

    for left_id, right_id, _ in false_positive_nonclones:
        key = _normalize_pair(left_id, right_id)
        if key in positive_keys:
            continue

        stats["false_positive_nonclone_pairs_before_code_filter"] += 1
        if left_id in getter_setter_code_ids or right_id in getter_setter_code_ids:
            stats["false_positive_nonclone_pairs_removed_by_getter_setter"] += 1
            continue

        if left_id in three_line_code_ids and right_id in three_line_code_ids:
            stats["false_positive_nonclone_pairs_removed_by_both_three_line"] += 1
            continue

        if left_id not in code_id_set or right_id not in code_id_set:
            stats["false_positive_nonclone_pairs_removed_by_other_code_filters"] += 1
            continue

        rows.append((left_id, right_id, 0))

    return rows, stats


def collect_code_ids(
    dump_path: Path,
    code_ids_to_keep: set[int] | None = None,
    restrict_to_keep: bool = False,
    keep_getter_setters: bool = False,
) -> tuple[list[int], dict[int, str], dict[str, int], dict[str, set[int]]]:
    ids: list[int] = []
    kept_codes: dict[int, str] = {}
    excluded_ids = {
        "getter_setter": set(),
        "unsupported_nested_java_method": set(),
    }
    stats = {
        "total_function_rows": 0,
        "detected_getter_setter_functions": 0,
        "excluded_getter_setter_functions": 0,
        "excluded_unsupported_nested_java_method_functions": 0,
    }

    for line in _iter_copy_table(dump_path, "public.pretty_printed_functions", "Pass 2/5: Function code ids"):
        function_id_raw, code_raw = line.split("\t", 1)
        function_id = int(function_id_raw)
        stats["total_function_rows"] += 1
        if restrict_to_keep and (code_ids_to_keep is None or function_id not in code_ids_to_keep):
            continue

        code = _copy_unescape(code_raw)
        if is_getter_setter(code):
            stats["detected_getter_setter_functions"] += 1
            excluded_ids["getter_setter"].add(function_id)
            if not keep_getter_setters:
                stats["excluded_getter_setter_functions"] += 1
                continue
        if has_unsupported_nested_java_method(code):
            stats["excluded_unsupported_nested_java_method_functions"] += 1
            excluded_ids["unsupported_nested_java_method"].add(function_id)
            continue

        ids.append(function_id)
        if code_ids_to_keep is not None and function_id in code_ids_to_keep:
            kept_codes[function_id] = code

    stats["usable_code_ids"] = len(ids)
    return ids, kept_codes, stats, excluded_ids


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
    if wanted <= 0:
        return set()

    pool_ids = list(dict.fromkeys(code_ids))
    pool_set = set(pool_ids)
    excluded_in_pool = {
        pair for pair in excluded_pairs
        if pair[0] in pool_set and pair[1] in pool_set
    }
    capacity = (len(pool_ids) * (len(pool_ids) - 1)) // 2 - len(excluded_in_pool)
    if wanted > capacity:
        raise RuntimeError(
            f"Cannot sample {wanted} candidate non-clones from {len(pool_ids)} code ids: "
            f"only {capacity} non-excluded unique pairs are possible."
        )

    candidates: set[tuple[int, int]] = set()
    if capacity <= max(2_000_000, wanted * 4):
        total_unordered_pairs = (len(pool_ids) * (len(pool_ids) - 1)) // 2
        with tqdm(total=total_unordered_pairs, desc="Enumerating candidate non-clone pool", unit="pair") as pbar:
            possible_pairs: list[tuple[int, int]] = []
            for i, left_id in enumerate(pool_ids):
                for right_id in pool_ids[i + 1:]:
                    key = _normalize_pair(left_id, right_id)
                    if key not in excluded_in_pool:
                        possible_pairs.append(key)
                    pbar.update(1)
        return set(rng.sample(possible_pairs, wanted))

    max_attempts = max(wanted * 200, 100_000)
    attempts = 0

    with tqdm(total=wanted, desc="Sampling candidate non-clones", unit="pair") as pbar:
        while len(candidates) < wanted and attempts < max_attempts:
            attempts += 1
            left_id = rng.choice(pool_ids)
            right_id = rng.choice(pool_ids)
            if left_id == right_id:
                continue

            key = _normalize_pair(left_id, right_id)
            if key in excluded_in_pool or key in candidates:
                continue

            candidates.add(key)
            pbar.update(1)

    if len(candidates) < wanted:
        raise RuntimeError(
            f"Could only sample {len(candidates)} non-clone candidates out of {wanted} "
            f"after {attempts} attempts from {len(pool_ids)} code ids. "
            "Use a larger negative pool or lower --target-pairs."
        )

    return candidates


def _nonclone_pair_capacity(code_ids: list[int], excluded_pairs: set[tuple[int, int]]) -> int:
    pool_set = set(code_ids)
    excluded_in_pool = sum(1 for pair in excluded_pairs if pair[0] in pool_set and pair[1] in pool_set)
    return (len(pool_set) * (len(pool_set) - 1)) // 2 - excluded_in_pool


def select_positive_pairs_with_code_coverage(
    positives: list[tuple[int, int, int]],
    required_code_ids: set[int],
    max_pairs: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    shuffled = positives[:]
    rng.shuffle(shuffled)

    uncovered = set(required_code_ids)
    selected: list[tuple[int, int, int]] = []
    selected_keys: set[tuple[int, int]] = set()

    for row in shuffled:
        left_id, right_id, _ = row
        if left_id not in uncovered and right_id not in uncovered:
            continue

        selected.append(row)
        selected_keys.add(_normalize_pair(left_id, right_id))
        uncovered.discard(left_id)
        uncovered.discard(right_id)

        if not uncovered or len(selected) >= max_pairs:
            break

    for row in shuffled:
        if len(selected) >= max_pairs:
            break
        key = _normalize_pair(row[0], row[1])
        if key in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(key)

    selected.sort()
    return selected, {
        "positive_code_ids_required": len(required_code_ids),
        "positive_code_ids_covered": len(required_code_ids) - len(uncovered),
        "positive_code_ids_uncovered": len(uncovered),
    }


def select_nonclone_pairs_with_code_budget(
    rows: list[tuple[int, int, int]],
    max_code_ids: int,
    max_pairs: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    if max_code_ids <= 0:
        selected = rows[:]
        if max_pairs > 0 and len(selected) > max_pairs:
            selected = rng.sample(selected, max_pairs)
            selected.sort()
        return selected, {
            "non_clone_code_id_budget_requested": 0,
            "non_clone_pairs_before_code_id_budget": len(rows),
            "non_clone_pairs_after_code_id_budget": len(selected),
            "non_clone_code_ids_after_budget": len({left for left, _, _ in selected} | {right for _, right, _ in selected}),
        }

    shuffled = rows[:]
    rng.shuffle(shuffled)

    selected: list[tuple[int, int, int]] = []
    selected_keys: set[tuple[int, int]] = set()
    selected_ids: set[int] = set()

    def add_row(row: tuple[int, int, int]) -> None:
        left_id, right_id, _ = row
        selected.append(row)
        selected_keys.add(_normalize_pair(left_id, right_id))
        selected_ids.add(left_id)
        selected_ids.add(right_id)

    for row in shuffled:
        if max_pairs > 0 and len(selected) >= max_pairs:
            break
        left_id, right_id, _ = row
        key = _normalize_pair(left_id, right_id)
        if key in selected_keys:
            continue
        new_ids = int(left_id not in selected_ids) + int(right_id not in selected_ids)
        if len(selected_ids) + new_ids > max_code_ids:
            continue
        add_row(row)

    for row in shuffled:
        if max_pairs > 0 and len(selected) >= max_pairs:
            break
        left_id, right_id, _ = row
        key = _normalize_pair(left_id, right_id)
        if key in selected_keys:
            continue
        if left_id in selected_ids and right_id in selected_ids:
            add_row(row)

    selected.sort()
    return selected, {
        "non_clone_code_id_budget_requested": max_code_ids,
        "non_clone_pairs_before_code_id_budget": len(rows),
        "non_clone_pairs_after_code_id_budget": len(selected),
        "non_clone_code_ids_after_budget": len(selected_ids),
    }


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


def write_codes_from_map(
    code_map: dict[int, str],
    needed_ids: set[int],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for function_id in sorted(needed_ids):
            code = code_map.get(function_id)
            if code is None:
                continue
            out.write(json.dumps({"idx": str(function_id), "func": code}, ensure_ascii=False))
            out.write("\n")
            written += 1
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
    preparation_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Extract a sampled single-clone-type BCB benchmark from the full BCB dump.")
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
        default=1.00,
        help=(
            "Exclusive upper bound for LEAST(similarity_line, similarity_token) in Type-3 positives. "
            "Use 1.00 to include moderate, strong, and very-strong Type-3 clones."
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
    parser.add_argument(
        "--max-non-clone-code-ids",
        type=int,
        default=0,
        help=(
            "For curated false-positive non-clone sampling, cap the number of unique function/code ids "
            "referenced by selected non-clone pairs. Use 0 to disable the cap."
        ),
    )
    parser.add_argument(
        "--keep-getter-setters",
        action="store_true",
        help="Keep getter/setter functions during code scanning instead of excluding their pairs outright.",
    )
    parser.add_argument(
        "--drop-non-clone-pairs-both-three-line",
        action="store_true",
        help="For curated non-clone pairs, remove pairs where both functions have exactly three non-empty lines.",
    )
    parser.add_argument(
        "--write-all-filtered-non-clone-code-ids",
        action="store_true",
        help=(
            "When using curated non-clones, write every filtered non-clone code id to data.jsonl, "
            "even if only a subset appears in train.txt."
        ),
    )
    parser.add_argument(
        "--all-filtered-non-clone-pairs",
        action="store_true",
        help=(
            "When using curated non-clones, keep every filtered non-clone pair instead of sampling "
            "down to --target-pairs."
        ),
    )
    parser.add_argument("--max-positive-pairs", type=int, default=0)
    parser.add_argument(
        "--cover-positive-code-ids",
        action="store_true",
        help="Select positive pairs to cover as many clone-type code ids as possible before adding extra positives.",
    )
    parser.add_argument(
        "--negative-pool",
        choices=["false-positives", "all-code-ids", "positive-code-ids"],
        default="positive-code-ids",
        help="Choose the code id pool used when sampling validated non-clone pairs.",
    )
    parser.add_argument(
        "--positive-fraction",
        type=_parse_positive_fraction,
        default=0.50,
        help=(
            "Maximum fraction of train.txt that may be positive clones. "
            "Use 1.0 to keep only positives when positives exceed --target-pairs."
        ),
    )
    parser.add_argument(
        "--balance-positive-to-negative-count",
        action="store_true",
        help=(
            "When using --negative-pool false-positives, downsample the selected positive clones "
            "and/or curated false-positive non-clones so both classes have the same count."
        ),
    )
    parser.add_argument(
        "--preselect-positive-pairs-before-code-scan",
        action="store_true",
        help=(
            "Downsample positive clone pairs to the configured positive cap before scanning function code. "
            "This avoids reading/storing code for millions of unselected clone ids."
        ),
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Write only selected positive clone pairs for this clone type; no non-clone rows are added.",
    )
    parser.add_argument(
        "--non-clone-only",
        action="store_true",
        help="Write only curated BCB false-positive non-clone pairs; clone-type positives are skipped.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dump.exists():
        raise FileNotFoundError(f"BCB dump not found: {args.dump}")
    if args.target_pairs < 1:
        raise ValueError("--target-pairs must be at least 1.")
    if args.max_non_clone_code_ids < 0:
        raise ValueError("--max-non-clone-code-ids must be at least 0.")
    if args.clone_type == 3 and args.type3_min_similarity >= args.type3_max_similarity:
        raise ValueError("--type3-min-similarity must be lower than --type3-max-similarity.")
    if args.clone_type == 4 and args.type4_max_similarity <= 0:
        raise ValueError("--type4-max-similarity must be greater than 0.")
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = bcb_type_dir(args.clone_type)

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.positive_only and args.non_clone_only:
        raise ValueError("--positive-only and --non-clone-only cannot be used together.")

    positives = [] if args.non_clone_only else collect_clone_type_pairs(
        args.dump,
        args.clone_type,
        args.type3_min_similarity,
        args.type3_max_similarity,
        args.type4_max_similarity,
    )
    full_clone_type_positive_pairs = len(positives)
    full_clone_type_code_ids = len({left_id for left_id, _, _ in positives} | {right_id for _, right_id, _ in positives})
    max_positive_pairs = max(1, int(args.target_pairs * args.positive_fraction))
    if args.max_positive_pairs > 0:
        max_positive_pairs = min(max_positive_pairs, args.max_positive_pairs)

    preselected_positive_pairs_before_code_scan = False
    if args.preselect_positive_pairs_before_code_scan and len(positives) > max_positive_pairs:
        positives = rng.sample(positives, max_positive_pairs)
        positives.sort()
        preselected_positive_pairs_before_code_scan = True

    false_positive_nonclones = collect_false_positive_pairs(args.dump) if (
        not args.positive_only and (args.negative_pool == "false-positives" or args.non_clone_only)
    ) else []
    positive_code_ids = {left_id for left_id, _, _ in positives} | {right_id for _, right_id, _ in positives}
    false_positive_code_ids = (
        {left_id for left_id, _, _ in false_positive_nonclones}
        | {right_id for _, right_id, _ in false_positive_nonclones}
    )
    restrict_code_scan = args.non_clone_only or args.negative_pool in {"positive-code-ids", "false-positives"}
    code_ids_to_keep = set(positive_code_ids)
    if args.non_clone_only or args.negative_pool == "false-positives":
        code_ids_to_keep.update(false_positive_code_ids)
    code_ids, positive_code_map, code_filter_stats, excluded_code_ids = collect_code_ids(
        args.dump,
        code_ids_to_keep,
        restrict_to_keep=restrict_code_scan,
        keep_getter_setters=args.keep_getter_setters,
    )
    code_id_set = set(code_ids)
    getter_setter_code_ids = set() if args.keep_getter_setters else excluded_code_ids["getter_setter"]
    three_line_code_ids = {
        function_id
        for function_id, code in positive_code_map.items()
        if _nonempty_line_count(code) == 3
    } if args.drop_non_clone_pairs_both_three_line else set()

    positives = [row for row in positives if row[0] in code_id_set and row[1] in code_id_set]
    positives, strict_filter_stats = filter_lecture_type_positives(
        positives,
        args.clone_type,
        positive_code_map,
        args.strict_lecture_filter,
    )
    total_available_positives = len(positives)
    available_positive_code_ids = {left_id for left_id, _, _ in positives} | {right_id for _, right_id, _ in positives}

    positive_coverage_stats = {
        "positive_code_ids_required": len(available_positive_code_ids),
        "positive_code_ids_covered": None,
        "positive_code_ids_uncovered": None,
    }
    if args.cover_positive_code_ids:
        positives, positive_coverage_stats = select_positive_pairs_with_code_coverage(
            positives,
            available_positive_code_ids,
            min(max_positive_pairs, len(positives)),
            rng,
        )
    elif len(positives) > max_positive_pairs:
        positives = rng.sample(positives, max_positive_pairs)
        positives.sort()

    positive_keys = {_normalize_pair(left_id, right_id) for left_id, right_id, _ in positives}
    wanted_negatives = max(0, args.target_pairs - len(positives))

    requested_wanted_negatives = wanted_negatives
    requested_candidate_count = 0 if wanted_negatives == 0 else max(wanted_negatives + 10_000, int(wanted_negatives * 1.25))
    candidate_count = requested_candidate_count
    requested_negative_pool = args.negative_pool
    negative_pool_limit_reason = None
    false_positive_nonclones_available = len(false_positive_nonclones)
    false_positive_nonclones_after_code_filter = None
    false_positive_code_filter_stats = {
        "false_positive_nonclone_pairs_before_code_filter": None,
        "false_positive_nonclone_pairs_removed_by_getter_setter": None,
        "false_positive_nonclone_pairs_removed_by_both_three_line": None,
        "false_positive_nonclone_pairs_removed_by_other_code_filters": None,
    }
    non_clone_code_budget_stats = {
        "non_clone_code_id_budget_requested": args.max_non_clone_code_ids,
        "non_clone_pairs_before_code_id_budget": None,
        "non_clone_pairs_after_code_id_budget": None,
        "non_clone_code_ids_after_budget": None,
    }
    filtered_non_clone_code_ids: set[int] = set()
    balanced_class_count = None
    if args.positive_only:
        negatives = []
        wanted_negatives = 0
        candidate_nonclones = set()
        validated_nonclones = set()
        negative_pool_source = "none-positive-only"
        negative_code_pool = []
        negative_pool_capacity = 0
        requested_wanted_negatives = 0
        requested_candidate_count = 0
        candidate_count = 0
    elif args.negative_pool == "false-positives" or args.non_clone_only:
        false_positive_rows, false_positive_code_filter_stats = filter_false_positive_nonclone_rows(
            false_positive_nonclones,
            code_id_set,
            positive_keys,
            getter_setter_code_ids,
            three_line_code_ids,
        )
        filtered_non_clone_code_ids = (
            {left_id for left_id, _, _ in false_positive_rows}
            | {right_id for _, right_id, _ in false_positive_rows}
        )
        if args.all_filtered_non_clone_pairs:
            requested_wanted_negatives = len(false_positive_rows)
            requested_candidate_count = 0
            candidate_count = 0
        elif not args.balance_positive_to_negative_count:
            requested_wanted_negatives = min(wanted_negatives, len(false_positive_rows))
        if args.balance_positive_to_negative_count:
            balanced_class_count = min(len(positives), len(false_positive_rows))
            if len(positives) > balanced_class_count:
                positives = rng.sample(positives, balanced_class_count)
                positives.sort()
                positive_keys = {_normalize_pair(left_id, right_id) for left_id, right_id, _ in positives}
                false_positive_rows, false_positive_code_filter_stats = filter_false_positive_nonclone_rows(
                    false_positive_nonclones,
                    code_id_set,
                    positive_keys,
                    getter_setter_code_ids,
                    three_line_code_ids,
                )
                filtered_non_clone_code_ids = (
                    {left_id for left_id, _, _ in false_positive_rows}
                    | {right_id for _, right_id, _ in false_positive_rows}
                )
            if len(false_positive_rows) > balanced_class_count:
                false_positive_rows = rng.sample(false_positive_rows, balanced_class_count)
                false_positive_rows.sort()
            requested_wanted_negatives = balanced_class_count
        false_positive_nonclones_after_code_filter = len(false_positive_rows)
        if requested_wanted_negatives <= 0:
            non_clone_code_budget_stats = {
                "non_clone_code_id_budget_requested": args.max_non_clone_code_ids,
                "non_clone_pairs_before_code_id_budget": len(false_positive_rows),
                "non_clone_pairs_after_code_id_budget": 0,
                "non_clone_code_ids_after_budget": 0,
            }
            false_positive_rows = []
        elif args.all_filtered_non_clone_pairs:
            non_clone_code_budget_stats = {
                "non_clone_code_id_budget_requested": args.max_non_clone_code_ids,
                "non_clone_pairs_before_code_id_budget": len(false_positive_rows),
                "non_clone_pairs_after_code_id_budget": len(false_positive_rows),
                "non_clone_code_ids_after_budget": len(filtered_non_clone_code_ids),
                "non_clone_code_id_budget_ignored_for_all_pairs": int(args.max_non_clone_code_ids > 0),
            }
        else:
            false_positive_rows, non_clone_code_budget_stats = select_nonclone_pairs_with_code_budget(
                false_positive_rows,
                args.max_non_clone_code_ids,
                requested_wanted_negatives,
                rng,
            )
        if len(false_positive_rows) < requested_wanted_negatives:
            print(
                f"[!] Only {len(false_positive_rows):,} curated false-positive non-clones are usable; "
                f"requested {requested_wanted_negatives:,}. Continuing with all usable curated negatives."
            )
        elif len(false_positive_rows) > requested_wanted_negatives:
            print(
                f"[*] Sampling {requested_wanted_negatives:,} curated false-positive non-clones "
                f"from {len(false_positive_rows):,} usable rows."
            )
            false_positive_rows = rng.sample(false_positive_rows, requested_wanted_negatives)
            false_positive_rows.sort()
        negatives = sorted(false_positive_rows)
        wanted_negatives = len(negatives)
        candidate_nonclones = {_normalize_pair(left_id, right_id) for left_id, right_id, _ in negatives}
        validated_nonclones = candidate_nonclones
        negative_pool_source = "false-positives"
        negative_code_pool = sorted({left_id for left_id, _, _ in negatives} | {right_id for _, right_id, _ in negatives})
        negative_pool_capacity = len(false_positive_rows)
    else:
        negative_code_pool = code_ids
        negative_pool_source = "all-code-ids"
        if args.negative_pool == "positive-code-ids":
            positive_only_pool = sorted(available_positive_code_ids)
            negative_code_pool = positive_only_pool
            negative_pool_source = "positive-code-ids"

        negative_pool_capacity = _nonclone_pair_capacity(negative_code_pool, positive_keys)
        if candidate_count > negative_pool_capacity:
            negative_pool_limit_reason = (
                f"{negative_pool_source} capacity {negative_pool_capacity} is below "
                f"requested candidate_count {candidate_count}; capping candidates to capacity"
            )
            print(f"[!] {negative_pool_limit_reason}")
            candidate_count = negative_pool_capacity

        candidate_nonclones = sample_candidate_nonclones(negative_code_pool, positive_keys, candidate_count, rng)
        validated_nonclones = remove_known_clones(args.dump, candidate_nonclones)

        if len(validated_nonclones) < wanted_negatives:
            print(
                f"[!] Only {len(validated_nonclones):,} validated non-clones are available from "
                f"{negative_pool_source}; requested {wanted_negatives:,}. "
                "Continuing with all available validated non-clones."
            )
            wanted_negatives = len(validated_nonclones)

        negatives = [(left_id, right_id, 0) for left_id, right_id in sorted(validated_nonclones)]
        negatives = rng.sample(negatives, wanted_negatives)

    rows = positives + negatives
    rng.shuffle(rows)

    needed_ids = {left_id for left_id, _, _ in rows} | {right_id for _, right_id, _ in rows}
    if args.write_all_filtered_non_clone_code_ids and filtered_non_clone_code_ids:
        needed_ids.update(filtered_non_clone_code_ids)
    data_jsonl_path = args.output_dir / "data.jsonl"
    missing_cached_code_ids = needed_ids - set(positive_code_map)
    if missing_cached_code_ids:
        print(
            f"[!] {len(missing_cached_code_ids):,} needed code ids were not cached during code scan; "
            "falling back to a final dump scan for data.jsonl."
        )
        written_functions = extract_codes(args.dump, needed_ids, data_jsonl_path)
    else:
        written_functions = write_codes_from_map(positive_code_map, needed_ids, data_jsonl_path)

    write_pairs(args.output_dir / "train.txt", rows)
    positive_train_path = args.output_dir / "train_positives.txt"
    type_labels_path = args.output_dir / "type_labels.tsv"
    for stale_path in (positive_train_path, type_labels_path):
        if stale_path.exists():
            stale_path.unlink()
    total_duration = time.perf_counter() - preparation_start

    train_semantics = (
        "Curated BCB false-positive non-clone split: all filtered usable non-clone pairs."
        if args.non_clone_only and args.all_filtered_non_clone_pairs
        else (
            "Curated BCB false-positive non-clone split sampled up to target_pairs."
            if args.non_clone_only
            else (
                "Mixed benchmark split: all selected positive clone pairs for this type "
                "plus sampled validated non-clone pairs up to target_pairs."
            )
        )
    )
    train_positives_semantics = None
    data_jsonl_semantics = (
        "All functions referenced by filtered curated non-clone pairs."
        if args.non_clone_only and args.write_all_filtered_non_clone_code_ids
        else "Only functions referenced by train.txt rows, not all functions in the BCB dump."
    )

    metadata = {
        "dump_path": str(args.dump.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "target_pairs": args.target_pairs,
        "seed": args.seed,
        "preparation_time_seconds": total_duration,
        "source": "public.pretty_printed_functions + public.clones",
        "train_txt_semantics": train_semantics,
        "train_positives_txt_semantics": train_positives_semantics,
        "type_labels_tsv_semantics": None,
        "data_jsonl_semantics": data_jsonl_semantics,
        "full_clone_type_positive_pairs": full_clone_type_positive_pairs,
        "full_clone_type_unique_code_ids": full_clone_type_code_ids,
        "available_code_ids": len(code_ids),
        **code_filter_stats,
        "keep_getter_setters": bool(args.keep_getter_setters),
        "drop_non_clone_pairs_both_three_line": bool(args.drop_non_clone_pairs_both_three_line),
        "write_all_filtered_non_clone_code_ids": bool(args.write_all_filtered_non_clone_code_ids),
        "all_filtered_non_clone_pairs": bool(args.all_filtered_non_clone_pairs),
        "three_line_code_ids_in_candidate_pool": len(three_line_code_ids),
        "filtered_non_clone_code_ids": len(filtered_non_clone_code_ids),
        "clone_type": args.clone_type,
        "available_positive_clones": total_available_positives,
        "available_positive_code_ids_after_getter_setter_filter": len(available_positive_code_ids),
        "preselect_positive_pairs_before_code_scan": bool(args.preselect_positive_pairs_before_code_scan),
        "preselected_positive_pairs_before_code_scan": preselected_positive_pairs_before_code_scan,
        "cover_positive_code_ids": args.cover_positive_code_ids,
        **positive_coverage_stats,
        "negative_pool": args.negative_pool,
        "positive_only": bool(args.positive_only),
        "non_clone_only": bool(args.non_clone_only),
        "requested_negative_pool": requested_negative_pool,
        "effective_negative_pool": negative_pool_source,
        "negative_pool_limit_reason": negative_pool_limit_reason,
        "negative_pool_code_ids": len(negative_code_pool),
        "negative_pool_pair_capacity": negative_pool_capacity,
        "false_positive_nonclone_pairs_available": false_positive_nonclones_available,
        "false_positive_nonclone_pairs_after_code_filter": false_positive_nonclones_after_code_filter,
        **false_positive_code_filter_stats,
        **non_clone_code_budget_stats,
        "requested_negative_pairs": requested_wanted_negatives,
        "effective_negative_pairs": wanted_negatives,
        "requested_candidate_nonclone_pairs": requested_candidate_count,
        "candidate_nonclone_pairs_requested": candidate_count,
        "candidate_nonclone_pairs_sampled": len(candidate_nonclones),
        "validated_nonclone_pairs": len(validated_nonclones),
        "positive_fraction": args.positive_fraction,
        "max_positive_pairs": max_positive_pairs,
        "balance_positive_to_negative_count": bool(args.balance_positive_to_negative_count),
        "balanced_class_count": balanced_class_count,
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
        "train_positives_pairs": None,
        "non_clones": len(negatives),
        "total_pairs": len(rows),
        "needed_function_ids": len(needed_ids),
        "written_functions": written_functions,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    output_variant = os.getenv("BCB_OUTPUT_VARIANT", str(args.clone_type))
    output_root = output_root_for("bcb", output_variant)
    record_pipeline_timing(
        output_root / "pipeline_timings.json",
        "01_extract_data",
        total_duration,
        {
            "dataset": "bcb",
            "clone_type": args.clone_type,
            "variant": output_variant,
            "data_dir": str(args.output_dir.resolve()),
            "dump_path": str(args.dump.resolve()),
            "functions_written": written_functions,
            "keep_getter_setters": bool(args.keep_getter_setters),
            "drop_non_clone_pairs_both_three_line": bool(args.drop_non_clone_pairs_both_three_line),
            "write_all_filtered_non_clone_code_ids": bool(args.write_all_filtered_non_clone_code_ids),
            "all_filtered_non_clone_pairs": bool(args.all_filtered_non_clone_pairs),
            "three_line_code_ids_in_candidate_pool": len(three_line_code_ids),
            "filtered_non_clone_code_ids": len(filtered_non_clone_code_ids),
            "positive_pairs": len(positives),
            "non_clone_pairs": len(negatives),
            "total_pairs": len(rows),
        },
    )

    print(f"\n[+] Sampled BCB Type-{args.clone_type} benchmark created.")
    print(f"    Output Dir: {args.output_dir}")
    print(f"    Functions written: {written_functions}")
    print(f"    Getter/setter functions detected: {code_filter_stats['detected_getter_setter_functions']}")
    print(f"    Getter/setter functions excluded: {code_filter_stats['excluded_getter_setter_functions']}")
    if args.drop_non_clone_pairs_both_three_line:
        print(f"    Three-line code ids in candidate pool: {len(three_line_code_ids)}")
        print(
            "    Non-clone pairs removed with both sides three-line: "
            f"{false_positive_code_filter_stats['false_positive_nonclone_pairs_removed_by_both_three_line']}"
        )
    print(
        "    Unsupported nested Java method functions excluded: "
        f"{code_filter_stats['excluded_unsupported_nested_java_method_functions']}"
    )
    print(f"    Type-{args.clone_type} clones: {len(positives)}")
    if strict_filter_stats["strict_lecture_filter_enabled"]:
        print(f"    Strict lecture rejected: {strict_filter_stats['strict_lecture_rejected']}")
    print(f"    Effective negative pool: {negative_pool_source} ({len(negative_code_pool)} code ids)")
    print(f"    Non-clones: {len(negatives)}")
    print(f"    Total pairs: {len(rows)}")
    print(f"    Preparation time: {total_duration:.2f}s")


if __name__ == "__main__":
    main()
