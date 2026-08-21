import os
import json
from tqdm import tqdm
import re
import javalang

from spectral_code.preprocessing.language_support import (
    LANGUAGE_SUPPORT,
    normalize_source_language,
    source_extension,
)


LANGUAGE_EXTENSIONS = {
    name: support.extension for name, support in LANGUAGE_SUPPORT.items()
}


def infer_wrapper_class_name(code_str: str, idx: str) -> str:
    qualified_this = re.search(r'\b([A-Z][A-Za-z0-9_]*)\.this\b', code_str)
    if qualified_this:
        return qualified_this.group(1)
    return f"Wrapper_{idx}"


def _count_jsonl_records(data_file: str) -> int:
    total_bytes = os.path.getsize(data_file)
    line_count = 0
    with open(data_file, "rb") as f, tqdm(
        total=total_bytes,
        desc="Counting JSONL records",
        unit="B",
        unit_scale=True,
    ) as pbar:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            line_count += chunk.count(b"\n")
            pbar.update(len(chunk))
    return line_count


def _method_size_metrics(code: str) -> dict[str, int]:
    lines = code.splitlines() or [""]
    return {
        "chars": len(code),
        "lines": len(lines),
        "longest_line": max(len(line) for line in lines),
    }


def _skip_reasons(
    metrics: dict[str, int],
    max_lines: int = 0,
    max_chars: int = 0,
    max_longest_line: int = 0,
) -> list[str]:
    reasons = []
    if max_lines > 0 and metrics["lines"] > max_lines:
        reasons.append(f"lines>{max_lines}")
    if max_chars > 0 and metrics["chars"] > max_chars:
        reasons.append(f"chars>{max_chars}")
    if max_longest_line > 0 and metrics["longest_line"] > max_longest_line:
        reasons.append(f"longest_line>{max_longest_line}")
    return reasons


def rename_java_method(code_str: str, idx: str) -> str:
    new_name = f"m_{idx}"
    wrapped = f'class Wrapper_{idx} {{\n{code_str}\n}}'
    try:
        tree = javalang.parse.parse(wrapped)
        target_node = None
        is_constructor = False
        
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            # Only rename top-level method if it's not a generic name yet
            # and ignore inner methods/anonymous classes
            if len(path) <= 3: # Typically tree -> ClassDeclaration -> MethodDeclaration
                target_node = node
                break
        
        if not target_node:
            for path, node in tree.filter(javalang.tree.ConstructorDeclaration):
                if len(path) <= 3:
                    target_node = node
                    is_constructor = True
                    break
        
        if target_node:
            line = target_node.position.line - 1
            old_name = target_node.name
            lines = wrapped.split('\n')
            
            # Find the old_name in the line
            match = re.search(r'\b' + re.escape(old_name) + r'\b\s*\(', lines[line])
            if match:
                idx_in_line = match.start()
                if is_constructor:
                    # turn into standard method so it doesn't fail Joern's syntax checks
                    lines[line] = lines[line][:idx_in_line] + f"void {new_name}" + lines[line][idx_in_line + len(old_name):]
                else:
                    lines[line] = lines[line][:idx_in_line] + new_name + lines[line][idx_in_line + len(old_name):]
                    
                return '\n'.join(lines[1:-1])
    except Exception:
        pass
    
    # Fallback to a signature-aware regex if javalang fails. Method names can
    # start with uppercase letters in BCB, so uppercase alone is not a constructor
    # signal. Treat it as a constructor only when no return type is present.
    lines = code_str.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('@'): continue
        m = re.search(
            r'^(?P<prefix>\s*(?:(?:public|protected|private|static|final|synchronized|abstract|native|strictfp)\s+)*)'
            r'(?:(?P<return_type>[\w<>\[\].?,\s]+?)\s+)?'
            r'(?P<name>[A-Za-z_$][\w$]*)\s*\(',
            line,
        )
        if m:
            old_name = m.group("name")
            if old_name in {"if", "for", "while", "switch", "catch"}:
                continue
            if m.group("return_type"):
                lines[i] = line[:m.start("name")] + new_name + line[m.end("name"):]
            else:
                lines[i] = line[:m.start("name")] + f"void {new_name}" + line[m.end("name"):]
            return '\n'.join(lines)
    return re.sub(r'\b(\w+)\s*\(', f'{new_name}(', code_str, count=1)


def _normalize_language(raw: str | None) -> str:
    return normalize_source_language(raw, default="java")


def _wrap_java(code: str, idx: str) -> str:
    code = rename_java_method(code, idx)
    wrapper_class = infer_wrapper_class_name(code, idx)
    return f"class {wrapper_class} {{\n    {code}\n}}\n"


def _wrap_csharp(code: str, idx: str) -> str:
    return f"class Wrapper_{idx} {{\n    {code}\n}}\n"


def _render_source_record(
    code: str,
    idx: str,
    language: str,
    *,
    source_mode: str | None = None,
) -> tuple[str, str]:
    language = _normalize_language(language)
    ext = source_extension(language)

    if language == "java" and source_mode != "compilation_unit":
        content = _wrap_java(code, idx)
    elif language == "csharp" and source_mode != "compilation_unit":
        content = _wrap_csharp(code, idx)
    else:
        content = code.rstrip() + "\n"

    return content, ext

def unpack_jsonl_to_java(
    data_file: str,
    output_tmp_dir: str,
    max_lines: int = 0,
    max_chars: int = 0,
    max_longest_line: int = 0,
    skipped_report_path: str | None = None,
):
    if not os.path.exists(data_file):
        print(f"[-] Data file not found: {data_file}")
        return []

    total_records = _count_jsonl_records(data_file)
    method_ids = []
    skipped_count = 0

    skipped_file = None
    if skipped_report_path:
        os.makedirs(os.path.dirname(skipped_report_path), exist_ok=True)
        skipped_file = open(skipped_report_path, "w", encoding="utf-8")

    try:
        f = open(data_file, "r", encoding="utf-8")
        iterator_context = f
    except Exception:
        if skipped_file:
            skipped_file.close()
        raise

    with iterator_context as f:
        iterator = tqdm(f, total=total_records, desc="Unpacking Java files", unit="method")
        for line in iterator:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            idx = str(record.get("idx", "unknown"))
            code = record.get('func', '')
            language = _normalize_language(record.get("lang"))
            source_mode = str(record.get("source_mode") or "").strip().lower() or None

            metrics = _method_size_metrics(code)
            reasons = _skip_reasons(metrics, max_lines, max_chars, max_longest_line)
            if reasons:
                skipped_count += 1
                if skipped_file:
                    skipped_file.write(json.dumps({
                        "idx": idx,
                        "reasons": reasons,
                        **metrics,
                    }, ensure_ascii=False))
                    skipped_file.write("\n")
                iterator.set_postfix(methods=len(method_ids), skipped=skipped_count, refresh=False)
                continue

            source_content, ext = _render_source_record(code, idx, language, source_mode=source_mode)
            out_path = os.path.join(output_tmp_dir, f"m_{idx}{ext}")
            with open(out_path, "w", encoding="utf-8") as jf:
                jf.write(source_content)
            method_ids.append(idx)
            iterator.set_postfix(methods=len(method_ids), skipped=skipped_count, refresh=False)

    if skipped_file:
        skipped_file.close()

    return method_ids
