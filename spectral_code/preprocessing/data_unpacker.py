import os
import json
from tqdm import tqdm
import re
import javalang


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
    
    # Fallback to simple regex if javalang fails (e.g. missing dependencies/tokens)
    lines = code_str.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('@'): continue
        m = re.search(r'\b(?!(?:if|for|while|switch|catch)\b)(\w+)\s*\(', line)
        if m:
            old_name = m.group(1)
            if old_name[0].isupper():
                lines[i] = line[:m.start(1)] + f"void {new_name}" + line[m.end(1):]
            else:
                lines[i] = line[:m.start(1)] + new_name + line[m.end(1):]
            return '\n'.join(lines)
    return re.sub(r'\b(\w+)\s*\(', f'{new_name}(', code_str, count=1)

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

            # Safely rename the method to m_{idx} and bypass constructor missing-return-type restrictions
            code = rename_java_method(code, idx)

            # Use the original outer class name when code contains OuterClass.this.
            # Otherwise Joern sees invalid Java and exports empty graphs for that method.
            wrapper_class = infer_wrapper_class_name(code, idx)

            # Enclose in a consistent synthetic class.
            java_content = f"class {wrapper_class} {{\n"
            java_content += f"    {code}\n"
            java_content += "}\n"

            out_path = os.path.join(output_tmp_dir, f"m_{idx}.java")
            with open(out_path, "w", encoding="utf-8") as jf:
                jf.write(java_content)
            method_ids.append(idx)
            iterator.set_postfix(methods=len(method_ids), skipped=skipped_count, refresh=False)

    if skipped_file:
        skipped_file.close()

    return method_ids
