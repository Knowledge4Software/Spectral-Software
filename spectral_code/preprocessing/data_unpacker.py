import os
import json
from tqdm import tqdm

def unpack_jsonl_to_java(data_file: str, output_tmp_dir: str):
    if not os.path.exists(data_file):
        print(f"[-] Data file not found: {data_file}")
        return []
        
    with open(data_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    method_ids = []
    for line in tqdm(lines, desc="Unpacking"):
        record = json.loads(line)
        idx = str(record.get("idx", "unknown"))
        out_path = os.path.join(output_tmp_dir, f"Method_{idx}.java")
        with open(out_path, "w", encoding="utf-8") as jf:
            jf.write(f"public class Method_{idx} {{\n{record.get('func', '')}\n}}")
        method_ids.append(idx)
        
    return method_ids
