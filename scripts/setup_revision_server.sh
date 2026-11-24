#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

server_env="${SPECTRA_VENV:-.venv-server}"
python3 -m venv --system-site-packages "$server_env"
server_python="$server_env/bin/python"
if ! "$server_python" -c 'import torch; assert torch.cuda.is_available() and tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 6)' 2>/dev/null; then
    "$server_python" -m pip install --upgrade "torch==${SPECTRA_TORCH_VERSION:-2.8.0}" \
        --index-url "${SPECTRA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
fi
"$server_python" -m pip install -r revision/requirements-server.txt
"$server_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'

if [[ "${SPECTRA_WARM_MODELS:-0}" == 1 ]]; then
    "$server_python" - <<'PY'
from huggingface_hub import snapshot_download

models = {
    "microsoft/codebert-base": "3b0952feddeffad0063f274080e3c23d75e7eb39",
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
}
for name, revision in models.items():
    print(snapshot_download(repo_id=name, revision=revision, allow_patterns=[
        "*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.jinja",
    ]))
PY
fi
