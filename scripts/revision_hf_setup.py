"""Generate the supported Hugging Face environment setup, outside training code."""
from __future__ import annotations

HF_PACKAGES = {
    "transformers": "==4.57.3",
    "huggingface-hub": "==0.36.0",
    "tokenizers": "==0.22.1",
}
LORA_PACKAGES = {"peft": "==0.17.1", "accelerate": "==1.10.1"}


def notebook_setup(*, require_lora=False):
    requirements = {**HF_PACKAGES, **(LORA_PACKAGES if require_lora else {})}
    return '''# Supported HF environment; preserve Kaggle's installed CUDA PyTorch.
def _spectra_prepare_hf():
    import importlib
    from importlib import metadata
    import json
    from pathlib import Path
    import subprocess
    import sys
    import tempfile
    from packaging.specifiers import SpecifierSet

    required = ''' + repr(requirements) + '''
    def installed(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    needs = [name for name, spec in required.items()
             if installed(name) is None or installed(name) not in SpecifierSet(spec)]
    if needs:
        changed_modules = {name.replace("-", "_") for name in needs}
        already_loaded = any(name.split(".")[0] in changed_modules for name in sys.modules)
        torch_version = metadata.version("torch")
        with tempfile.TemporaryDirectory(prefix="spectra_hf_") as directory:
            constraints = Path(directory) / "keep_kaggle_torch.txt"
            constraints.write_text("torch==" + torch_version + "\\n", encoding="utf-8")
            subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                            "--constraint", str(constraints),
                            *(name + spec for name, spec in required.items())], check=True)
        importlib.invalidate_caches()
        if already_loaded:
            raise RuntimeError("HF packages were corrected, but old modules are loaded. "
                               "Restart the kernel/session and Run All, or use a fresh Save & Run All. "
                               "Do not continue this kernel after changing packages.")
    actual = {name: installed(name) for name in required}
    if any(actual[name] is None or actual[name] not in SpecifierSet(spec)
           for name, spec in required.items()):
        raise RuntimeError("The supported HF environment could not be installed: " + str(actual))
    actual["torch"] = metadata.version("torch")
    print(json.dumps({"status": "hf_environment_ready", "versions": actual}), flush=True)

_spectra_prepare_hf()
'''
