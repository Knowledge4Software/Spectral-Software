from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from spectral_code.utils.dataset_paths import OUTPUTS_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_python_pipeline(
    script_relative_path: str,
    script_args: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items()})

    script_path = PROJECT_ROOT / script_relative_path
    workdir = Path(cwd) if cwd is not None else PROJECT_ROOT
    command = [sys.executable, str(script_path)]
    if script_args:
        command.extend(str(arg) for arg in script_args)

    return subprocess.run(
        command,
        cwd=str(workdir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def output_root_for_dataset(dataset_name: str) -> Path:
    return OUTPUTS_ROOT / dataset_name
