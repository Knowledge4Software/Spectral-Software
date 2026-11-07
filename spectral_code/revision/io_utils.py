"""Atomic replacement with a bounded retry for transient Windows file locks."""
from __future__ import annotations

import os
import time
from pathlib import Path


def replace_file(source: Path, destination: Path):
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.025 * 2**attempt)
