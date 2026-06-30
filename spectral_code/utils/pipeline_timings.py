from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_pipeline_timing(
    timings_path: Path,
    stage: str,
    seconds: float,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    if timings_path.exists():
        with timings_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"stages": {}}

    payload.setdefault("stages", {})[stage] = {
        "seconds": seconds,
        "minutes": seconds / 60,
        "updated_at_utc": _utc_now(),
        **(details or {}),
    }
    payload["updated_at_utc"] = _utc_now()

    with timings_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
