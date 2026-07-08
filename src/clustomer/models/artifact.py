from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib


def save_artifact(payload: dict[str, Any], path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = {"model", "scaler", "features", "segment_names", "model_version"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artifact thiếu các trường bắt buộc: {sorted(missing)}")
    return artifact
