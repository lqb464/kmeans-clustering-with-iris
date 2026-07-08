from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    country: str
    country_display: str
    development_cutoff: str
    validation_end: str
    holdout_end: str


@dataclass(frozen=True)
class ModelConfig:
    algorithm: str
    n_clusters: int
    features: tuple[str, ...]
    transform: str
    random_state: int
    manual_review_quantile: float
    artifact_path: str
    segment_names_path: str


@dataclass(frozen=True)
class GovernanceConfig:
    min_segment_share: float
    max_payload_rows: int
    drift_psi_warning: float


@dataclass(frozen=True)
class Settings:
    project: dict[str, Any]
    data: DataConfig
    model: ModelConfig
    governance: GovernanceConfig


def load_settings(path: str | Path = "configs/config.yaml") -> Settings:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model = dict(payload["model"])
    model["features"] = tuple(model["features"])
    return Settings(
        project=payload["project"],
        data=DataConfig(**payload["data"]),
        model=ModelConfig(**model),
        governance=GovernanceConfig(**payload["governance"]),
    )
