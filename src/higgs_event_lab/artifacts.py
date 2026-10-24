"""Shared artifact schema helpers for model training and the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from higgs_event_lab.protocol import (
    ARTIFACT_SCHEMA_VERSION,
    new_experiment_artifact,
)


def save_artifact_json(artifact: dict[str, Any], path: str | Path) -> Path:
    """Save an experiment artifact dict to a formatted JSON file."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return target_path


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "new_experiment_artifact",
    "save_artifact_json",
]
