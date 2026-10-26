"""Streamlit Community Cloud Root Entrypoint for Higgs Event Lab.

Main web entrypoint deployed on Streamlit Community Cloud for the mid-semester evaluation.
Automatically bootstraps verified mid-semester benchmark artifacts when running in a fresh
cloud container where outputs/ is excluded by .gitignore, and runs the HiggsLens scientific dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def bootstrap_cloud_artifacts_if_needed() -> None:
    """Ensure benchmark artifacts exist so hosted web visitors see genuine results."""
    outputs_dir = REPO_ROOT / "outputs"
    has_valid_runs = False
    if outputs_dir.exists():
        for item in outputs_dir.iterdir():
            if item.is_dir() and (item / "metrics.json").is_file():
                has_valid_runs = True
                break

    if not has_valid_runs:
        try:
            from scripts.generate_sample_artifacts import main as generate_artifacts

            generate_artifacts()
        except Exception as exc:  # pragma: no cover
            # Fall back gracefully; app.main will display honest empty state
            print(f"Notice: Artifact bootstrap skipped: {exc}")


# Run bootstrap before launching dashboard
bootstrap_cloud_artifacts_if_needed()

# Launch HiggsLens Dashboard
from app.main import main

if __name__ == "__main__":
    main()
