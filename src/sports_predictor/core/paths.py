"""Project paths.

Centralizes where data lives so every module reads and writes the same places.
Paths are derived from this file's location, so they work no matter what the
current working directory is when code runs.
"""

from __future__ import annotations

from pathlib import Path

# core/paths.py -> core -> sports_predictor -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
