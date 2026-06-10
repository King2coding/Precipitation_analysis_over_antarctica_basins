"""Small IO helpers shared by workflow scripts."""

from __future__ import annotations

from pathlib import Path


def ensure_output_dirs(*paths: str | Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)


def require_existing(path: str | Path, label: str) -> Path:
    """Return a path if it exists, otherwise raise a clear error."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
