"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_PATH_KEYS = (
    "basins_dir",
    "racmo_dir",
    "grace_excel",
    "discharge_excel",
    "gpcp_monthly_dir",
    "era5_monthly_file",
    "gpm_root_dir",
    "uahipa_monthly_file",
    "cloudsat_dir",
    "output_dir",
    "figures_dir",
    "tables_dir",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read workflow config files.") from exc

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def missing_path_keys(config: dict[str, Any]) -> list[str]:
    """Return required path keys absent from a paths config."""
    paths = config.get("paths", config)
    return [key for key in REQUIRED_PATH_KEYS if key not in paths]


def is_placeholder(value: Any) -> bool:
    """Detect example values that should not be treated as real paths."""
    if value is None:
        return True
    text = str(value)
    return text.startswith("/path/to/") or text in {"", "TODO", "CHANGE_ME"}


def path_status(config: dict[str, Any]) -> list[dict[str, str]]:
    """Report whether configured input/output paths exist.

    Placeholder paths are reported separately so the example config can be
    checked without requiring local study data.
    """
    paths = config.get("paths", config)
    rows: list[dict[str, str]] = []
    for key in REQUIRED_PATH_KEYS:
        value = paths.get(key)
        if is_placeholder(value):
            status = "placeholder"
        else:
            status = "exists" if Path(value).expanduser().exists() else "missing"
        rows.append({"key": key, "path": "" if value is None else str(value), "status": status})
    return rows
