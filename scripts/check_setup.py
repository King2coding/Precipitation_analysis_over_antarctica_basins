#!/usr/bin/env python
"""Validate the local environment and workflow configuration."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

from antarctic_precip_pmb.basins import validate_basin_mapping, validate_region_definitions
from antarctic_precip_pmb.config import load_yaml, missing_path_keys, path_status


REQUIRED_IMPORTS = (
    "numpy",
    "pandas",
    "xarray",
    "scipy",
    "matplotlib",
    "yaml",
)

OPTIONAL_GEOSPATIAL_IMPORTS = (
    "rioxarray",
    "rasterio",
    "cartopy",
    "pyproj",
)


def check_imports(names: tuple[str, ...]) -> list[str]:
    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/example_paths.yaml", help="Path config YAML.")
    parser.add_argument(
        "--strict-paths",
        action="store_true",
        help="Fail if configured non-placeholder data paths do not exist.",
    )
    args = parser.parse_args()

    validate_basin_mapping()
    validate_region_definitions()

    missing_required = check_imports(REQUIRED_IMPORTS)
    if missing_required:
        print("Missing required Python packages:", ", ".join(missing_required))
        return 1

    missing_geo = check_imports(OPTIONAL_GEOSPATIAL_IMPORTS)
    if missing_geo:
        print("Missing geospatial packages needed for full data runs:", ", ".join(missing_geo))

    cfg = load_yaml(args.config)
    missing_keys = missing_path_keys(cfg)
    if missing_keys:
        print("Missing config path keys:", ", ".join(missing_keys))
        return 1

    statuses = path_status(cfg)
    for row in statuses:
        print(f"{row['key']}: {row['status']} ({row['path']})")

    if args.strict_paths and any(row["status"] == "missing" for row in statuses):
        return 1

    print("Setup check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
