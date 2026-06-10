#!/usr/bin/env python
"""Prepare and validate basin definitions for the study workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from antarctic_precip_pmb.basins import load_basin_grid, validate_basin_mapping, validate_region_definitions
from antarctic_precip_pmb.config import load_yaml
from antarctic_precip_pmb.constants import CRS_SH_STEREO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/example_paths.yaml")
    parser.add_argument("--basin-file", default="bedmap3_basins_0.1deg.tif")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths", cfg)
    basin_path = Path(paths["basins_dir"]).expanduser() / args.basin_file

    validate_basin_mapping()
    validate_region_definitions()

    if args.dry_run:
        print(f"Would load basin grid: {basin_path}")
        return 0

    basins = load_basin_grid(basin_path, CRS_SH_STEREO)
    print("Loaded basin grid.")
    print("dims:", dict(basins.sizes))
    print("crs:", basins.rio.crs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
