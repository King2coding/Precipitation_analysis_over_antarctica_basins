#!/usr/bin/env python
"""Fill GRACE/altimetry monthly storage anomaly and uncertainty tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from antarctic_precip_pmb.config import load_yaml
from antarctic_precip_pmb.constants import YEAR_END, YEAR_START
from antarctic_precip_pmb.grace import (
    fill_storage_deseasonalized_linear,
    fill_uncertainty_linear,
    monthly_storage_table,
)
from antarctic_precip_pmb.io import ensure_output_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/example_paths.yaml")
    parser.add_argument("--storage-sheet", default="Basin_Timeseries (Gt) Update")
    parser.add_argument("--uncertainty-sheet", default="1-sigma_Error(Gt) Update")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths", cfg)
    out_dir = Path(paths["output_dir"]).expanduser() / "intermediate" / "grace"
    excel = Path(paths["grace_excel"]).expanduser()

    print(f"GRACE workbook: {excel}")
    print(f"Output directory: {out_dir}")
    if args.dry_run:
        return 0

    ensure_output_dirs(out_dir)
    start = f"{YEAR_START}-01-01"
    end = f"{YEAR_END}-12-01"

    storage_raw = pd.read_excel(excel, sheet_name=args.storage_sheet)
    storage = monthly_storage_table(storage_raw, start_date=start, end_date=end)
    storage_filled = fill_storage_deseasonalized_linear(storage)
    storage_filled.to_pickle(out_dir / "grace_storage_anomaly_tier1.pkl")

    uncertainty_raw = pd.read_excel(excel, sheet_name=args.uncertainty_sheet)
    uncertainty = monthly_storage_table(uncertainty_raw, start_date=start, end_date=end)
    uncertainty_filled = fill_uncertainty_linear(uncertainty)
    uncertainty_filled.to_pickle(out_dir / "grace_storage_uncertainty_tier1.pkl")

    print("Saved filled GRACE storage anomaly and uncertainty tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
