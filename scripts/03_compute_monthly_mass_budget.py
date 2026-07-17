#!/usr/bin/env python
"""Compute monthly basin PMB after component tables are prepared."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from antarctic_precip_pmb.config import load_yaml
from antarctic_precip_pmb.grace import forward_delta_s
from antarctic_precip_pmb.io import ensure_output_dirs
from antarctic_precip_pmb.pmb import compute_pmb_gt_month


def read_component(path: Path) -> pd.DataFrame:
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/example_paths.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths", cfg)
    out_dir = Path(paths["output_dir"]).expanduser() / "intermediate" / "pmb"
    grace_dir = Path(paths["output_dir"]).expanduser() / "intermediate" / "grace"

    component_paths = {
        "storage": grace_dir / "grace_storage_anomaly_tier1.pkl",
        "discharge": Path(paths.get("monthly_discharge_table", "/path/to/monthly_discharge.csv")).expanduser(),
        "basal_melt": Path(paths.get("monthly_basal_melt_table", "/path/to/monthly_basal_melt.csv")).expanduser(),
        "sublimation": Path(paths.get("monthly_sublimation_table", "/path/to/monthly_sublimation.csv")).expanduser(),
    }
    for name, path in component_paths.items():
        print(f"{name}: {path}")

    if args.dry_run:
        return 0

    missing = [f"{name}: {path}" for name, path in component_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing component table(s):\n" + "\n".join(missing))

    ensure_output_dirs(out_dir)
    storage = read_component(component_paths["storage"])
    delta_s = forward_delta_s(storage)
    discharge = read_component(component_paths["discharge"])
    basal_melt = read_component(component_paths["basal_melt"])
    sublimation = read_component(component_paths["sublimation"])
    pmb = compute_pmb_gt_month(discharge, basal_melt, delta_s, sublimation)
    pmb.to_csv(out_dir / "monthly_pmb_gt_month.csv")
    print("Saved monthly PMB table in Gt/month.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
