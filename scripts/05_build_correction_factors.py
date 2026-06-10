#!/usr/bin/env python
"""Compute PMB-based seasonal scalar correction factors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from antarctic_precip_pmb.config import load_yaml
from antarctic_precip_pmb.correction_factors import scalar_correction_factors
from antarctic_precip_pmb.io import ensure_output_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/example_paths.yaml")
    parser.add_argument("--input", help="Seasonal regional product table CSV.")
    parser.add_argument("--reference-product", default="$P_{\\mathrm{MB}}$")
    parser.add_argument("--target-product", default="UA-HIPA")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths", cfg)
    out_dir = Path(paths["tables_dir"]).expanduser()
    input_path = Path(args.input).expanduser() if args.input else None
    print(f"Input table: {input_path or 'not provided'}")
    print(f"Output directory: {out_dir}")
    if args.dry_run:
        return 0
    if input_path is None or not input_path.exists():
        raise FileNotFoundError("Provide --input pointing to a seasonal regional product table.")

    ensure_output_dirs(out_dir)
    table = pd.read_csv(input_path)
    factors = scalar_correction_factors(
        table,
        ref_product=args.reference_product,
        target_product=args.target_product,
    )
    out_file = out_dir / "pmb_scalar_correction_factors.csv"
    factors.to_csv(out_file, index=False)
    print(f"Saved correction factors: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
