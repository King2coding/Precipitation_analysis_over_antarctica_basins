#!/usr/bin/env python
"""Build product-comparison tables after PMB and product fields are prepared."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from antarctic_precip_pmb.config import load_yaml
from antarctic_precip_pmb.constants import REGION_BASINS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/example_paths.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    products = cfg.get("products", {})
    print("Configured regions:", ", ".join(REGION_BASINS))
    print("Configured products:", ", ".join(products) if products else "none")
    print("This step expects gridded monthly products remapped to the common basin grid.")
    if args.dry_run:
        return 0
    raise NotImplementedError(
        "Full product comparison requires local product data paths. "
        "Use the source scripts as scientific reference while migrating each product loader."
    )


if __name__ == "__main__":
    raise SystemExit(main())
