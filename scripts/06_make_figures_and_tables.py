#!/usr/bin/env python
"""Create manuscript figures and tables from prepared comparison outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from antarctic_precip_pmb.config import load_yaml
from antarctic_precip_pmb.io import ensure_output_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/example_paths.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths", cfg)
    ensure_output_dirs(paths["figures_dir"], paths["tables_dir"])
    print("Figure directory:", paths["figures_dir"])
    print("Table directory:", paths["tables_dir"])
    print("Figure/table recipes are documented in docs/workflow.md.")
    if args.dry_run:
        return 0
    raise NotImplementedError("Figure migration should proceed panel-by-panel from the source scripts.")


if __name__ == "__main__":
    raise SystemExit(main())
