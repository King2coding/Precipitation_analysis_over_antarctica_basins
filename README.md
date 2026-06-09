# Antarctic Precipitation PMB Workflow

This repository contains the Antarctic basin precipitation and
precipitation-minus-balance/mass-budget workflow used to compare precipitation
products against a basin-scale PMB estimate.

The original analysis scripts are preserved at the repository root. The cleaned,
coauthor-friendly workflow is being organized under:

- `src/antarctic_precip_pmb/`: reusable, import-safe Python modules.
- `scripts/`: runnable workflow steps.
- `config/`: example paths and study parameters.
- `docs/`: workflow notes and manuscript-method assumptions.
- `data/`: documentation for required external inputs.
- `outputs/`: documentation for generated outputs.
- `tests/`: lightweight validation tests.

## Scientific Summary

The workflow estimates basin-scale mass-budget precipitation using:

```text
P_MB = discharge + basal_melt + deltaS + sublimation
```

The inspected source scripts use monthly GRACE/altimetry storage anomaly data
and compute:

```text
deltaS_m = S_{m+1} - S_m
```

with `deltaS_m` assigned to the starting month `m`. RACMO2.4p1 sublimation is
converted into a positive loss term before being added to the mass budget.

The main study period is 2013-2020. Basin IDs 2-19 are used, with West
Antarctica defined as IDs 10-17 and East Antarctica as IDs 2-9 plus 18-19.

## Setup

Create an environment with the packages in `environment.yml`, then copy the
example path config:

```bash
cp config/example_paths.yaml config/paths.yaml
python scripts/check_setup.py --config config/paths.yaml
```

Edit `config/paths.yaml` to point to local data. Large data and generated
outputs must remain outside Git history.

## Workflow

```bash
python scripts/01_prepare_basin_data.py --config config/paths.yaml
python scripts/02_fill_grace_storage.py --config config/paths.yaml
python scripts/03_compute_monthly_mass_budget.py --config config/paths.yaml
python scripts/04_compare_precip_products.py --config config/paths.yaml
python scripts/05_build_correction_factors.py --config config/paths.yaml --input outputs/tables/seasonal_regional_products.csv
python scripts/06_make_figures_and_tables.py --config config/paths.yaml
```

Each script supports `--dry-run` for path and action checks.

## Current Caveat

The manuscript file `Antarctic_Snowfall-PMB-method-V0.docx` was not available
during this reorganization phase. Method notes in `docs/` are therefore derived
from the source scripts and should be reconciled with the manuscript before the
workflow is considered publication-final.
