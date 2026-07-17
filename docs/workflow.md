# Antarctic Precipitation-PMB Workflow

This repository now separates the shareable workflow from the original study
script archive. Existing scripts remain read-only source material; new reusable
logic lives in `src/antarctic_precip_pmb/`.

## Scientific Flow

1. Prepare basin data.
   - Load `bedmap3_basins_0.1deg.tif`.
   - Keep IMBIE/Rignot basin IDs 2-19.
   - Define Antarctica, West Antarctica, and East Antarctica regions.

2. Fill GRACE/altimetry storage anomaly data.
   - Read `DataCombo_RignotBasins_update.xlsx`.
   - Use the storage anomaly sheet `Basin_Timeseries (Gt) Update`.
   - Build a full monthly 2013-2020 table.
   - Fill storage anomaly gaps using deseasonalized linear interpolation.
   - Fill 1-sigma uncertainty using direct linear interpolation.

3. Compute monthly mass-budget precipitation.
   - Use the source-script equation:
     `P_MB = discharge + basal_melt + deltaS + sublimation`.
   - Use forward deltaS convention:
     `deltaS_m = S_{m+1} - S_m`, assigned to month `m`.
   - Distribute annual discharge and basal melt to monthly values when required.
   - Regrid RACMO2.4p1 `subltot` to the basin grid and treat sublimation as a
     positive loss term.

4. Compare precipitation products.
   - Remap PMB, ERA5, GPCP v3.3, GPM PMW V07, GPM PMW V08, and UA-HIPA to a
     common basin-comparison grid.
   - Build monthly, seasonal, annual, basin, and regional summaries.
   - Include CloudSat 2007-2010 climatology as a separate comparison.

5. Build correction factors.
   - Compute scalar seasonal correction factors as `PMB / target_product`.
   - Current source scripts focus on UA-HIPA and GPM PMW V07/V08 correction
     experiments.

6. Make figures and tables.
   - Generate manuscript figures from prepared intermediate tables.
   - Keep generated figures and tables out of Git.

## Command Outline

```bash
python scripts/check_setup.py --config config/example_paths.yaml
python scripts/01_prepare_basin_data.py --config config/paths.yaml
python scripts/02_fill_grace_storage.py --config config/paths.yaml
python scripts/03_compute_monthly_mass_budget.py --config config/paths.yaml
python scripts/04_compare_precip_products.py --config config/paths.yaml
python scripts/05_build_correction_factors.py --config config/paths.yaml --input outputs/tables/seasonal_regional_products.csv
python scripts/06_make_figures_and_tables.py --config config/paths.yaml
```

Use `--dry-run` on each script to verify configured paths and intended actions
without processing data.
