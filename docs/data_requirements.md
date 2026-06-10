# Data Requirements

The workflow depends on external research data that should not be committed.

## Basin And Budget Inputs

- `bedmap3_basins_0.1deg.tif`
- `DataCombo_RignotBasins_update.xlsx`
- `antarctic_discharge_2013-2022_imbie.xlsx`
- RACMO2.4p1 monthly `subltot`

## Precipitation Products

- ERA5 monthly total precipitation.
- GPCP v3.3 monthly files.
- GPM passive microwave constellation V07 and V08 monthly files.
- UA-HIPA monthly 2013-2020 product.
- CloudSat Antarctic monthly, seasonal, and annual climatology files.

## Output Policy

Keep generated NetCDF, HDF, GeoTIFF, CSV, pickle, Excel, and figure products
outside Git history. Use `outputs/` or a configured external output directory.
