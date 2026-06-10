# Antarctic Snowfall PMB Benchmark Workflow

This repository contains the workflow for the manuscript:

**Benchmarking Antarctic Snowfall Products With a Basin-Scale Mass-Budget
Constraint: ERA5, GPCP V3.3, and GPM Passive Microwave Estimates**.

The analysis builds a basin-scale mass-budget precipitation estimate (`PMB`) and
uses it as an independent benchmark for Antarctic snowfall products during
2013-2020. The evaluated products are ERA5, GPCP V3.3, and GPM passive
microwave (`PMW`) Version 7 and Version 8 estimates.

## Study Summary

Accurate Antarctic snowfall estimates are needed for ice-sheet mass balance,
sea-level change, and high-latitude water-cycle studies, but direct
precipitation observations over the Antarctic Ice Sheet are sparse. This
workflow estimates snowfall indirectly from the cryospheric mass budget and then
compares gridded precipitation products against that basin-scale constraint.

The manuscript reports that ERA5 agrees most closely with PMB over the full
Antarctic Ice Sheet, with a 2013-2020 mean of about **175 mm/yr** compared with
**171 mm/yr** for PMB and a basin-scale correlation of **0.96**. GPCP V3.3
captures the broad accumulation pattern but is moderately drier, with a mean of
about **159 mm/yr** and basin-scale correlation of **0.85**. GPM PMW V07 strongly
underestimates Antarctic snowfall, near **25 mm/yr**, while GPM PMW V08 improves
the constellation mean to about **131 mm/yr**, increases basin-scale correlation
to **0.98**, and reduces the dry bias from roughly **-84%** to **-28%**.

## Mass-Budget Framework

The workflow estimates basin-scale mass-budget precipitation using:

```text
P_MB = discharge + basal_melt + deltaS + sublimation
```

where `deltaS` is the monthly basin storage change from GRACE/GRACE-FO,
`discharge` is ice export across grounding-line flux gates, `basal_melt` is
included with lateral mass loss, and `sublimation` is the RACMO2.4p1
sublimation loss term. Evapotranspiration and runoff are neglected in the
manuscript because they are generally small over the grounded Antarctic Ice
Sheet, so Antarctic precipitation is treated as snowfall.

The workflow uses monthly GRACE/altimetry storage anomaly data and computes:

```text
deltaS_m = S_{m+1} - S_m
```

with `deltaS_m` assigned to the starting month `m`. Annual discharge and
basal-melt estimates are distributed uniformly across the months of each
calendar year so all terms can be evaluated on the monthly PMB time step.
RACMO2.4p1 `subltot` is converted into a positive sublimation loss term before
being added to the mass budget.

## Domain And Diagnostics

The analysis uses the Rignot/IMBIE Antarctic drainage-basin framework. The
source mask contains 19 labeled sectors, including one island sector; this
workflow focuses on the 18 grounded Antarctic drainage basins, IDs 2-19.
Regional aggregation is performed for:

- Full Antarctic Ice Sheet (`AIS`)
- West Antarctica (`WAIS`), with Antarctic Peninsula sectors grouped into WAIS
- East Antarctica (`EAIS`)

Regional means use cosine-latitude weighting to account for grid-cell area
variation. Product evaluation is organized around mean annual basin maps,
regional annual means, seasonal climatologies, annual and seasonal time series,
basin-scale scatter comparisons, and basin-ranked product spread.

## Repository Layout

The original analysis scripts are preserved at the repository root. The cleaned,
coauthor-friendly workflow is organized under:

- `src/antarctic_precip_pmb/`: reusable, import-safe Python modules.
- `scripts/`: runnable workflow steps.
- `config/`: example paths and study parameters.
- `docs/`: workflow notes and manuscript-method assumptions.
- `data/`: documentation for required external inputs.
- `outputs/`: documentation for generated outputs.
- `tests/`: lightweight validation tests.

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

## Required Data

Large scientific inputs are not committed to this repository. The workflow
expects local paths to:

- IMBIE/Rignot basin mask, especially `bedmap3_basins_0.1deg.tif`.
- GRACE/GRACE-FO basin storage-change workbook.
- Annual discharge and basal-melt workbook.
- RACMO2.4p1 Antarctic sublimation data.
- ERA5 monthly precipitation.
- GPCP V3.3 monthly precipitation.
- GPM PMW V07 and V08 constellation products.
- UA-HIPA and CloudSat products used in supporting comparisons.

Generated NetCDF, HDF, GeoTIFF, Excel, pickle, CSV, and figure outputs should
remain outside Git history.
