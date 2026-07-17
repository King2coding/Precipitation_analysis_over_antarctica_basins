# Original Script Inventory

The original scripts are preserved as source material. The cleaned workflow
should migrate logic into side-effect-free modules and command-line scripts.

## Central Workflow Scripts

- `Monthly_mass_budget_precip_RignotBasin_07May2026_old_analogous.py`
  - Main PMB construction.
  - Basin grid setup.
  - GRACE/storage anomaly to deltaS.
  - Discharge, basal melt, RACMO sublimation.
  - PMB maps, uncertainty, annual/seasonal/monthly summaries.

- `Comparative_assessment_of_precip_products_with_PMB_over_antarctica_basins_GPM_V7_V8_26May2026.py`
  - Main product comparison.
  - PMB, ERA5, GPCP v3.3, UA-HIPA, GPM V07/V08, CloudSat.
  - Regional and basin comparisons, anomalies, figures, and tables.

- `Filling_the_Gaps_in_Grace_data.py`
  - Focused GRACE/altimetry storage anomaly gap filling.
  - Tier 1 deseasonalized linear interpolation.
  - Tier 2 harmonic diagnostic.
  - 1-sigma uncertainty gap filling.

- `build_uahipa_pmb_correction_factors.py`
  - PMB-based seasonal correction factors.
  - UA-HIPA grid correction-factor products.

## Helper-Heavy Scripts

- `program_utils.py`
  - Large mixed utility module with imports, constants, basin mappings,
    product processors, statistics, aggregation, and plotting.

- `program_utile_13Apr2026.py`
  - Additional basin/product processing functions, GPM V07/V08 helpers,
    correction-factor helpers, and plotting routines.

- `Extra_util_functions.py`
  - Additional PMB diagnostics, GRACE uncertainty, basin/month masking,
    aggregation, and plotting helpers.

## Likely Archive Or Diagnostic Scripts

- Older dated comparison and PMB scripts.
- `create_precip_products_of_antarctica_basins.py`
- `create_gpm_precip_products_of_antarctica_basins.py`
- `create_uahipa_monthly_2013_2020.py`
- `Diagnosing_trend_issues.py`
- `cf_diagnostic_apply_test.py`
