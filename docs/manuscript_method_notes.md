# Manuscript Method Notes

The requested manuscript file, `Antarctic_Snowfall-PMB-method-V0.docx`, was not
available in the repository or under `/home/kkumah/Projects` during this phase.
These notes therefore summarize assumptions found in the study scripts and
should be reconciled against the manuscript before final publication.

## Script-Derived Assumptions

- Main analysis period: 2013-2020.
- Main basin set: IMBIE/Rignot basins with IDs 2-19.
- Region split:
  - West Antarctica: 10-17.
  - East Antarctica: 2-9, 18, 19.
- PMB equation:
  `P_MB = discharge + basal_melt + deltaS + sublimation`.
- GRACE/storage anomaly source is a basin-scale monthly storage anomaly, not
  deltaS directly.
- Active deltaS convention in the inspected mass-budget script is forward
  difference:
  `deltaS_m = S_{m+1} - S_m`, assigned to the starting month.
- RACMO `subltot` is converted once into positive sublimation loss before being
  added in the PMB equation.
- Formal PMB uncertainty currently emphasizes GRACE/altimetry deltaS
  uncertainty; discharge, basal melt, and sublimation fractional uncertainties
  are configurable but set to zero in the inspected script.

## Items To Confirm Against Manuscript

- Whether the final method uses forward or backward deltaS labeling.
- Whether GRACE deltaS climatology correction remains disabled.
- Final product list and product version labels.
- Whether CloudSat 2007-2010 is main text or supplemental.
- Whether UA-HIPA and GPM correction factors are final results or sensitivity
  analyses.
- Exact wording for basin definitions, units, and time-period support.
