"""Mass-budget precipitation calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


PMB_COMPONENTS = ("discharge", "basal_melt", "deltaS", "sublimation")


def compute_pmb_gt_month(
    discharge: pd.DataFrame,
    basal_melt: pd.DataFrame,
    delta_s: pd.DataFrame,
    sublimation: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly PMB in Gt/month from aligned basin component tables."""
    aligned = [df.sort_index() for df in (discharge, basal_melt, delta_s, sublimation)]
    discharge_a, basal_melt_a, delta_s_a, sublimation_a = aligned
    common_index = discharge_a.index
    common_columns = discharge_a.columns
    for df in aligned[1:]:
        common_index = common_index.intersection(df.index)
        common_columns = common_columns.intersection(df.columns)
    if common_index.empty or common_columns.empty:
        raise ValueError("PMB components have no common dates or basins.")
    return (
        discharge_a.loc[common_index, common_columns]
        + basal_melt_a.loc[common_index, common_columns]
        + delta_s_a.loc[common_index, common_columns]
        + sublimation_a.loc[common_index, common_columns]
    )


def propagate_pmb_uncertainty_gt_month(
    sigma_delta_s: pd.DataFrame,
    sigma_discharge: pd.DataFrame | None = None,
    sigma_basal_melt: pd.DataFrame | None = None,
    sigma_sublimation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Propagate independent 1-sigma component uncertainty in quadrature."""
    variance = sigma_delta_s.astype(float) ** 2
    for optional in (sigma_discharge, sigma_basal_melt, sigma_sublimation):
        if optional is not None:
            optional = optional.reindex(index=variance.index, columns=variance.columns)
            variance = variance + optional.astype(float) ** 2
    return np.sqrt(variance)
