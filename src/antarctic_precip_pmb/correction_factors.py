"""PMB-based precipitation-product correction factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def scalar_correction_factors(
    table: pd.DataFrame,
    *,
    ref_product: str,
    target_product: str,
    group_cols: tuple[str, ...] = ("region", "season"),
    value_col: str = "precipitation",
) -> pd.DataFrame:
    """Compute grouped scalar factors as reference / target."""
    required = set(group_cols) | {"product", value_col}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns for correction factors: {sorted(missing)}")
    pivot = table.pivot_table(index=list(group_cols), columns="product", values=value_col, aggfunc="mean")
    if ref_product not in pivot or target_product not in pivot:
        raise ValueError("Reference or target product not found in table.")
    factor = pivot[ref_product] / pivot[target_product]
    factor = factor.replace([np.inf, -np.inf], np.nan).rename("correction_factor")
    return factor.reset_index()


def apply_scalar_correction(values, factor):
    """Apply multiplicative correction factors to product values."""
    return values * factor
