"""Monthly, seasonal, annual, and regional aggregation helpers."""

from __future__ import annotations

import pandas as pd

from .constants import SEASONS


def month_to_season(month: int) -> str:
    """Return conventional meteorological season for a calendar month."""
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    raise ValueError(f"Invalid month: {month}")


def monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month climatology from a monthly dataframe."""
    return df.groupby(df.index.month).mean()


def annual_totals(df: pd.DataFrame, require_complete_years: bool = True) -> pd.DataFrame:
    """Annual totals from monthly values."""
    counts = df.groupby(df.index.year).count()
    totals = df.groupby(df.index.year).sum(min_count=1)
    if require_complete_years:
        complete_years = counts.index[(counts == 12).all(axis=1)]
        totals = totals.loc[complete_years]
    totals.index.name = "year"
    return totals


def seasonal_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean monthly value by conventional season."""
    labels = pd.Categorical([month_to_season(m) for m in df.index.month], categories=SEASONS, ordered=True)
    out = df.groupby(labels, observed=False).mean()
    out.index.name = "season"
    return out
