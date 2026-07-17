"""GRACE/altimetry storage anomaly gap filling and deltaS handling."""

from __future__ import annotations

import pandas as pd


def decimal_year_to_month_start(decimal_year: float, mode: str = "nearest") -> pd.Timestamp:
    """Convert a decimal year to a month-start timestamp."""
    year = int(decimal_year)
    fraction = float(decimal_year) - year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    date = start + (end - start) * fraction
    if mode == "floor":
        return date.to_period("M").to_timestamp()
    if mode == "nearest":
        return (date + pd.Timedelta(days=15)).to_period("M").to_timestamp()
    raise ValueError("mode must be 'nearest' or 'floor'")


def monthly_storage_table(
    df: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    time_col: str = "Time",
    mode: str = "nearest",
) -> pd.DataFrame:
    """Convert David/Rignot decimal-year storage anomaly data to a monthly table."""
    work = df.copy()
    work["date"] = work[time_col].apply(lambda x: decimal_year_to_month_start(x, mode=mode))
    work["Year"] = work["date"].dt.year
    work["Month"] = work["date"].dt.month
    basin_cols = [c for c in work.columns if c not in {time_col, "date", "Year", "Month"}]
    monthly = work.groupby(["Year", "Month"], as_index=False)[basin_cols].mean()
    monthly["date"] = pd.to_datetime(dict(year=monthly["Year"], month=monthly["Month"], day=1))
    monthly = monthly.set_index("date").sort_index()
    full_index = pd.date_range(start=start_date, end=end_date, freq="MS")
    out = monthly[basin_cols].reindex(full_index)
    out.index.name = "date"
    return out


def fill_storage_deseasonalized_linear(storage: pd.DataFrame) -> pd.DataFrame:
    """Fill monthly storage anomalies using deseasonalized linear interpolation."""
    filled = storage.copy()
    for basin in storage.columns:
        series = storage[basin]
        climatology = series.groupby(series.index.month).mean()
        deseason = series - series.index.month.map(climatology)
        deseason_interp = deseason.interpolate(method="linear", limit_area="inside")
        filled[basin] = deseason_interp + deseason_interp.index.month.map(climatology)
    return filled


def fill_uncertainty_linear(error: pd.DataFrame) -> pd.DataFrame:
    """Fill monthly 1-sigma uncertainty by direct linear interpolation."""
    return error.interpolate(method="linear", limit_area="inside")


def forward_delta_s(storage: pd.DataFrame) -> pd.DataFrame:
    """Compute deltaS_m = S_{m+1} - S_m and assign it to month m."""
    delta = storage.shift(-1) - storage
    delta.index.name = storage.index.name
    return delta.iloc[:-1]
