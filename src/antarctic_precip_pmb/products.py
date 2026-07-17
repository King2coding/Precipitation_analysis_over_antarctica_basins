"""Precipitation product normalization helpers."""

from __future__ import annotations

import pandas as pd


def normalize_month_start_index(values) -> pd.DatetimeIndex:
    """Normalize time-like values to month-start timestamps."""
    return pd.to_datetime(values).to_period("M").to_timestamp()


def monthly_rate_to_accumulation_mm(rate, time_values, unit: str):
    """Convert common product rates to monthly accumulation in mm/month."""
    unit_norm = unit.lower().replace(" ", "")
    days = pd.to_datetime(time_values).days_in_month
    if unit_norm in {"mm/day", "mm/d", "mmday-1"}:
        return rate * days
    if unit_norm in {"mm/hour", "mm/hr", "mmh-1"}:
        return rate * 24 * days
    if unit_norm in {"m/month", "m"}:
        return rate * 1000
    if unit_norm in {"mm/month", "mm"}:
        return rate
    raise ValueError(f"Unsupported precipitation unit: {unit}")
