"""Unit conversions used by the mass-budget workflow."""

from __future__ import annotations

import numpy as np


def gt_to_mm_water_equivalent(gt, area_m2):
    """Convert basin mass in Gt to mm water equivalent over basin area."""
    return np.asarray(gt, dtype=float) * 1.0e12 / np.asarray(area_m2, dtype=float)


def mm_water_equivalent_to_gt(mm, area_m2):
    """Convert mm water equivalent over basin area to Gt."""
    return np.asarray(mm, dtype=float) * np.asarray(area_m2, dtype=float) / 1.0e12
