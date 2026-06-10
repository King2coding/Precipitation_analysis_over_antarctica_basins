"""Study constants that should remain explicit and reviewable."""

from __future__ import annotations

YEAR_START = 2013
YEAR_END = 2020
YEARS = tuple(range(YEAR_START, YEAR_END + 1))

SEASONS = ("DJF", "MAM", "JJA", "SON")

CRS_WGS84 = "+proj=longlat +datum=WGS84 +no_defs"
CRS_SH_STEREO = (
    "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 "
    "+lon_0=0 +datum=WGS84"
)

BASIN_ID_TO_NAME = {
    2: "A-Ap",
    3: "Ap-B",
    4: "B-C",
    5: "C-Cp",
    6: "Cp-D",
    7: "D-Dp",
    8: "Dp-E",
    9: "E-Ep",
    10: "Ep-F",
    11: "F-G",
    12: "G-H",
    13: "H-Hp",
    14: "Hp-I",
    15: "I-Ipp",
    16: "Ipp-J",
    17: "J-Jpp",
    18: "Jpp-K",
    19: "K-A",
}

BASIN_NAME_TO_ID = {name: basin_id for basin_id, name in BASIN_ID_TO_NAME.items()}

WAIS_BASINS = (10, 11, 12, 13, 14, 15, 16, 17)
EAIS_BASINS = (2, 3, 4, 5, 6, 7, 8, 9, 18, 19)
AIS_BASINS = WAIS_BASINS + EAIS_BASINS

REGION_BASINS = {
    "Antarctica": AIS_BASINS,
    "West Antarctica": WAIS_BASINS,
    "East Antarctica": EAIS_BASINS,
}

PMB_EQUATION = "P_MB = discharge + basal_melt + deltaS + sublimation"
DELTAS_CONVENTION = "forward"
DELTAS_DESCRIPTION = "deltaS_m = S_{m+1} - S_m, assigned to month m"
