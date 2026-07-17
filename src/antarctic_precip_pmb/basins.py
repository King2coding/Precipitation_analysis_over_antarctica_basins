"""Basin definitions, masks, and region utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .constants import AIS_BASINS, BASIN_ID_TO_NAME, REGION_BASINS


def validate_basin_mapping(mapping: Mapping[int, str] = BASIN_ID_TO_NAME) -> None:
    """Validate that the IMBIE/Rignot basin mapping is complete and unique."""
    expected = set(range(2, 20))
    actual = set(mapping)
    if actual != expected:
        raise ValueError(f"Expected basin IDs 2..19, found {sorted(actual)}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Basin names must be unique.")


def validate_region_definitions(region_basins: Mapping[str, Sequence[int]] = REGION_BASINS) -> None:
    """Check that WAIS and EAIS partition the Antarctic basin set."""
    ais = set(region_basins["Antarctica"])
    wais = set(region_basins["West Antarctica"])
    eais = set(region_basins["East Antarctica"])
    if ais != set(AIS_BASINS):
        raise ValueError("Antarctica region does not match configured AIS basins.")
    if wais & eais:
        raise ValueError(f"WAIS and EAIS overlap: {sorted(wais & eais)}")
    if wais | eais != ais:
        raise ValueError("WAIS and EAIS do not cover all AIS basins.")


def mask_valid_basin_ids(values, valid_ids: Sequence[int] = AIS_BASINS):
    """Mask a NumPy/xarray basin-id grid to valid study basin IDs."""
    valid = np.isin(values, np.asarray(valid_ids))
    try:
        return values.where(valid)
    except AttributeError:
        return np.where(valid, values, np.nan)


def load_basin_grid(path, crs: str | None = None):
    """Load a basin raster as an xarray DataArray.

    Heavy geospatial dependencies are imported inside the function so package
    imports stay lightweight for tests and documentation builds.
    """
    import rioxarray  # noqa: F401
    import xarray as xr
    from rasterio.crs import CRS

    basins = xr.open_dataarray(path)
    basins = basins.where((basins > 1) & basins.notnull())
    if crs and not basins.rio.crs:
        basins = basins.rio.write_crs(CRS.from_proj4(crs), inplace=False)
    return basins
