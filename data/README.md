# Required Input Data

Large scientific inputs are not stored in this repository. Configure local paths
in `config/paths.yaml`, copied from `config/example_paths.yaml`.

Expected inputs include:

- IMBIE/Rignot basin raster, especially `bedmap3_basins_0.1deg.tif`.
- GRACE/altimetry basin workbook: `DataCombo_RignotBasins_update.xlsx`.
- Discharge and basal melt workbook: `antarctic_discharge_2013-2022_imbie.xlsx`.
- RACMO2.4p1 monthly sublimation file containing `subltot`.
- ERA5 monthly total precipitation.
- GPCP v3.3 monthly precipitation files.
- GPM passive microwave constellation files for V07 and V08.
- UA-HIPA monthly precipitation product.
- CloudSat Antarctic snowfall climatology files for the 2007-2010 comparison.

Do not commit NetCDF, HDF, GeoTIFF, Excel, pickle, CSV, or generated image
products to Git.
