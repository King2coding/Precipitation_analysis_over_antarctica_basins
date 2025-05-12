#%%
# packages
import warnings
warnings.filterwarnings("ignore")

import gc
import os

import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt

import subprocess

import xarray as xr

from rasterio.warp import Resampling
from pyproj import CRS

from osgeo import gdal, osr

import h5py

import cartopy.crs as ccrs
import cartopy.crs as ccrs

from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.cm import ScalarMappable

import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm


#%%
# floating variables
misc_out = r'/ra1/pubdat/AVHRR_CloudSat_proj/miscelaneous_outs'

crs = "+proj=longlat +datum=WGS84 +no_defs"  
crs_format = 'proj4'

crs_stereo = "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84"
#%%
# fucntions
# function to read and process ERA5 precip file
def process_era5_file_to_basin(era5_xr_data, er_tme, basins, fle_svnme):

        er_time = pd.to_datetime(er_tme).strftime('%Y%m%d')

        er_precip_dat = era5_xr_data.sel(time=er_tme).sel(lat=slice(-55, -90))

        yshp, xshp = er_precip_dat.shape

        minx = er_precip_dat['lon'].min().item()
        maxy = er_precip_dat['lat'].max().item()
        px_sz = round(er_precip_dat['lon'].diff('lon').mean().item(), 2)

        dest_flnme = os.path.join(misc_out, os.path.basename(fle_svnme).replace('.nc', '_sh.nc'))

        gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, er_precip_dat.data)

        output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.nc', '_sh_stere.nc'))

        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

        subprocess.run(gdalwarp_command, shell=True)

        # Read the stereographic projection file
        er_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

        os.remove(dest_flnme)
        os.remove(output_file_stereo)

        # Clip the data to the bounds of the basin dataset
        er_xrr_clip = er_xrr_sh_stereo.sel(
            x=slice(-3333250, 3333250),
            y=slice(-3333250, 3333250)
        ).squeeze()

        ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

        # Explicitly set the CRS before reprojecting
        er_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)

        er_xrr_clip_res = er_xrr_clip.rio.reproject(
            er_xrr_clip.rio.crs,
            shape=basins['zwally'].shape,  # set the shape as the basin data shape
            resampling=Resampling.nearest,
            transform=basins['zwally'].rio.transform()
        )

        # Add the time coordinate to the reprojected DataArray
        er_xrr_clip_res_arr = er_xrr_clip_res['Band1'].values
        er_xrr_clip_res_arr = np.where(basins['zwally'].values > 0, er_xrr_clip_res_arr, np.nan)
        er_xrr_clip_res = xr.DataArray(
            np.expand_dims(er_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
            dims=['time', 'y', 'x'],
            coords={'time': [np.datetime64(pd.to_datetime(er_time, format='%Y%m%d'), 'D')], 
                    'y': er_xrr_clip_res.coords['y'], 
                    'x': er_xrr_clip_res.coords['x']},
            name='precipitation'  # Change the variable name to 'precipitation'
        )

        return er_xrr_clip_res
#----------------------------------------------------------------------------
# function to read and process AIRS precip file
def process_airs_file_to_basin(airs_xrr_data, ai_tme, basins, fle_svnme):
        
        airs_time = pd.to_datetime(ai_tme).strftime('%Y%m%d')

        airs_precip_dat = airs_xrr_data.sel(time=ai_tme).sel(lat=slice(-55, -90))


        yshp, xshp = airs_precip_dat.shape

        minx = airs_precip_dat['lon'].min().item()
        maxy = airs_precip_dat['lat'].max().item()
        px_sz = round(airs_precip_dat['lon'].diff('lon').mean().item(), 2)

        dest_flnme = os.path.join(misc_out, os.path.basename(fle_svnme).replace('.nc', '_sh.nc'))

        gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, airs_precip_dat.data)

        output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.nc', '_sh_stere.nc'))

        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

        subprocess.run(gdalwarp_command, shell=True)

        # Read the stereographic projection file
        ai_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

        os.remove(dest_flnme)
        os.remove(output_file_stereo)

        # Clip the data to the bounds of the basin dataset
        ai_xrr_clip = ai_xrr_sh_stereo.sel(
            x=slice(-3333250, 3333250),
            y=slice(-3333250, 3333250)
        ).squeeze()

        ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

        # Explicitly set the CRS before reprojecting
        ai_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)
        ai_xrr_clip_res = ai_xrr_clip.rio.reproject(
            ai_xrr_clip.rio.crs,
            shape=basins['zwally'].shape,  # set the shape as the basin data shape
            resampling=Resampling.nearest,
            transform=basins['zwally'].rio.transform()
        )

        # Add the time coordinate to the reprojected DataArray
        ai_xrr_clip_res_arr = ai_xrr_clip_res['Band1'].values
        ai_xrr_clip_res_arr = np.where(basins['zwally'].values > 0, ai_xrr_clip_res_arr, np.nan)
        ai_xrr_clip_res = xr.DataArray(
            np.expand_dims(ai_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
            dims=['time', 'y', 'x'],
            coords={'time': [np.datetime64(pd.to_datetime(airs_time, format='%Y%m%d'), 'D')], 'y': ai_xrr_clip_res.coords['y'], 'x': ai_xrr_clip_res.coords['x']},
            name='precipitation'  # Change the variable name to 'precipitation'
        )

        return ai_xrr_clip_res
#----------------------------------------------------------------------------
def process_ssmis_file_to_basin(ss, basins):

        ss_time = os.path.basename(ss).split('.')[4].split('-')[0]

        ssmi_precip = xr.open_dataarray(ss)
        ssmi_precip = ssmi_precip * 24  # Multiply the "precipitation" data by 24

        ss_xrr_sh = ssmi_precip.sel(lat=slice(-55, -90)).squeeze()
        yshp, xshp = ss_xrr_sh.shape

        minx = ss_xrr_sh['lon'].min().item()
        maxy = ss_xrr_sh['lat'].max().item()
        px_sz = round(ss_xrr_sh['lon'].diff('lon').mean().item(), 2)

        dest_flnme = os.path.join(misc_out, os.path.basename(ss).replace('.nc', '_sh.nc'))

        gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, ss_xrr_sh.data)

        output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.nc', '_sh_stere.nc'))

        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

        subprocess.run(gdalwarp_command, shell=True)

        # Read the stereographic projection file
        ss_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

        os.remove(dest_flnme)
        os.remove(output_file_stereo)

        # Clip the data to the bounds of the basin dataset
        ss_xrr_clip = ss_xrr_sh_stereo.sel(
            x=slice(-3333250, 3333250),
            y=slice(-3333250, 3333250)
        ).squeeze()

        ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

        # Explicitly set the CRS before reprojecting
        ss_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)
        ss_xrr_clip_res = ss_xrr_clip.rio.reproject(
            ss_xrr_clip.rio.crs,
            shape=basins['zwally'].shape,  # set the shape as the basin data shape
            resampling=Resampling.nearest,
            transform=basins['zwally'].rio.transform()
        )

        # Add the time coordinate to the reprojected DataArray
        ss_xrr_clip_res_arr = ss_xrr_clip_res['Band1'].values
        ss_xrr_clip_res_arr = np.where(basins['zwally'].values > 0, ss_xrr_clip_res_arr, np.nan)
        ss_xrr_clip_res = xr.DataArray(
            np.expand_dims(ss_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
            dims=['time', 'y', 'x'],
            coords={'time': [ss_time], 'y': ss_xrr_clip_res.coords['y'], 'x': ss_xrr_clip_res.coords['x']},
            name='precipitation'  # Change the variable name to 'precipitation'
        )

        return ss_xrr_clip_res
#----------------------------------------------------------------------------
def process_avhrr_file_to_basin(file,yr,basins):
    av_x, av_y = return_sim_cords(file)
    av_arr, av_time = read_tif_sim_file(file,yr)

    av_arr = np.expand_dims(av_arr, axis=0) if av_arr.ndim == 2 else av_arr  # Ensure img_arr has 3 dimensions
    av_xrr = xr.DataArray(av_arr, dims=['time', 'lat', 'lon'], coords={'time': [av_time], 'lat': av_y, 'lon': av_x})
    av_xrr_sh = av_xrr.sel(lat=slice(-55, -90)).squeeze()
    yshp, xshp = av_xrr_sh.shape

    minx = av_xrr_sh['lon'].min().item()
    maxy = av_xrr_sh['lat'].max().item()
    px_sz = round(av_xrr_sh['lon'].diff('lon').mean().item(), 2)

    dest_flnme = os.path.join(misc_out, os.path.basename(file).replace('.tif', '_sh.tif'))

    gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, av_xrr_sh.data)

    output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.tif', '_sh_stere.tif'))

    gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

    subprocess.run(gdalwarp_command, shell=True)

    # Read the stereographic projection file
    av_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

    os.remove(dest_flnme)
    os.remove(output_file_stereo)

    # Clip the data to the bounds of the basin dataset
    av_xrr_clip = av_xrr_sh_stereo.sel(
        x=slice(-3333250, 3333250),
        y=slice(3333250, -3333250)
    ).squeeze()

    av_xrr_clip_res = av_xrr_clip.rio.reproject(
        av_xrr_clip.rio.crs,
        shape=basins['zwally'].shape,  # set the shape as the basin data shape
        resampling=Resampling.nearest,
        transform=basins['zwally'].rio.transform()
    )

    # Add the time coordinate to the reprojected DataArray
    av_xrr_clip_res_arr = av_xrr_clip_res['band_data'].values
    av_xrr_clip_res_arr = np.where(basins['zwally'].values > 0, av_xrr_clip_res_arr, np.nan)
    av_xrr_clip_res = xr.DataArray(
        np.expand_dims(av_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
        dims=['time', 'y', 'x'],
        coords={'time': [av_time], 'y': av_xrr_clip_res.coords['y'], 'x': av_xrr_clip_res.coords['x']},
        name='precipitation'  # Change the variable name to 'precipitation'
    )
    # av_xrr_clip_res = av_xrr_clip_res.rename({'band_data': 'precipitation'})

    return av_xrr_clip_res
#----------------------------------------------------------------------------
# function to process IMERG file to basin
def process_imerg_file_to_basin(im, misc_out, basins):
    img_x, img_y = return_imerg_cords(im)
    img_arr, img_time = read_nc_imger_file(im, 'imerg_fn')

    img_arr = np.expand_dims(img_arr, axis=0) if img_arr.ndim == 2 else img_arr  # Ensure img_arr has 3 dimensions
    img_xrr = xr.DataArray(img_arr, dims=['time', 'lat', 'lon'], coords={'time': [img_time], 'lat': img_y, 'lon': img_x})

    img_xrr_sh = img_xrr.sel(lat=slice(-55, -90)).squeeze()
    yshp, xshp = img_xrr_sh.shape

    minx = img_xrr_sh['lon'].min().item()
    maxy = img_xrr_sh['lat'].max().item()

    px_sz = round(img_xrr_sh['lon'].diff('lon').mean().item(), 2)

    dest_flnme = os.path.join(misc_out, os.path.basename(im).replace('.nc4', '.tif'))

    gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, img_xrr_sh.data)

    output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('.tif', '_sh_stere.tif'))
    gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'
    subprocess.run(gdalwarp_command, shell=True)

    # Read the stereographic projection file
    img_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

    os.remove(dest_flnme)
    os.remove(output_file_stereo)

    # Clip the data to the bounds of the second dataset
    img_xrr_clip = img_xrr_sh_stereo.sel(
        x=slice(-3333250, 3333250),
        y=slice(3333250, -3333250)
    ).squeeze()

    img_xrr_clip_res = img_xrr_clip.rio.reproject(
        img_xrr_clip.rio.crs,
        shape=basins['zwally'].shape,  # set the shape as the autosnow data shape (1800, 3600)
        resampling=Resampling.nearest,
        transform=basins['zwally'].rio.transform()
    )
    # Add the time coordinate to the reprojected DataArray
    img_xrr_clip_res_arr = img_xrr_clip_res['band_data'].values
    img_xrr_clip_res_arr = np.where(basins['zwally'].values > 0, img_xrr_clip_res_arr, np.nan)
    img_xrr_clip_res = xr.DataArray(
        np.expand_dims(img_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
        dims=['time', 'y', 'x'],
        coords={'time': [img_time], 'y': img_xrr_clip_res.coords['y'], 'x': img_xrr_clip_res.coords['x']},
        name='precipitation'  # Change the variable name to 'precipitation'
    )       

    return img_xrr_clip_res

#----------------------------------------------------------------------------
# function to save 2d arrays to disk using gdal
def gdal_based_save_array_to_disk(dstflnme, xpx, ypx, px_sz, minx, maxy, crs, crs_format, arrayTosave):
        
        """
        Save an array to disk using GDAL.

        Parameters:
        - dstflnme: Destination filename including the file path.
        - xpx: Number of pixels in x (width).
        - ypx: Number of pixels in y (height).
        - px_sz: Size of the pixel (e.g., 0.5 degrees).
        - minx: Minimum x (longitude) value.
        - maxy: Maximum y (latitude) value (top-left corner).
        - crs: Coordinate reference system in proj4 or WKT format.
        - crs_format: 'proj4' or 'WKT', indicating the format of the CRS.
        - arrayTosave: The array to save to disk.
        """        
        
        # Calculate the bottom latitude and clamp it to -90.0 if necessary
        bottom_latitude = maxy - (ypx * px_sz)
        if bottom_latitude < -90.0:
            bottom_latitude = -90.0
            # Adjust ypx based on the clamped latitude
            ypx = int((maxy - bottom_latitude) / px_sz)
            # Slice the array to match the new dimensions
            arrayTosave = arrayTosave[:ypx, :]

        srs = osr.SpatialReference()
        if crs_format == 'proj4':
            srs.ImportFromProj4(crs)
            srs = srs.ExportToProj4()
        elif crs_format == 'WKT':
            srs.ImportFromWkt(crs)
            srs = srs.ExportToWkt()

        driver = gdal.GetDriverByName('GTiff')

        # Print dimensions to debug
        # print(f"Array Shape After Clamping: {arrayTosave.shape}")
        # print(f"Raster Size: (ypx={ypx}, xpx={xpx})")

        dataset = driver.Create(dstflnme, xpx, ypx, 1, gdal.GDT_Float64)

        dataset.SetGeoTransform((
            minx,    
            px_sz,   
            0.0,     
            maxy,    
            0.0,     
            -px_sz   
        ))  

        dataset.SetProjection(srs)
        dataset.GetRasterBand(1).WriteArray(arrayTosave)
        dataset.FlushCache()
        del(dataset)
#----------------------------------------------------------------------------
# function to reproject data using gdal
def gdal_based_stereo_reproj(input_file,hemisphere, y_min,y_max):     

    misc_out = r'/ra1/pubdat/AVHRR_CloudSat_proj/miscelaneous_outs'

    input_xr_dat = xr.open_dataset(input_file)     

    # Apply conditional selection based on the hemisphere and latitude bounds
    if 'x' in input_xr_dat.dims or 'y' in input_xr_dat.dims:
        input_xr_dat = input_xr_dat.rename({'x': 'lon', 'y': 'lat'})

    xr_dat = input_xr_dat.sel(lat=slice(y_max,y_min))
    # xr_dat = input_xr_dat.where((input_xr_dat.y >= y_min) & (input_xr_dat.y <= y_max), drop=True)

    if 'band' in xr_dat.dims and xr_dat.sizes['band'] == 1:
        xr_dat = xr_dat.squeeze('band')

    # xr_dat = xr_dat.transpose('y', 'x')
    array2sve = xr_dat.data

    x_dim = 'x' if 'x' in xr_dat.dims else 'lon'
    y_dim = 'y' if 'y' in xr_dat.dims else 'lat'

    ypx, xpx = xr_dat.shape  
    px_sz = xr_dat[x_dim].diff(x_dim).mean().item()  
    minx = xr_dat[x_dim].min().item()  
    maxy = xr_dat[y_dim].max().item()  
    
    crs = "+proj=longlat +datum=WGS84 +no_defs"  
    crs_format = 'proj4'  
    
    output_file_wgs = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', f'_{hemisphere.lower()}.tif'))
    
    gdal_based_save_array_to_disk(output_file_wgs, xpx, ypx, px_sz, minx, maxy, crs, crs_format, array2sve)

    # Print latitude bounds for debugging
    miny = xr_dat[y_dim].min().item()
    maxy = xr_dat[y_dim].max().item()
    print(f"{hemisphere} Data Latitude Bounds: miny={miny}, maxy={maxy}")

    # if hemisphere == 'NH':
    #     output_file_stereo = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', '_nh_stere.tif'))
    #     gdalwarp_command = f'gdalwarp -t_srs "EPSG:3413" -r near {output_file_wgs} {output_file_stereo}'
    # elif hemisphere == 'SH':
    #     output_file_stereo = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', '_sh_stere.tif'))
    #     gdalwarp_command = f'gdalwarp -t_srs "EPSG:3976" -r near {output_file_wgs} {output_file_stereo}'

    if hemisphere == 'NH':
        output_file_stereo = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', '_nh_stere.tif'))
        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=0 +datum=WGS84" -r near {output_file_wgs} {output_file_stereo}'
    elif hemisphere == 'SH':
        output_file_stereo = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', '_sh_stere.tif'))
        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +datum=WGS84" -r near {output_file_wgs} {output_file_stereo}'


    # print(f"Reprojecting {hemisphere} data...")
    subprocess.run(gdalwarp_command, shell=True)

    xr_dt_stereo = xr.open_dataarray(output_file_stereo).squeeze()

    os.remove(output_file_wgs)

    os.remove(output_file_stereo)   

    return xr_dt_stereo

#----------------------------------------------------------------------------
# function to read IMGER file
def read_nc_imger_file(file_path, product):
    imerg_precip_data = xr.open_dataset(file_path)
    if product == 'imerg_fn':
        precip_aray = imerg_precip_data.precipitation.data 
    elif product == 'imerg_mw':
        precip_aray = imerg_precip_data.MWprecipitation.data 
    precip_aray = np.flip(precip_aray[0,:,:].transpose(), axis=0)
    imerg_time = imerg_precip_data.attrs['BeginDate']
    imerg_precip_data.close()
    
    # Convert time to pandas datetime
    imerg_time_index = pd.to_datetime(imerg_time,format='%Y-%m-%d')

    del(imerg_precip_data,imerg_time)
    
    return precip_aray, imerg_time_index

#----------------------------------------------------------------------------
# function to read IMGER file and return coordinates
def return_imerg_cords(file):
    file_dat = xr.open_dataset(file)
    lon = file_dat.coords['lon'].values
    lat = np.flip(file_dat.coords['lat']).values

    del(file_dat)

    return lon, lat

#----------------------------------------------------------------------------
# function to read SSMIS file
def read_tif_sim_file(file,yr):

    sim_precip_array = xr.open_dataarray(file).data[0] 

    sim_time = [x for x in os.path.split(file)[1].split('_') if x.startswith(yr)][0]


    sim_time_index = pd.to_datetime(sim_time,format='%Y%m%d')

    return sim_precip_array, sim_time_index

#----------------------------------------------------------------------------
# function to read SSMIS file and return coordinates
def return_sim_cords(file):
    file_dat = xr.open_dataarray(file)
    lon = file_dat.coords['x'].values
    lat = file_dat.coords['y'].values

    file_dat.close()
    return lon, lat

#----------------------------------------------------------------------------
# function to run batched processing
def run_batched_processing(batches, basins):
    """
    Function to process precipitation data in batches and calculate mean precipitation for each basin.

    Parameters:
    - batches: List of lists, where each inner list contains file paths to precipitation data for a batch.
    - basins: xarray DataArray representing the basin regions, with unique IDs for each basin.

    Returns:
    - batch_results: List of xarray DataArrays containing mean precipitation for each basin in each batch.
    """
     
    # Initialize a list to store the results from each batch
    batch_results = []

    # Get the total number of batches to process
    total_batches = len(batches)
    print(f"Total number of batches to process: {total_batches}")

    # Loop through each batch
    for idx, batch in enumerate(batches, start=1):
        # Display progress for the current batch
        print(f"Processing batch {idx}/{total_batches} ({(idx / total_batches) * 100:.2f}%)...")

        # Concatenate all precipitation data in the current batch along the 'time' dimension
        precip_bsn_xrr = xr.concat([xr.open_dataarray(x) for x in batch], dim='time')

        # Create an empty DataArray to store mean precipitation mapped to basins
        precip_xrr_basin_mapped = xr.full_like(basins, np.nan, dtype=float)

        # Loop through each basin ID (Zwally basins are numbered from 1 to 27)
        for basin_id in range(1, 28):
            # Create a mask for the current basin
            basin_mask = basins == basin_id

            # Mask the precipitation data for the current basin
            basin_precip = precip_bsn_xrr.where(basin_mask.data)

            # Calculate the mean precipitation for the current basin across time and spatial dimensions
            basin_mean_precip = basin_precip.mean(dim=['time', 'x', 'y'], skipna=True)

            # Map the calculated mean precipitation back to the basin region
            precip_xrr_basin_mapped = precip_xrr_basin_mapped.where(~basin_mask, basin_mean_precip)

        # Clean up variables to free memory
        del(precip_bsn_xrr, basin_precip, basin_mean_precip, basin_mask, basin_id)

        # Append the result for this batch to the results list
        batch_results.append(precip_xrr_basin_mapped)

    # Indicate that all batches have been processed
    print("All batches have been processed.")

    return batch_results

#----------------------------------------------------------------------------
# plot

def compare_mean_precp_plot(arr_lst_mean, vmin=0, vmax=300, cbar_tcks=None):
    """
    Plots a multi-row grid of mean precipitation for different products over 27 basins.

    Parameters:
    arr_lst_mean (list of tuples): List of tuples with product names and their mean precipitation data.
    vmin (float): Minimum value for colorbar.
    vmax (float): Maximum value for colorbar.
    """
    proj = ccrs.SouthPolarStereo()

    # Use the 'jet' colormap
    cmap = plt.cm.jet
    levels = np.linspace(vmin, vmax, 28)  # 27 basins + 1 for boundaries
    norm = BoundaryNorm(levels, cmap.N)

    # Determine the number of rows and columns for the grid
    n_products = len(arr_lst_mean)
    ncols = 3  # Number of columns
    nrows = (n_products + ncols - 1) // ncols  # Calculate rows needed

    # Create a GridSpec with precise control over spacing
    fig = plt.figure(figsize=(28, 10 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.025, hspace=0.15)

    # Loop through each dataset to create the plots
    axes = []
    for i, (product_name, data) in enumerate(arr_lst_mean):
        ax = fig.add_subplot(gs[i], projection=proj)
        ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
        ax.coastlines(lw=0.25, resolution="110m", zorder=2)

        # Plot the data
        data['zwally'].plot(
            ax=ax,
            transform=ccrs.SouthPolarStereo(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False
        )

        ax.add_feature(cfeature.OCEAN, zorder=1, edgecolor=None, lw=0, color="silver", alpha=0.5)
        ax.set_title(product_name, fontsize=20)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                          linestyle='--', color='k', linewidth=0.75)
        gl.xlocator = MaxNLocator(nbins=5)
        gl.ylocator = MaxNLocator(nbins=5)
        gl.xlabel_style = {'size': 20, 'color': 'k'}
        gl.ylabel_style = {'size': 20, 'color': 'k'}

        # Only show specific labels
        gl.top_labels = False
        gl.bottom_labels = i >= n_products - ncols  # Bottom labels only for the last row
        gl.left_labels = i % ncols == 0  # Left labels for the first column
        gl.right_labels = i % ncols == ncols - 1  # Right labels for the last column

        axes.append(ax)

    # Create a colorbar at the bottom spanning all subplots
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,  # Attach the colorbar to all axes for better placement
        orientation="horizontal",
        fraction=0.04,  # Fraction of the original axes height
        pad=0.15,  # Distance from the bottom of the subplots
        extend="max"
    )
    cb.set_ticks(cbar_tcks)  # Set integer ticks
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Precipitation [mm/year]", fontsize=20)

    # Show the plot
    plt.tight_layout()
