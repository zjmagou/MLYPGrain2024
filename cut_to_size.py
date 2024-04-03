#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:27:54 2023

@author: maeko
"""

import pandas as pd
import geopandas as gpd
import rioxarray
import os
import xarray as xr
import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

print('Done Importing.')


def rectangle_clip(gdf, xrds):
    # Find the bounding box of the shapefile
    bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]
    xrds.rio.write_crs("EPSG:4326", inplace=True)
    return xrds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])


def single_prov(provGDF, experiment, model, tas, pr):
    ## Temperature
    try:
        hw = xr.open_dataset(f'waves/{experiment}_{model}_heatwave.nc')
        hwave = hw.where(hw['heatwave_frequency'] >= 0)

        hwave.rio.set_spatial_dims('lon', 'lat')
        hwave.rio.write_crs('EPSG:4326', inplace=True)

        cw = xr.open_dataset(f'waves/{experiment}_{model}_coldwave.nc')
        cwave = cw.where(cw['coldwave_frequency'] >= 0)

        cwave.rio.set_spatial_dims('lon', 'lat')
        cwave.rio.write_crs('EPSG:4326', inplace=True)
    except:
        print(f'No wave data for {experiment} {model}, skipping.')
        return

    c3 = rectangle_clip(provGDF, hwave)
    c4 = rectangle_clip(provGDF, cwave)

    c1 = rectangle_clip(provGDF, tas)
    c2 = rectangle_clip(provGDF, pr)

    return [c1, c2, c3, c4]


def main(experiment, model, ssp=True, obs=True):
    def get_monthly(var):
        if ssp:
            if var == 'tas':
                folder_path = 'CMIP6-SSP-Temp'
            if var == 'pr':
                folder_path = 'CMIP6-SSP-PR'

        else:
            if var == 'tas':
                folder_path = 'CMIP6-Historical-Temp'
            if var == 'pr':
                folder_path = 'CMIP6-Historical-PR'

        def groupby_month(da):
            year_month_idx = pd.MultiIndex.from_arrays([da['time.year'].values, da['time.month'].values])
            da.coords['year_month'] = ('time', year_month_idx)
            if var == 'tas':
                return da.groupby('year_month').mean('time')
            if var == 'pr':
                return da.groupby('year_month').sum('time')

        # Use glob.glob to find files matching the pattern
        li = glob.glob(f'NASA-CMIP/{folder_path}/{model}/{var}_day_{model}_{experiment}_*.nc')

        try:
            temp = xr.open_mfdataset(li, combine="nested", concat_dim="time")

        except OSError:
            warnings.warn(f'No {var} data for {model}, {experiment}, skipping.')
            return None

        else:

            gr = groupby_month(temp)
            datetime_values = pd.to_datetime(['{}-{}'.format(year, month) for year, month in gr['year_month'].values])

            # Create a new DataArray with the datetime values
            new_year_month = xr.DataArray(datetime_values, dims=["year_month"], name="new_year_month")

            # Merge this new DataArray into your dataset
            # This step should replace the existing 'year_month' DataArray
            gr = gr.drop_vars('year_month').assign_coords({'year_month': new_year_month})
            gr.rio.write_crs('EPSG:4326', inplace=True)

            return gr

    def process_prov_data(prov_data):
        # Unpack the returned datasets
        tas, pr, heatwave, coldwave = prov_data

        # Retrieve the ADM1_PCODE value
        adm1_pcode = shpli[i]["ADM1_PCODE"].values[0]

        print(f'Saving province {adm1_pcode}.')
        # Save the datasets to netcdf files
        tas.to_netcdf(f'tas.month.{model}.{experiment}.{adm1_pcode}.nc')
        pr.to_netcdf(f'pr.month.{model}.{experiment}.{adm1_pcode}.nc')
        heatwave.to_netcdf(f'heatwave.year.{model}.{experiment}.{adm1_pcode}.nc')
        coldwave.to_netcdf(f'coldwave.year.{model}.{experiment}.{adm1_pcode}.nc')

    tas = get_monthly('tas')
    pr = get_monthly('pr')

    # shp
    gdf = gpd.read_file("SHP/chn_adm_ocha_2020_shp", layer=2)

    JS = gdf[gdf['ADM1_PCODE'] == 'CN032']
    HB = gdf[gdf['ADM1_PCODE'] == 'CN042']
    HN = gdf[gdf['ADM1_PCODE'] == 'CN043']
    ZJ = gdf[gdf['ADM1_PCODE'] == 'CN033']
    JX = gdf[gdf['ADM1_PCODE'] == 'CN036']
    AH = gdf[gdf['ADM1_PCODE'] == 'CN034']
    SH = gdf[gdf['ADM1_PCODE'] == 'CN031']

    shpli = [JS, HB, HN, ZJ, JX, AH]

    partial_single_prov = partial(single_prov, experiment=experiment, model=model, tas=tas, pr=pr)

    with ProcessPoolExecutor(max_workers=8) as executor:
        # Map single_prov to each element in shpli
        results = executor.map(partial_single_prov, shpli)

        # Process each result
        for i, result in enumerate(results):
            print(result)
            if isinstance(result, list):
                if not isinstance(result[0], type(None)):
                    process_prov_data(result)


if __name__ == "__main__":

    import warnings

    warnings.simplefilter('ignore')

    experiments = ['ssp585']  #'ssp126','ssp245', 'ssp370',
    models = [f.name for f in os.scandir("NASA-CMIP/CMIP6-SSP-Temp") if f.is_dir()][22:]

    for exp in experiments:
        for mo in models:
            main(experiment=exp, model=mo, obs=False)
            print(f'Finished for model {mo}.')
