# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import warnings

ssp = False


def extract_year_from_filepath(filepath):
    # Assuming filenames are in the format: "someprefix_YEAR.nc"
    # Modify this function based on your actual filename format to extract the year
    if filepath.split('_')[-1].split('.')[0].startswith('v'):
        return int(filepath.split('_')[-2])
    else:
        return int(filepath.split('_')[-1].split('.')[0])


def get_monthly(model, experiment, var, years):
    """
    Reduce the dimensions of a daily dataset to monthly.

    :param model: String
    :param experiment: String
    :param var: 'tas', 'pr', 'heatwave' or 'coldwave'
    :param years: (Optional) tuple of length 2
    :return:
    """

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

    pattern = f'NASA-CMIP/{folder_path}/{model}/{var}_day_{model}_{experiment}_*.nc'

    # Use glob.glob to find files matching the pattern
    li = glob.glob(pattern)

    if years != 'All':
        if isinstance(years, tuple) and len(years) == 2:
            yyyy = [extract_year_from_filepath(fp) for fp in li]
            li = [li[i] for i in range(len(yyyy)) if years[0] <= yyyy[i] <= years[1]]

        else:
            raise ValueError('Input to :param years is bad.')

    if len(li) == 0:
        return None
    else:
        temp = xr.open_mfdataset(li, combine="nested", concat_dim="time")
        gr = groupby_month(temp)
        datetime_values = pd.to_datetime(['{}-{}'.format(year, month) for year, month in gr['year_month'].values])

        # Create a new DataArray with the datetime values
        new_year_month = xr.DataArray(datetime_values, dims=["year_month"], name="new_year_month")

        # Merge this new DataArray into your dataset
        # This step should replace the existing 'year_month' DataArray
        gr = gr.drop_vars('year_month').assign_coords({'year_month': new_year_month})

        return gr


def plot_taylor_diagram(data_ref, data_models):
    """
    Plots a Taylor diagram for 3D xArray datasets.
    
    :param data_ref: Reference xArray dataset (3D).
    :param data_models: Variable number of xArray datasets to compare (3D).
    """
    # Reduce the reference data to 1D by taking the mean over two dimensions
    data_ref_1d = data_ref.mean(dim=("lat", "lon"))

    # Calculate standard deviation for the reference dataset
    std_ref = data_ref_1d.std(dim='time')

    # Prepare the Taylor diagram
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, polar=True)

    # Add each model dataset to the Taylor diagram
    for data in data_models:
        # Reduce the model data to 1D
        data_1d = data.mean(dim=("lat", "lon"))

        # Calculate standard deviation and correlation
        std_dev = data_1d.std(dim='year_month')
        corr = np.corrcoef(data_ref_1d, data_1d)[0, 1]

        # Convert correlation to radians
        corr_angle = np.arccos(corr)

        # Plot data point
        ax1.plot(corr_angle, std_dev, 'o', label=data.name)

    # Add reference dataset point
    ax1.plot(0, std_ref, 'o', label='Reference')

    # Additional plot settings
    ax1.set_ylim([0, 1.5 * std_ref])
    ax1.set_xlabel('Standard Deviation')
    ax1.set_ylabel('Correlation')
    ax1.legend()

    plt.show()

if __name__ == '__main__':

    with warnings.catch_warnings():
        experiment = 'historical'
        var = 'tas'

        warnings.simplefilter('ignore')
        data_disk = 'I:/NASA-CMIP/CMIP6-Historical-Temp'
        models = [f.name for f in os.scandir(data_disk) if f.is_dir()][:10]

        refdata = xr.open_dataset('Observed_data/tas/air.mon.mean.nc')['air']

        projlist = []
        y=(1990, 1995)
        for model in models:

            try:
                ds = get_monthly(model, experiment, var, years=y)
                if ds is None:
                    print(f'No data for {model}, or data is incorrectly formatted such that retrieving is not possible.')
                else:
                    projlist.append(ds['tas'])

            except:
                raise RuntimeError(f'At model {model}, an error was raised.')

        plot_taylor_diagram(refdata, projlist, years)
