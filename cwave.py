from scipy import ndimage
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import glob
import ast
import numpy as np

def process_yearly_data(yearly_files_and_thresholds):
    coldwave_file, heatwave_file, coldwave_threshold_data, heatwave_threshold_data = yearly_files_and_thresholds
    ds_min = xr.open_dataset(coldwave_file)
    ds_max = xr.open_dataset(heatwave_file)
    
    # Function to interpolate threshold data
    def interpolate_threshold_data(threshold_data, ds):
        zoom_factors = (ds.dims['lat'] / 360, ds.dims['lon'] / 720)
        interpolated_threshold = ndimage.zoom(threshold_data, zoom_factors, order=1)
        
        return xr.DataArray(
            interpolated_threshold,
            dims=("lat", "lon"),
            coords={
                "lat": ds.coords["lat"],
                "lon": ds.coords["lon"],
            },
        )

    
    # Interpolate both coldwave and heatwave threshold data
    coldwave_threshold_da = interpolate_threshold_data(coldwave_threshold_data, ds_min)
    heatwave_threshold_da = interpolate_threshold_data(heatwave_threshold_data, ds_max)
    
    # Identify coldwave and heatwave candidate days
    coldwave_candidate = (ds_min['tmin'] < coldwave_threshold_da)
    heatwave_candidate = (ds_max['tmax'] > heatwave_threshold_da)
    
    # Define a function to detect waves (both cold and heatwaves)
    def detect_waves(candidate_days):
        return (candidate_days.astype(int, casting='unsafe').fillna(0).diff('time') > 0).sum('time')
    
    # Directly compute coldwaves and heatwaves for each pixel
    coldwave = detect_waves(coldwave_candidate)
    heatwave = detect_waves(heatwave_candidate)


    # Identify the start of each coldwave and heatwave
    #coldwave_start = coldwave_candidate & (~coldwave_candidate.shift(time=1, fill_value=False)) 
    #heatwave_start = heatwave_candidate & (~heatwave_candidate.shift(time=1, fill_value=False))

    # Count coldwave and heatwave events per year for each pixel
    coldwave_frequency = coldwave
    heatwave_frequency = heatwave
    
    return coldwave_frequency, heatwave_frequency


def save_to_netcdf(frequencies, years, output_file_base):
    coldwave_ds_list = []
    heatwave_ds_list = []

    for idx, year in enumerate(years):
        coldwave_ds = xr.Dataset(
            {
                "coldwave_frequency": (["lat", "lon"], frequencies['coldwave'][idx].data)
            },
            coords={
                "lat": (["lat"], frequencies['coldwave'][idx].coords["lat"].data),
                "lon": (["lon"], frequencies['coldwave'][idx].coords["lon"].data),
                "time": year
            }
        )
        
        heatwave_ds = xr.Dataset(
            {
                "heatwave_frequency": (["lat", "lon"], frequencies['heatwave'][idx].data)
            },
            coords={
                "lat": (["lat"], frequencies['heatwave'][idx].coords["lat"].data),
                "lon": (["lon"], frequencies['heatwave'][idx].coords["lon"].data),
                "time": year
            }
        )

        coldwave_ds_list.append(coldwave_ds)
        heatwave_ds_list.append(heatwave_ds)

    combined_coldwave_ds = xr.concat(coldwave_ds_list, dim='time')
    combined_heatwave_ds = xr.concat(heatwave_ds_list, dim='time')

    combined_coldwave_ds.to_netcdf(f"{output_file_base}_coldwave.nc")
    combined_heatwave_ds.to_netcdf(f"{output_file_base}_heatwave.nc")

def extract_year_from_filepath(filepath):
    # Assuming filenames are in the format: "someprefix_YEAR.nc"
    # Modify this function based on your actual filename format to extract the year
    return int(filepath.split('.')[1])

def main(disk='I:/Observed_data'):

    # Assuming the structure of the threshold files remains constant
    coldwave_threshold_file = 'tmin_thr10.nc'
    heatwave_threshold_file = 'tmax_thr90.nc'

    coldwave_threshold_ds = xr.open_dataset(coldwave_threshold_file)
    heatwave_threshold_ds = xr.open_dataset(heatwave_threshold_file)

    # Read the threshold data
    coldwave_threshold_ds = xr.open_dataset(coldwave_threshold_file)
    heatwave_threshold_ds = xr.open_dataset(heatwave_threshold_file)
    

    # Only adjust longitude
    coldwave_threshold_ds = coldwave_threshold_ds.assign_coords(
        lon=(((coldwave_threshold_ds.lon + 180) % 360) - 180)
    )
    
    # Sorting by longitude (and latitude if flipped)
    coldwave_threshold_ds = coldwave_threshold_ds.sortby('lon')
    heatwave_threshold_ds = heatwave_threshold_ds.sortby('lon')
    

    coldwave_threshold_data = np.nan_to_num(coldwave_threshold_ds['tmin'].values)
    heatwave_threshold_data = np.nan_to_num(heatwave_threshold_ds['TmaxThreshold'][0, :].values)
    
    coldwave_yearly_files = sorted(glob.glob(f'{disk}/tasmin/*.nc'))
    heatwave_yearly_files = sorted(glob.glob(f'{disk}/tasmax/*.nc'))

    combined_args = zip(coldwave_yearly_files, heatwave_yearly_files, 
                        [coldwave_threshold_data] * len(coldwave_yearly_files), 
                        [heatwave_threshold_data] * len(heatwave_yearly_files))

    frequencies = {'coldwave': [], 'heatwave': []}
    years = [extract_year_from_filepath(fp) for fp in coldwave_yearly_files]

    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ProcessPoolExecutor() as executor:
            for coldwave_freq, heatwave_freq in executor.map(process_yearly_data, combined_args):
                frequencies['coldwave'].append(coldwave_freq)
                frequencies['heatwave'].append(heatwave_freq)

    save_to_netcdf(frequencies, years, 'cpc_observed')

if __name__ == '__main__':
        main()
