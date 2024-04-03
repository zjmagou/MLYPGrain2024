import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import glob
import numpy as np
import torch
import os
import gc

from xarray import DataArray


# Function to interpolate threshold data
def interpolate_threshold_data(threshold_data, ds):
    threshold_data_gpu = torch.from_numpy(threshold_data).unsqueeze(0).unsqueeze(0).float().cuda()

    # Calculate the output size
    output_size = (int(threshold_data.shape[0] * 2), int(threshold_data.shape[1] * 2))

    # Use PyTorch for zooming
    interpolated_threshold_gpu = torch.nn.functional.interpolate(threshold_data_gpu, size=output_size, mode='bilinear', align_corners=False)

    # Squeeze to remove extra dimensions and move data back to CPU
    interpolated_threshold = interpolated_threshold_gpu.squeeze().cpu().numpy()
    
    # Match the latitude range of ds
    # Find the indices in the interpolated data that correspond to the latitudes in ds
    lat_indices = np.where(
        (ds.coords["lat"].values[0] <= ds.coords["lat"]) & (ds.coords["lat"] <= ds.coords["lat"].values[-1])
    )[0]
    lat_indices = np.full(len(lat_indices), len(interpolated_threshold)) - lat_indices - 1
    
    # Crop the interpolated data to these indices
    cropped_threshold = interpolated_threshold[sorted(lat_indices), :]
    
    return xr.DataArray(
        cropped_threshold,
        dims=("lat", "lon"),
        coords={
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
        },
    )


def process_yearly_data(yearly_files_and_thresholds):

    coldwave_file, heatwave_file, coldwave_threshold_data, heatwave_threshold_data = yearly_files_and_thresholds
    
    # Define a function to detect waves (both cold and heatwaves)
    def detect_waves_gpu(candidate_days):
        # Convert the input NumPy array to a PyTorch tensor
        candidate_days_tensor = torch.tensor(candidate_days.to_numpy(), device='cuda')
        
        # Replace NaN values with 0
        candidate_days_tensor = candidate_days_tensor.type(torch.int32)
        candidate_days_tensor[torch.isnan(candidate_days_tensor)] = 0
        
        # Calculate the difference along 'time' axis
        # Assuming 'time' axis is the first axis (axis=0)
        diff_tensor = torch.diff(candidate_days_tensor, dim=0)
        
        # Count the positive differences
        result_tensor = (diff_tensor > 0).sum(dim=0)
        
        # Convert the result back to a NumPy array if needed (here we'll keep it on GPU)
        result = result_tensor.cpu().numpy() # Uncomment this line if you need the result on CPU as a NumPy array
        
        return xr.DataArray(
            result,
            dims=("lat", "lon"),
            coords={
                "lat": candidate_days.coords["lat"],
                "lon": candidate_days.coords["lon"],
            },
        )
    
    def detect_waves(candidate_days):
        return (candidate_days.astype('int8', casting='unsafe').fillna(0).diff('time') > 0).sum('time')
    
    try:
    
        ds_min = xr.open_dataset(coldwave_file)
        ds_max = xr.open_dataset(heatwave_file)
        
        # Interpolate both coldwave and heatwave threshold data
        coldwave_threshold_da = interpolate_threshold_data(coldwave_threshold_data, ds_min)
        heatwave_threshold_da = interpolate_threshold_data(heatwave_threshold_data, ds_max)
        
        # Directly compute coldwaves and heatwaves for each pixel
        coldwave_candidate: DataArray = (ds_min['tasmin'] < coldwave_threshold_da)
        coldwave = detect_waves(coldwave_candidate)

        del ds_min, coldwave_candidate, coldwave_threshold_da
        gc.collect()

        heatwave_candidate: DataArray = (ds_max['tasmax'] > heatwave_threshold_da)
        heatwave = detect_waves(heatwave_candidate)

        del ds_max, heatwave_candidate, heatwave_threshold_da
        gc.collect()
        
    except RuntimeError as e:
        raise RuntimeError(f"Runtime error of {coldwave_file} or {heatwave_file}") from e
    
    return coldwave, heatwave


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

    gc.collect()

def extract_year_from_filepath(filepath):
    # Assuming filenames are in the format: "someprefix_YEAR.nc"
    # Modify this function based on your actual filename format to extract the year
    if filepath.split('_')[-1].split('.')[0].startswith('v'):
        return int(filepath.split('_')[-2])
    else:
        return int(filepath.split('_')[-1].split('.')[0])

def main(data_disk='I:/NASA-CMIP', isK=True):
    flip_lat = True
    flip_lon = False
    experiments = ['historical']
    models = [f.name for f in os.scandir('D:/formatted_grid') if f.is_dir()]#[14:]
    
    # Assuming the structure of the threshold files remains constant
    coldwave_threshold_file = 'tmin_thr10.nc'
    heatwave_threshold_file = 'tmax_thr90.nc'
    
    # Read the threshold data
    coldwave_threshold_ds = xr.open_dataset(coldwave_threshold_file)
    heatwave_threshold_ds = xr.open_dataset(heatwave_threshold_file)
    
    if flip_lon:
        coldwave_threshold_ds = coldwave_threshold_ds.assign_coords(
            lon=(((coldwave_threshold_ds.lon + 180) % 360) - 180)
        )
        
        heatwave_threshold_ds = heatwave_threshold_ds.assign_coords(
            lon=(((heatwave_threshold_ds.lon + 180) % 360) - 180)
        )
        
    if flip_lat:
        coldwave_threshold_ds = coldwave_threshold_ds.reindex(lat=coldwave_threshold_ds.lat[::-1])
        heatwave_threshold_ds = heatwave_threshold_ds.reindex(lat=heatwave_threshold_ds.lat[::-1])

    if flip_lon:
        coldwave_threshold_ds = coldwave_threshold_ds.sortby('lon')
        heatwave_threshold_ds = heatwave_threshold_ds.sortby('lon')
    
    if flip_lat:
        coldwave_threshold_ds = coldwave_threshold_ds.sortby('lat')
        heatwave_threshold_ds = heatwave_threshold_ds.sortby('lat')

    coldwave_threshold_data = np.nan_to_num(coldwave_threshold_ds['tmin'].values) + 273.15
    heatwave_threshold_data = np.nan_to_num(heatwave_threshold_ds['TmaxThreshold'][0, :].values) + 273.15

    del coldwave_threshold_ds, heatwave_threshold_ds
    gc.collect()

    
    for experiment in experiments:
        for model in models:
            if os.path.exists(f"waves/{experiment}_{model}_coldwave.nc"):
                print(f'Already processed for {experiment}_{model}, skipping.')
            elif os.path.exists(f"{experiment}_{model}_coldwave.nc"):
                print(f'Already processed for {experiment}_{model}, skipping.')        
            else:
                print(f'Processing for {experiment}_{model}.')
                try:
                    process_experiment_model(experiment, model, coldwave_threshold_data, heatwave_threshold_data, isK, flip_lat, flip_lon, disk=data_disk)
                except error as e:
                    print(f'Nothing made for {model}. Check later.')
                    #pass
   
def process_experiment_model(experiment, model, coldwave_threshold_data, heatwave_threshold_data, isK, flip_lat, flip_lon, disk='I:/NASA-CMIP'):

    coldwave_yearly_files = sorted(glob.glob(f'{disk}/{model}/tasmin_day_{model}_{experiment}_*.nc'))
    heatwave_yearly_files = sorted(glob.glob(f'{disk}/{model}/tasmax_day_{model}_{experiment}_*.nc'))
    
    if len(coldwave_yearly_files) > 0:
        if len(heatwave_yearly_files) > 0:
            
            coldwave_yearly_files = [s.replace("\\", "/") for s in coldwave_yearly_files]
            heatwave_yearly_files = [s.replace("\\", "/") for s in heatwave_yearly_files]
            
            combined_args = zip(coldwave_yearly_files, heatwave_yearly_files, 
                                [np.asarray(coldwave_threshold_data)] * len(coldwave_yearly_files), 
                                [np.asarray(heatwave_threshold_data)] * len(heatwave_yearly_files))
        
            frequencies = {'coldwave': [], 'heatwave': []}
            years = [extract_year_from_filepath(fp) for fp in coldwave_yearly_files]
        
            import warnings
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with ProcessPoolExecutor() as executor:
                    for coldwave_freq, heatwave_freq in executor.map(process_yearly_data, combined_args):
                        frequencies['coldwave'].append(coldwave_freq)
                        frequencies['heatwave'].append(heatwave_freq)
            
            save_to_netcdf(frequencies, years, f'{experiment}_{model}')
            print(f"Made netCDFs for {model}.")
            
        else: print(f"No data available for model {model}.")
            
    else: print(f"No data available for model {model}.")

if __name__ == '__main__':
        main(data_disk='D:/NASA-CMIP/CMIP6-Historical-Temp')