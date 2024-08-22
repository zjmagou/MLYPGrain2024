This is the... **Method Implementations for A Space-Aware Analysis of Future Climate-driven Grain Yield Variability in the Middle-Lower Yangtze Plains: Focusing on CMIP6-Projected Temperature Extremes**

# Introduction

   Grain crops are vulnerable to anthropogenic climate change and extreme temperature events (ETEs). Reanalysis of ScenarioMIP data from 25 NASA's NEX-GDDP-CMIP6 downscaled global coupled models indicates an increase in heatwave (HW) frequency and a general decline in coldwave (CW) frequency across the MLYP during 2021-2100. However, the impact of the spatio-temporal distribution of ETEs on an area's total grain yield was often neglected in previous studies. Focusing on the Middle-Lower Yangtze Plains (MLYP), this study proposes a space-aware Deep Learning model that incorporates Convolutional Autoencoders and the Random Forest regression to represent grain yield through socio-economic and meteorological factors. The proposed model performs significantly superior to the benchmark multilinear yield model. By 2100, grain yield over the MLYP is projected to decrease by over 100 tons for the low-radiative-forcing/sustainable development scenario (SSP126) and the medium-radiative-forcing scenario (SSP245), and about 270 tons for the high-radiative-forcing/fossil-fueled development scenario (SSP585). Grain yield may experience less decline than previously projected by studies using Representative Concentration Pathways (RCPs). This difference is likely due to a decrease in CWs, which can offset the effects of more frequent HWs on grain yield, combined with alterations in supply-side policies. Notably, the frequency of encoded HWs and CWs has a stronger impact on grain yield compared to precipitation and labor indicator; higher levels of projected HW frequency correspond with increased yield volatility over time. This study emphasizes the need for developing crop-specific mitigation/adaptation strategies against heat and cold stress amidst global warming.

# File Structure and Availability

Under the directory at which the repo is situated are other datasets that supply the data with which the model is trained and from which projections are made. 

AMES NEX GDDP CMIP6 GCM data are in the `/NASA-CMIP/` folder, which needs to be created and populated with data before running any methods. It has the following subfolders:

- `/NASA-CMIP/CMIP6-Historical-PR/`holds pr of the CMIP6 historical experiment of 25 GCMs.
- `/NASA-CMIP/CMIP6-Historical-Temps/`holds tas, tasmax and tasmin of the CMIP6 historical experiment of 25 GCMs.
- `/NASA-CMIP/CMIP6-SSP-PR/{model}/` holds pr of the CMIP6 ScenarioMIP SSP126, SSP245, SSP370, SSP585 experiments of 25 GCMs, each in a folder with the GCM name as folder name.
- `/NASA-CMIP/CMIP6-SSP-Temps/{model}/`holds tas, tasmax and tasmin of the CMIP6 ScenarioMIP SSP126, SSP245, SSP370, SSP585 experiments of 25 GCMs, each in a folder with the GCM name as folder name.

Statistically reanalyzed HW and CW frequency data are in the `/waves/` folder, which will be automatically created by`cwave.py` and `nasa_cwave.py` if not created already. Clipped climate data are in the `/formatted_grid/` folder,  which will be automatically created by `cut_to_size.py` if not created already. Output data will be stored automatically in `/final_output/`.

The following ***source datasets*** related to this project are available on their respective locations:
- Raw **AMES NEX GDDP CMIP6** produced by [Thrasher et al., 2022](https://www.nature.com/articles/s41597-022-01393-4) is available on NASA's [NCCS THREDDS Data Server](https://ds.nccs.nasa.gov/thredds/catalog/AMES/NEX/GDDP-CMIP6/catalog.html)
- Dataset from public source **Observed_data** (compressed folder) is available on Microsoft OneDrive as [Observed_data.zip](https://bssgj-my.sharepoint.com/:u:/g/personal/michael_mu13973-binj_basischina_com/ETXlnzOAXLZBhCMZQxf6wxABP9ovb5bY542BR_Asqyb6Xw?e=pUhIKv)
- Statistically reanalyzed dataset **waves** (compressed folder) is available on Microsoft OneDrive as [waves.zip](https://bssgj-my.sharepoint.com/:u:/g/personal/michael_mu13973-binj_basischina_com/EdjJqEPY27dAlpgWm9at0AsBGfoy3cxeMx9fjZGs1CRW6w?e=f46eGX)

The following ***intermediate datasets*** are available on Zenodo via [this DOI](https://doi.org/10.5281/zenodo.10924805):
- Model input dataset **formatted_grid** (compressed folder) for ConvAE and RF training, evaluation and future projection
- ETE threshold datasets `tmax_90.nc` for hot days and `tmin_10.nc` for cold days.
- Auxilary order files `hwave_order.csv` and `coldwave_order.csv` for projection visualization
- Model projection dataset **final_outputs** (compressed folder)

# Method Structure

Data preprocessing is automated into python files: `cwave.py` and `nasa_cwave.py` converts tasmax and tasmin data into CW and HW frequencies, and `cut_to_size.py` clips climate data to the bounds of the MLYP provinces.

The models and methods outlined in this study are implemented through Jupyter notebooks of different functions. Training of ConvAEs is done in `autoencoder.ipynb` and the benchmark model in `yaumain.ipynb`. `modelsort.ipynb` is used to produce `coldwave_order.csv` and `heatwave_order.csv`, which can be found on Zenodo. `plots.ipynb` might be useful if figures are to be reproduced. The main notebook where RF Regression is trained and projection is made is`rf_and_models.ipynb`, and it is structured as follows:

1. Data preprocessing and auxiliary methods for modeling
2. ConvAE spatio-temporal (via stAE) / spatial (via sAE) dimension reduction on observed data
3. 8:2 train/validation:test split on observed data and grid search and k-fold cross-validation to optimize the RF regression model
4. Evaluate the model with MAPE, Explained Variance Level, and Poisson Deviance
5. Feed dummy data through the trained RF regression model to isolate and verify single-variable impacts
6. Create class wrappers and automate steps 1 and 2 for future projections
7. Plotting

# Reproducing results

Download the repo into an empty directory. Install all the packages in `MzjEnv_new.yaml`. It is strongly recommended to import the YAML into a compatible package manager (such as Anaconda) and create a new virtual environment dedicated to this project.

The ***intermediate datasets*** are required for a reproduction of the study's proposed model. A minimum storage of 13 GB will be needed for reproducing the proposed model with trained autoencoders.

The ***source datasets*** are required for a full reproduction of the study's data analysis. A minimum storage of 10 TB will be needed for reproducing the proposed model with trained autoencoders AND results related to ScenarioMIP future projection experiments. You will need to collect related variables and data of AMES NEX GDDP CMIP6 and format the file structure into the one outlined in previous subsections.

Benchmark experiment with B2-RF can be found in `rf_and_models.ipynb`. Certain independent pellucid implementations of the study's methods may have not been included. We welcome information / inquiries.
