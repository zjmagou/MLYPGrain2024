This is the... **Method Implementations for Impact of Temperature Extremes on Grain Yield Projections in the Middle-Lower Yangtze Plains**

# Introduction

*The paper abstract is reproduced.*

   Grain crops are demonstrated to be largely susceptible to the influence of anthropogenic climate change and extreme temperature events (ETEs). However, the impact of the spatio-temporal distribution of ETEs on an area's total grain yield was often neglected in preceding studies. Focusing on the Middle-Lower Yangtze Plains (MLYP), this study proposes a space-aware Deep Learning model that incorporates Convolutional Autoencoders (ConvAEs) and the Random Forest (RF) regression to represent grain yield through socio-economic and meteorological factors. Reanalysis on ScenarioMIP data of 25 NASA's NEX-GDDP-CMIP6 global downscaled CMIP6 models shows that annual heatwave (HWs) continues to rise as coldwave (CWs) mostly declines in the MLYP over the years 2021-2100. The optimized model performs significantly superior to a benchmark FGLS yield model; the encoded HW frequency and the encoded CW frequency each impacts significantly harder on yield than the indicator of labor involved and the encoded spatio-temporal precipitation, and higher levels of projected HW frequency corresponds with more volatile yield over time. From the 2020s until 2100, grain yield in the MLYP could be expected to decrease around 107 tons for SSP1 and SSP2, and 270 tons for SSP5. This study projects lower levels of grain yield decrease in the MLYP both at mid-century and end-of-century compared to previous regional studies utilizing RCP models due to difference in area of study, scenario setup, and the offsetting effect of the declining CW frequency on HW and warming's negative impacts. Additionally, this study supports the literature with further evidence of global warming reducing crop productivity, evaluates the current agricultural policies implemented in the MLYP provinces to promote yield security, and calls for development and implementation of crop-specific mitigation/adaptation strategies against heat \& cold stress.

# File Structure and Availability

Under the directory at which the repo is situated are other datasets that supply the data with which the model is trained and from which projections are made. 

AMES NEX GDDP CMIP6 GCM data are in the `/NASA-CMIP/` folder, which needs to be created and populated with data before running any methods. It has the following subfolders:

- `/NASA-CMIP/CMIP6-Historical-PR/`holds pr of the CMIP6 historical experiment of 25 GCMs.
- `/NASA-CMIP/CMIP6-Historical-Temps/`holds tas, tasmax and tasmin of the CMIP6 historical experiment of 25 GCMs.
- `/NASA-CMIP/CMIP6-SSP-PR/{model}/` holds pr of the CMIP6 ScenarioMIP SSP126, SSP245, SSP370, SSP585 experiments of 25 GCMs, each in a folder with the GCM name as folder name.
- `/NASA-CMIP/CMIP6-SSP-Temps/{model}/`holds tas, tasmax and tasmin of the CMIP6 ScenarioMIP SSP126, SSP245, SSP370, SSP585 experiments of 25 GCMs, each in a folder with the GCM name as folder name.

Statistically reanalyzed HW and CW frequency data are in the `/waves/` folder, which will be automatically created by`cwave.py` and `nasa_cwave.py` if not created already. Clipped climate data are in the `/formatted_grid/` folder,  which will be automatically created by `cut_to_size.py` if not created already. Output data will be stored automatically in `/final_output/`.

The following source datasets related to this project are available on their respective locations:
- Raw **AMES NEX GDDP CMIP6** produced by [Thrasher et al., 2022](https://www.nature.com/articles/s41597-022-01393-4) is available on NASA's [NCCS THREDDS Data Server](https://ds.nccs.nasa.gov/thredds/catalog/AMES/NEX/GDDP-CMIP6/catalog.html)
- Dataset from public source **Observed_data** (compressed folder) is available on Microsoft OneDrive as [Observed_data.zip](https://bssgj-my.sharepoint.com/:u:/g/personal/michael_mu13973-binj_basischina_com/ETXlnzOAXLZBhCMZQxf6wxABP9ovb5bY542BR_Asqyb6Xw?e=pUhIKv)
- Statistically reanalyzed dataset **waves** (compressed folder) is available on Microsoft OneDrive as [waves.zip](https://bssgj-my.sharepoint.com/:u:/g/personal/michael_mu13973-binj_basischina_com/EdjJqEPY27dAlpgWm9at0AsBGfoy3cxeMx9fjZGs1CRW6w?e=f46eGX)

The following intermediate datasets are available on Zenodo via [this DOI](https://doi.org/10.5281/zenodo.10924805):
- Model input dataset **formatted_grid** (compressed folder)
- Model auxilary files `hwave_order.csv` and `coldwave_order.csv`
- Model projection dataset **final_outputs** (compressed folder)
- Raw data used to plot Taylor diagrams and separating yield into ensembles are available (compressed folder)

# Method Structure

Data preprocessing is automated into python files: `cwave.py` and `nasa_cwave.py` converts tasmax and tasmin data into CW and HW frequencies, and `cut_to_size.py` clips climate data to the bounds of the MLYP provinces. Implementation for the methods outlined in this study is structured into notebooks of different functions. Training of ConvAEs is done in `autoencoder.ipynb` and the benchmark model in `yaumain.ipynb`. `modelsort.ipynb` is used to produce `coldwave_order.csv` and `heatwave_order.csv`, which can be found on Zenodo `plots.ipynb` might be useful if figures are to be reproduced. The main notebook where RF Regression is trained and projection is made is`rf_and_models.ipynb`, and it is structured as follows:

1. Data preprocessing and auxiliary methods for modeling
2. ConvAE spatio-temporal (via stAE) / spatial (via sAE) dimension reduction on observed data
3. 8:2 train/validation:test split on observed data and grid search and k-fold cross-validation to optimize the RF regression model
4. Evaluate the model with MAPE, Explained Variance Level, and Poisson Deviance
5. Feed dummy data through the trained RF regression model to isolate and verify single-variable impacts
6. Create class wrappers and automate steps 1 and 2 for future projections
7. Plotting

# Reproducing results

Download the repo into an empty directory. Install all the packages in `MzjEnv_new.yaml`. It is strongly recommended to import the YAML into a compatible package manager (such as Anaconda) and create a new virtual environment dedicated to this project.

The **intermediate datasets** are required for a reproduction of the study's proposed model. A minimum storage of 13 GB will be needed for reproducing the proposed model with trained autoencoders.

The **source datasets** are required for a full reproduction of the study's data analysis. A minimum storage of 10 TB will be needed for reproducing the proposed model with trained autoencoders AND results related to ScenarioMIP future projection experiments. You will need to collect related variables and data of AMES NEX GDDP CMIP6 and format the file structure into the one outlined in previous subsections.
