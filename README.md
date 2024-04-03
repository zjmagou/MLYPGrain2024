This is the... **Method Implementations for Impact of Temperature Extremes on Grain Yield Projections in the Middle-Lower Yangtze Plains**

# Introduction

   Grain crops are demonstrated to be largely susceptible to the influence of anthropogenic climate change and extreme temperature events (ETEs). However, the impact of the spatio-temporal distribution of ETEs on an area's total grain yield was often neglected in preceding studies. Focusing on the Middle-Lower Yangtze Plains (MLYP), this study proposes a space-aware Deep Learning model that incorporates Convolutional Autoencoders (ConvAEs) and the Random Forest (RF) regression to represent grain yield through socio-economic and meteorological factors. Reanalysis on ScenarioMIP data of 25 NASA's NEX-GDDP-CMIP6 global downscaled CMIP6 models shows that annual heatwave (HWs) continues to rise as coldwave (CWs) mostly declines in the MLYP over the years 2021-2100. The optimized model performs significantly superior to a benchmark FGLS yield model; the encoded HW frequency and the encoded CW frequency each impacts significantly harder on yield than the indicator of labor involved and the encoded spatio-temporal precipitation, and higher levels of projected HW frequency corresponds with more volatile yield over time. From the 2020s until 2100, grain yield in the MLYP could be expected to decrease around 107 tons for SSP1 and SSP2, and 270 tons for SSP5. This study projects lower levels of grain yield decrease in the MLYP both at mid-century and end-of-century compared to previous regional studies utilizing RCP models due to difference in area of study, scenario setup, and the offsetting effect of the declining CW frequency on HW and warming's negative impacts. Additionally, this study supports the literature with further evidence of global warming reducing crop productivity, evaluates the current agricultural policies implemented in the MLYP provinces to promote yield security, and calls for development and implementation of crop-specific mitigation/adaptation strategies against heat \& cold stress.

# Implementation Structure

Implementation for the methods outlined in this study is structured into notebooks of different functions. The main notebook is`rf_and_models.ipynb`, and it is structured as follows:

1. Data preprocessing and auxilary methods for modeling
2. ConvAE spatio-temporal (via stAE) / spatial (via sAE) dimension reduction
3. 8:2 train/validation:test split and grid search and k-fold cross-validation to optmize the RF regression model
4. Evaluate the model with MAPE, Explained Variance Level, and Poisson Deviance
5. Feed dummy data through the trained RF regression model to isolate and verify single-variable impacts
6. Create class wrappers and auxilary methods for future projections

Training of ConvAEs is done in `autoencoder.ipynb` and the benchmark model in `yaumain.ipynb`. Other notebooks are auxilary, and may be useful if figures were to be reproduced.
