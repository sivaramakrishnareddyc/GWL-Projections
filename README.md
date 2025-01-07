# GWL-Projections

Code for GWL projections

 Sivarama Krishna Reddy Chidepudi, Nicolas Massei, Abderrahim Jardani, et al. Groundwater level projections for aquifers affected by annual to decadal hydroclimate variations. ESS Open Archive . September 02, 2024.
DOI: 10.22541/essoar.172526712.23981307/v1

**Climate data processing**.

Download bias-corrected CMIP6 data of 16 GCMs from NASA NEX-GDDP Dataset by running cmip6_download.py file.

Since the global data is available by years, combine all years into a single file using the combine_netcdf.py file. 

Extract the necessary input projections at the locations of GWL stations using the saving_data.py file.

**DL Models**.

Use DL_ERA5_Multistation_wavelet_train_val.py to train GRU models for each GWL type and check train validation scores.

Use cdf_all.py to plot the metrics.

Use DL_ERA5_Multistation_wavelet.py to  make the projections

Use map_plots.py to compare the trend results on maps.
