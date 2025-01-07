#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 20:28:22 2023

@author: chidesiv
"""



# # List of models
# models = [
#     'ACCESS-CM2', 'CanESM5']#, 'CNRM-ESM2-1', 'EC-Earth3-Veg-LR', 'GFDL-CM4', 'INM-CM4-8',
# #     'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'UKESM1-0-LL', 'BCC-ESM1',
# #     'CESM2', 'CESM2-WACCM', 'CNRM-CM6-1', 'GFDL-ESM4', 'INM-CM5-0', 'IPSL-CM6A-ATM-HR', 'MIROC-ES2L', 'MPI-ESM-1-2-HAM',
# #     'MRI-ESM2-0', 'NorESM2-MM', 'SAM0-UNICON', 'AWI-CM-1-1-MR', 'EC-Earth3', 'FGOALS-g3', 'HadGEM3-GC31-LL',
# # #     'HadGEM3-GC31-MM', 'KACE-1-0-G', 'MCM-UA-1-0', 'NESM3', 'TaiESM1', 'GISS-E2-1-G'
# # ]




# # Selected models: ACCESS-ESM1-5,'CanESM5', CMCC-ESM2, 'CNRM-CM6-1', EC-Earth3 , 'GFDL-ESM4', GISS-E2-1-G, FGOALS-g3,INM-CM5-0,IPSL-CM6A-LR ,MIROC6 ,MPI-ESM1-2-HR
# #MRI-ESM2-0,NorESM2-MM,UKESM1-0-LL,TaiESM1



            
           

# import os
# import urllib.request
# import subprocess

# selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'GISS-E2-1-G', 'FGOALS-g3',
#                    'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1']

# scenarios = ['ssp370', 'ssp585', 'historical']  # 'ssp245',
# variables = ['pr', 'tas']

# base_directory = "/media/chidesiv/DATA2/Climate_projections/Data/"

# for model in selected_models:
#     for scenario in scenarios:
#         for variable in variables:
#             base_url = f"https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/NEX-GDDP-CMIP6/{model}/{scenario}/r1i1p1f1/{variable}/{variable}_day_{model}_{scenario}_r1i1p1f1_gn_"
#             GCM1 = [f"{base_url}{year}.nc" for year in range(2015, 2101)]
            
#             output_directory = os.path.join(base_directory, "BiasCorrected", model, scenario, variable)
#             os.makedirs(output_directory, exist_ok=True)
            
#             # Download files if they don't exist
#             for i in range(len(GCM1)):
#                 Save_GCM = os.path.join(output_directory, f"{variable}_day_{model}_{scenario}_2015_2100.nc")
#                 if not os.path.exists(Save_GCM):
#                     urllib.request.urlretrieve(GCM1[i], Save_GCM)
            
#             # Merge files
#             merge_command = f"cdo mergetime {variable}_day*.nc {variable}_day_{model}_{scenario}_2015_2100.nc"
#             subprocess.run(merge_command, shell=True, cwd=output_directory)
            
#             # Select region
#             region_command = f"cdo sellonlatbox,-7,12,40,54 {variable}_day_{model}_{scenario}_2015_2100.nc FR_{variable}_day_{model}_{scenario}_2015_2100.nc"
#             subprocess.run(region_command, shell=True, cwd=output_directory)
            
#             # Remove intermediate files
#             intermediate_files = [os.path.join(output_directory, f"{variable}_day_{model}_{scenario}_{year}.nc") for year in range(2015, 2101)]
#             for file in intermediate_files:
#                 if os.path.exists(file):
#                     os.remove(file)





import os
import xarray as xr
import pandas as pd
import numpy as np
import urllib.request

# Replace these with the variables and scenarios you are interested in
variables = [ 'rsds','']
scenarios = [ 'ssp585','', '']

# selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'GISS-E2-1-G', 'FGOALS-g3',
#                     'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1']

selected_models = [ 'TaiESM1','',] #'CNRM-CM6-1','GISS-E2-1-G''FGOALS-g3',

base_directory = "/media/chidesiv/DATA1/Bias_corrected_GCMs/Data/"

for model in selected_models:
    for scenario in scenarios:
        for variable in variables:
            
            base_url = f"https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/NEX-GDDP-CMIP6/{model}/{scenario}/r1i1p1f1/{variable}/{variable}_day_{model}_{scenario}_r1i1p1f1_gn_"
            GCM1 = [f"{base_url}{year}.nc" for year in range(2015, 2101)]

            # Base path for Save_GCM
            base_directory = "/media/chidesiv/DATA1/Bias_corrected_GCMs/Data/"
            output_directory = f"{base_directory}{model}/{scenario}/{variable}/"

            # Generate paths for the years 2015 to 2100
            Save_GCM = [f"{output_directory}{variable}_day_{model}_{scenario}_{year}.nc" for year in range(2015, 2101)]

            # Create the output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            # Download files
            for i in range(len(GCM1)):
                urllib.request.urlretrieve(GCM1[i], Save_GCM[i])
                
#             ds=xr.open_dataset(f"{output_directory}{variable}_day_{model}_{scenario}_{year}.nc" for year in range(2015, 2101))
            
            
#             # # Process the downloaded files
#             # years = range(2015, 2101)
#             # ds_list = []
    
#             # for year in years:
#             #     file_name = os.path.join(output_directory, f'{variable}_day_{model}_{scenario}_{year}.nc')
                
#             #     try:
#             #         ds = xr.open_dataset(file_name)
#             #     except FileNotFoundError:
#             #         continue
                
#             #     data = ds[variable].values
#             #     lats, lons = ds['lat'].values, ds['lon'].values
#             #     lats = lats[np.newaxis, :, :]
#             #     lons = lons[np.newaxis, :, :]

#             #     time = pd.date_range(str(ds.time.values[0]), periods=data.shape[0], freq='D').to_numpy()
#             #     ds = xr.Dataset({variable: (['time', 'lat', 'lon'], data)},
#             #                     coords={'lat': (['lat'], lats, {'units': 'degrees_north', 'long_name': 'latitude'}),
#             #                             'lon': (['lon'], lons, {'units': 'degrees_east', 'long_name': 'longitude'}),
#             #                             'time': time},
#             #                     attrs={'units': ds[variable].units, 'long_name': ds[variable].long_name})

#             #     ds_list.append(ds)

#             # # Concatenate the datasets along the 'time' dimension
#             # ds_concat = xr.concat(ds_list, dim='time')

#             # # Write the concatenated dataset to a new NetCDF file
#             # output_file = os.path.join(output_directory, f'combined_data_{model}_{scenario}.nc')
#             # ds_concat.to_netcdf(output_file)



# import os
# import xarray as xr
# import pandas as pd
# import numpy as np
# import urllib.request

# # Replace these with the variables and scenarios you are interested in
# variables = ['pr', 'tas']
# scenarios = ['ssp245', 'ssp370', 'ssp585', 'historical']

# selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'GISS-E2-1-G', 'FGOALS-g3',
#                    'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1']

# base_directory = "/media/chidesiv/DATA2/Climate_projections/Data/"

# for model in selected_models:
#     for scenario in scenarios:
#         for variable in variables:
            
#             # base_url = f"https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/NEX-GDDP-CMIP6/{model}/{scenario}/r1i1p1f1/{variable}/{variable}_day_{model}_{scenario}_r1i1p1f1_gn_"
#             # GCM1 = [f"{base_url}{year}.nc" for year in range(2015, 2101)]

#             # Base path for Save_GCM
#             base_directory = "/media/chidesiv/DATA2/Climate_projections/Data/"
#             output_directory = f"{base_directory}BiasCorrected/{model}/{scenario}/{variable}/"

#             # # Generate paths for the years 2015 to 2100
#             # Save_GCM = [f"{output_directory}{variable}_day_{model}_{scenario}_{year}.nc" for year in range(2015, 2101)]

#             # # Create the output directory if it doesn't exist
#             # os.makedirs(output_directory, exist_ok=True)

#             # # Download files
#             # for i in range(len(GCM1)):
#             #     urllib.request.urlretrieve(GCM1[i], Save_GCM[i])
                
#             ds=xr.open_dataset(f"{output_directory}{variable}_day_{model}_{scenario}_{year}.nc" for year in range(2015, 2101))
            
            
#             # # Process the downloaded files
#             # years = range(2015, 2101)
#             # ds_list = []
    
#             # for year in years:
#             #     file_name = os.path.join(output_directory, f'{variable}_day_{model}_{scenario}_{year}.nc')
                
#             #     try:
#             #         ds = xr.open_dataset(file_name)
#             #     except FileNotFoundError:
#             #         continue
                
#             #     data = ds[variable].values
#             #     lats, lons = ds['lat'].values, ds['lon'].values
#             #     lats = lats[np.newaxis, :, :]
#             #     lons = lons[np.newaxis, :, :]

#             #     time = pd.date_range(str(ds.time.values[0]), periods=data.shape[0], freq='D').to_numpy()
#             #     ds = xr.Dataset({variable: (['time', 'lat', 'lon'], data)},
#             #                     coords={'lat': (['lat'], lats, {'units': 'degrees_north', 'long_name': 'latitude'}),
#             #                             'lon': (['lon'], lons, {'units': 'degrees_east', 'long_name': 'longitude'}),
#             #                             'time': time},
#             #                     attrs={'units': ds[variable].units, 'long_name': ds[variable].long_name})

#             #     ds_list.append(ds)

#             # # Concatenate the datasets along the 'time' dimension
#             # ds_concat = xr.concat(ds_list, dim='time')

#             # # Write the concatenated dataset to a new NetCDF file
#             # output_file = os.path.join(output_directory, f'combined_data_{model}_{scenario}.nc')
#             # ds_concat.to_netcdf(output_file)


# import os
# import xarray as xr
# import urllib.request

# # Replace these with the variables and scenarios you are interested in
# variables = ['pr', 'tas']
# scenarios = ['ssp245', 'ssp370','ssp585']

# selected_models = ['ACCESS-ESM1-5']  # Add other models as needed
# base_directory = "/media/chidesiv/DATA2/Climate_projections/Data/"

# for model in selected_models:
#     for scenario in scenarios:
#         for variable in variables:
#             output_directory = os.path.join(base_directory, "BiasCorrected", model, scenario, variable)

#             # Merge files for each variable
#             pattern = os.path.join(output_directory, f"{variable}_day_{model}_{scenario}_*.nc")
#              # List of files from 2015 to 2100
#             files = [pattern.replace('*', str(year)) for year in range(2015, 2101)]

#             print(files)
#             ds = xr.open_mfdataset(files, engine='netcdf4', combine="by_coords")

#             # Select region
#             ds_region = ds.sel(lon=slice(-12, 12), lat=slice(40, 54))

#             # Save the result as a new NetCDF file
#             output_file = os.path.join(output_directory, f"combined_dataFR_{model}_{scenario}_{variable}.nc")
#             ds_region.to_netcdf(output_file)



#             ds = xr.open_mfdataset(files, engine='netcdf4', combine="by_coords")

#             # Select region
#             ds_region = ds.sel(lon=slice(-12, 12), lat=slice(40, 54))

#             # Save the result as a new NetCDF file
#             output_file = os.path.join(output_directory, f"combined_dataFR_{model}_{scenario}_{variable}.nc")
#             ds_region.to_netcdf(output_file)






# import rioxarray
# import xarray as xr
# import os
# import glob

# # Replace these with the variables and scenarios you are interested in
# variables = ['pr', 'tas']
# scenarios = ['ssp245', 'ssp370', 'ssp585']
# selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2']  # Add other models as needed
# base_directory = "/media/chidesiv/DATA2/Climate_projections/Data/"

# # Loop through each model, scenario, and variable
# for model in selected_models:
#     for scenario in scenarios:
#         for variable in variables:
#             output_directory = os.path.join(base_directory, "BiasCorrected", model, scenario, variable)

#             # Merge files for each variable
#             pattern = os.path.join(output_directory, f"{variable}_day_{model}_{scenario}_*.nc")

#             # Get a list of files that match the pattern
#             files = sorted(glob.glob(pattern))

#             # Check if there are files to merge
#             if files:
#                 # Open the files using xarray and concatenate along the 'time' dimension
#                 ds = xr.open_mfdataset(files, combine='by_coords')

#                 # Define the bounding box for the region of France
#                 lat_min, lat_max = 41.303, 51.124  # Approximate latitude range for France
#                 lon_min, lon_max = -5.559, 9.561   # Approximate longitude range for France

#                 # Slice the dataset for the region of France
#                 ds_region = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

#                 # Set the CRS to EPSG:4326
#                 ds_region.rio.write_crs("EPSG:4326", inplace=True)

#                 # Reproject to EPSG:4326
#                 ds_region = ds_region.rio.reproject("EPSG:4326", transform=ds_region.rio.transform(recalc=True))

#                 # Print information about the reprojected dataset
#                 print(ds_region)

#                 # Save the result as a new NetCDF file
#                 output_file = os.path.join(output_directory, f"combined_dataFR_{model}_{scenario}_{variable}_reprojected.nc")
#                 ds_region.to_netcdf(output_file)
#             else:
#                 print(f"No files found for {variable}, {model}, {scenario}")





# import xarray as xr
# import os
# import glob

# base_directory = "/media/chidesiv/DATA1/Bias_corrected_GCMs/Data/"

# # Function to merge NetCDF files in a folder
# def merge_netcdf_files(folder_path, output_file):
    

#     pattern = os.path.join(folder_path, "*.nc")
#     print("Pattern:", pattern)
#     files = sorted(glob.glob(pattern))
#     print("Found files:", files)
#     # files = sorted(glob.glob(pattern))

#     if files:
#         ds = xr.open_mfdataset(files, combine='by_coords')
#         ds.to_netcdf(output_file)
#         print(f"Merged files in {folder_path} to {output_file}")
#     else:
#         print(f"No NetCDF files found in {folder_path}")

# # Replace these with the variables and scenarios you are interested in
# variables = ['pr'] #'pr', 'tas'
# scenarios = ['ssp245']#, 'ssp370', 'ssp585'
# selected_models = ['FGOALS-g3']  # Add other models as needed

# # Loop through each model, scenario, and variable
# for model in selected_models:
#     for scenario in scenarios:
#         for variable in variables:
#             output_directory = os.path.join(base_directory_2, model, scenario, variable)

#             # Merge files for each variable
#             pattern = os.path.join(output_directory, f"{variable}_day_{model}_{scenario}_*.nc")
#             output_file = os.path.join(output_directory, f"combined_data_{model}_{scenario}_{variable}.nc")

#             merge_netcdf_files(output_directory, output_file)












# import xarray as xr
# import os
# import glob

# # Replace these with the variables and scenarios you are interested in
# variables = ['pr', 'tas']
# scenarios = ['ssp245', 'ssp370', 'ssp585']
# selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2']  # Add other models as needed
# base_directory = "/media/chidesiv/DATA2/Climate_projections/Data/"

# # Loop through each model, scenario, and variable
# for model in selected_models:
#     for scenario in scenarios:
#         for variable in variables:
#             output_directory = os.path.join(base_directory, "BiasCorrected", model, scenario, variable)

#             # Merge files for each variable
#             pattern = os.path.join(output_directory, f"{variable}_day_{model}_{scenario}_*.nc")

#             # Get a list of files that match the pattern
#             files = sorted(glob.glob(pattern))

#             # Check if there are files to merge
#             if files:
#                 # Open the files using xarray and concatenate along the 'time' dimension
#                 ds = xr.open_mfdataset(files, combine='by_coords')

#                 # Define the bounding box for the region of France
#                 lat_min, lat_max = 41.303, 51.124  # Approximate latitude range for France
#                 lon_min, lon_max = -5.559, 9.561   # Approximate longitude range for France

#                 # Slice the dataset for the region of France
#                 ds_region = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

#                 # Print information about the sliced dataset
#                 print(ds_region)

#                 # Save the result as a new NetCDF file
#                 output_file = os.path.join(output_directory, f"combined_dataFR_{model}_{scenario}_{variable}.nc")
#                 ds_region.to_netcdf(output_file)
#             else:
#                 print(f"No files found for {variable}, {model}, {scenario}")




# import xarray as xr
# import os

# # Replace these with the variables and scenarios you are interested in
# variables = ['pr', 'tas']
# scenarios = ['ssp245', 'ssp370', 'ssp585']
# selected_models = ['ACCESS-ESM1-5']  # Add other models as needed
# base_directory = "/media/chidesiv/DATA2/Climate_projections/Data/"

# # Create a list to store individual datasets
# datasets = []

# # Loop through each model, scenario, and variable
# for model in selected_models:
#     for scenario in scenarios:
#         for variable in variables:
#             output_directory = os.path.join(base_directory, "BiasCorrected", model, scenario, variable)

#             # Merge files for each variable
#             pattern = os.path.join(output_directory, f"{variable}_day_{model}_{scenario}_*.nc")

#             # Get a list of files that match the pattern
#             files = sorted(glob.glob(pattern))

#             # Check if there are files to merge
#             if files:
#                 # Open the files using xarray and concatenate along the 'time' dimension
#                 ds = xr.open_mfdataset(files, combine='by_coords')
#                 datasets.append(ds)
#             else:
#                 print(f"No files found for {variable}, {model}, {scenario}")

# # Concatenate the datasets along the 'time' dimension
# combined_dataset = xr.concat(datasets, dim='time')

# # Define the bounding box for the region of France
# lat_min, lat_max = 41.303, 51.124  # Approximate latitude range for France
# lon_min, lon_max = -5.559, 9.561   # Approximate longitude range for France

# # Slice the dataset for the region of France
# france_dataset = combined_dataset.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

# # Print the sliced dataset information
# print(france_dataset)

# # Optionally, you can save the sliced dataset to a new NetCDF file
# france_dataset.to_netcdf('/path/to/sliced_france_dataset.nc')






# # import xarray as xr
# import os

# # Specify the directory containing your NetCDF files
# nc_files_directory = '/path/to/your/netcdf/files'

# # Create a list to store individual datasets
# datasets = []

# # Loop through each year from 2015 to 2100
# for year in range(2015, 2101):
#     # Construct the file path for the current year
#     file_path = os.path.join(nc_files_directory, f'data_{year}.nc')

#     # Check if the file exists before trying to open it
#     if os.path.exists(file_path):
#         # Open the NetCDF file using xarray
#         ds = xr.open_dataset(file_path)

#         # Append the dataset to the list
#         datasets.append(ds)
#     else:
#         print(f"File not found for year {year}")

# # Concatenate the datasets along the 'time' dimension
# combined_dataset = xr.concat(datasets, dim='time')

# # Define the bounding box for the region of France
# lat_min, lat_max = 41.303, 51.124  # Approximate latitude range for France
# lon_min, lon_max = -5.559, 9.561   # Approximate longitude range for France

# # Slice the dataset for the region of France
# france_dataset = combined_dataset.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

# # Print the sliced dataset information
# print(france_dataset)

# # Optionally, you can save the sliced dataset to a new NetCDF file
# france_dataset.to_netcdf('/path/to/sliced_france_dataset.nc')
