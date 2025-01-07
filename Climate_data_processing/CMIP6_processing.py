#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:21:06 2024

@author: chidesiv
"""



import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

def convert_times(time_series):
    try:
        # Attempt to convert cftime Datetime objects to pandas timestamps
        times = pd.to_datetime(time_series.indexes['time'].to_datetimeindex())
    except Exception as e:
        print(f"Error converting times: {e}")
        times = time_series.indexes['time']  # Fallback to the original times if conversion fails
    return times

# Assuming GWL_type is defined; if not, specify it here
GWL_type= "mixed"  # Replace 'Your_GWL_Class' with the actual class you're interested in

shapefile_path = "/media/chidesiv/DATA2/Phase2/QGIS/classify/final/1976_final_classes.shp"
shape = gpd.read_file(shapefile_path)

# Extract the coordinates
coords = shape[shape['Class'] == GWL_type].geometry.apply(lambda x: x.coords[:]).apply(pd.Series)
coords['code_bss'] = shape['code_bss']
coords['Class'] = shape['Class']
# Rename the columns
coords.columns = ['x', 'code_bss', 'Class']

# Convert the DataFrame to include longitude and latitude columns
coords[['long', 'lat']] = pd.DataFrame(coords.x.tolist(), index=coords.index)
# Drop the original column
coords.drop(columns=['x'], inplace=True)

base_directory = "/media/chidesiv/DATA2/Climate_projections/Combined/" #/media/chidesiv/DATA1/Bias_corrected_GCMs/Data/
selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3',
                   'GFDL-ESM4', 'GISS-E2-1-G', 'FGOALS-g3', 'INM-CM5-0', 'IPSL-CM6A-LR',
                   'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1']

variables = ['pr', 'tas']

scenarios = ['ssp245', 'ssp370', 'ssp585']

for index, row in coords.iterrows():
    lat = row['lat']
    lon = row['long']
    for model in selected_models:
        for scenario in scenarios:
            for variable in variables:
                input_file = os.path.join(base_directory, variable, f"combined_data_{model}_{scenario}_{variable}.nc")
                if os.path.exists(input_file):
                    ds = xr.open_dataset(input_file)
                #     if 'lat' in ds.coords and 'lon' in ds.coords:
                #         lat_resolution = abs(ds['lat'].diff('lat').mean())
                #         lon_resolution = abs(ds['lon'].diff('lon').mean())
                        
                #         print(f"Model: {model}, Scenario: {scenario}, Variable: {variable}")
                #         print(f"Latitude resolution: {lat_resolution.values} degrees")
                #         print(f"Longitude resolution: {lon_resolution.values} degrees")
                #     else:
                #         print(f"Latitude or longitude dimensions not found in {input_file}")
                    
                # # Close the dataset after processing to free up resources
                #     ds.close()
                # else:
                #     print(f"File does not exist: {input_file}")
                        
                    # Extract time series for the location
                    time_series = ds.sel(lat=lat, lon=lon, method='nearest')[variable]
                    
                    print(time_series)
                    
                    times = convert_times(time_series)
                    time_series['time'] = times
                    
                    # Plotting the time series
                    plt.figure(figsize=(10, 6))
                    time_series.plot()
                    plt.title(f'Time Series for {model}, {scenario}, {variable}\n at location ({lat}, {lon})')
                    plt.xlabel('Time')
                    plt.ylabel(variable)
                    plt.grid(True)
                    plt.show()
                else:
                    print(f"File not found: {input_file}")
