#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:21:06 2024

@author: chidesiv
"""
import geopandas as gpd
import pandas as pd

# Assuming your shapefile reading is correct, but let's correct the coordinate extraction
shape = gpd.read_file("/media/chidesiv/DATA2/Phase2/QGIS/classify/final/1976_final_classes.shp")

# This extracts a list of tuples for coordinates; we expect point geometries here
coords = shape.geometry.apply(lambda geom: geom.xy).apply(pd.Series)
coords['code_bss'] = shape['code_bss']
coords['Class'] = shape['Class']

# Since geom.xy returns a tuple of arrays (x, y), we extract them like this
coords['long'] = coords[0].apply(lambda x: x[0])
coords['lat'] = coords[1].apply(lambda x: x[0])

# Drop the original columns now that we've extracted long and lat
coords = coords.drop([0, 1], axis=1)

import os
import pandas as pd
import xarray as xr
import geopandas as gpd

def extract_and_save_data(base_directory, models, scenarios, variables, coords, period_start, period_end):
    print(f"Starting data extraction and saving process...")
    
    for code in coords['code_bss'].unique():
        print(f"Processing code: {code}")
        coord = coords.loc[coords['code_bss'] == code].iloc[0]
        lat, lon = coord['lat'], coord['long']
        
        for model in models:
            for scenario in scenarios:
                print(f"Processing model: {model}, Scenario: {scenario}")
                combined_df = pd.DataFrame()
                
                for variable in variables:
                    file_path = os.path.join(base_directory, variable, f"combined_data_{model}_{scenario}_{variable}.nc")
                    print(f"Checking file: {file_path}")
                    if not os.path.exists(file_path):
                        print(f"File does not exist: {file_path}")
                        continue
                    
                    ds = xr.open_dataset(file_path)
                    ds_time_sliced = ds.sel(time=slice(period_start, period_end))
                    # Then, use the nearest method for lat and lon separately
                    ds_sel = ds_time_sliced.sel(lat=lat, lon=lon, method='nearest')
                   
                    df = ds_sel.to_dataframe().reset_index()
                    df = df[['time', variable]]
                    df.rename(columns={'time': 'Date', variable: variable}, inplace=True)
                    
                    if combined_df.empty:
                        combined_df = df
                    else:
                        combined_df = pd.merge(combined_df, df, on='Date', how='outer')
                
                if not combined_df.empty:
                    combined_df['lat'] = lat
                    combined_df['long'] = lon
                    combined_df['code'] = code
                    
                    save_dir = os.path.join(base_directory, model, scenario)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "{}.csv".format(code.replace('/','_')))
                    print(f"Saving to: {save_path}")
                    combined_df.to_csv(save_path, index=False)
                else:
                    print(f"No data found for code {code} in model {model}, scenario {scenario}. Skipping...")




# Example call to the function
base_directory = "/media/chidesiv/DATA2/Climate_projections/Combined/"
models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'GISS-E2-1-G', 'FGOALS-g3', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1']
scenarios = ['ssp245', 'ssp370', 'ssp585']
variables = ['pr', 'tas']

period_start = '2015-01-01'
period_end = '2100-12-30'

extract_and_save_data(base_directory, models, scenarios, variables, coords, period_start, period_end)
