#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:06:12 2024

@author: chidesiv
"""




import xarray as xr
import os
import glob
import psutil
from multiprocessing import Pool

# Function to calculate chunk size based on desired max memory usage
def calculate_chunk_size(max_memory_usage_gb, file_size_gb):
    # Calculate the number of chunks needed to fit within the max memory limit
    num_chunks = int(file_size_gb / max_memory_usage_gb) + 1
    # Calculate chunk size
    chunk_size = int(file_size_gb / num_chunks)
    return {'time': chunk_size}

# Function to process model, scenario, and variable
def process_model_scenario_variable(model_scenario_variable):
    model, scenario, variable = model_scenario_variable
    output_directory = os.path.join(base_directory_2, variable)
    os.makedirs(output_directory, exist_ok=True)
    pattern = os.path.join(base_directory, model, scenario, variable, "*.nc")
    files_to_merge = glob.glob(pattern)
    if files_to_merge:
        # Determine file size to estimate chunk size
        total_file_size_gb = sum(os.path.getsize(file) for file in files_to_merge) / (1024**3)  # Convert bytes to GB
        chunk_size = calculate_chunk_size(max_memory_usage_gb, total_file_size_gb)
        with xr.open_mfdataset(files_to_merge, combine='by_coords', chunks=chunk_size) as ds:
            output_file = os.path.join(output_directory, f"combined_data_{model}_{scenario}_{variable}.nc")
            ds.to_netcdf(output_file)
            print(f"Combined file created for {model}, {scenario}, {variable}")
    else:
        print(f"No files found for {model}, {scenario}, {variable}")

if __name__ == "__main__":
    # Define max memory usage in GB
    max_memory_usage_gb = 10
    
    # Get available RAM size
    available_ram_gb = psutil.virtual_memory().total / (1024**3)  # Convert bytes to GB
    print(f"Available RAM: {available_ram_gb} GB")

    # Adjust number of processes based on available RAM
    num_processes = min(psutil.cpu_count(logical=False), int(available_ram_gb / max_memory_usage_gb))
    print(f"Number of processes: {num_processes}")

    # Your existing code
    base_directory = "/media/chidesiv/DATA1/Bias_corrected_GCMs/Data/"
    base_directory_2 = "/media/chidesiv/DATA2/Climate_projections/Combined/Combined_netcdf/"
    selected_models = ['NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1'] #'FGOALS-g3',
      
#, 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0'
                                    #'GFDL-ESM4','CNRM-CM6-1','EC-Earth3', 'ACCESS-ESM1-5','CMCC-ESM2',  'IPSL-CM6A-LR','CanESM5','GISS-E2-1-G','INM-CM5-0',
    variables = ['rsds', 'sfcWind']
    scenarios = ['ssp245', 'ssp370', 'ssp585']
    
    model_scenario_variables = [(model, scenario, variable) 
                                for model in selected_models
                                for scenario in scenarios
                                for variable in variables]

    with Pool(num_processes) as pool:
        pool.map(process_model_scenario_variable, model_scenario_variables)
