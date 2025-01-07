#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:15:16 2024

@author: chidesiv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# # shape = gpd.read_file("/media/chidesiv/DATA2/Phase2/QGIS/classify/final/1976_final_classes.shp")

# # extract the coordinates
# coords = shape[shape['Class'] == GWL_type].geometry.apply(lambda x: x.coords[:]).apply(pd.Series)


# coords['code_bss'] = shape['code_bss']
# coords['Class'] = shape['Class']
# # rename the columns
# coords.columns = ['x','code_bss','Class']


# # convert the DataFrame to two columns
# coords[['long','lat']] = pd.DataFrame(coords.x.tolist(), index=coords.index)

# # drop the original column
# coords.drop(columns=['x'], inplace=True)




# from pyproj import Transformer
# transformer = Transformer.from_crs( "EPSG:2154","EPSG:4326")
# x, y = coords['lat'], coords['long']


selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'GISS-E2-1-G', 'FGOALS-g3', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1']

scenarios = ['ssp245', 'ssp370', 'ssp585']

Classes=['annual','inertial','mixed']



selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'GISS-E2-1-G', 'FGOALS-g3', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1']
scenarios = ['ssp245', 'ssp370', 'ssp585']

# Assuming you have a GeoDataFrame `gdf` with geometry and 'code_bss' matching those in results_df
# Load your spatial data (replace 'path_to_your_shapefile.shp' with your actual shapefile path)
# gdf = gpd.read_file('path_to_your_shapefile.shp')

# for Class in Classes:  # Replace with your actual classes if they are different
#     results_csv_path = os.path.join(base_directory, f'mann_kendall_test_results{Class}.csv')
#     results_df = pd.read_csv(results_csv_path)

#     for model in selected_models:
#         fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
#         fig.suptitle(f'Mann-Kendall Test Results for {model} - {Class}', fontsize=16)
        
#         for idx, scenario in enumerate(scenarios):
#             ax = axs[idx]
#             ax.set_title(scenario)
            
#             # Merge the results with the GeoDataFrame
#             scenario_df = results_df[(results_df['Model'] == model) & (results_df['Scenario'] == scenario)]
#             merged_gdf = gdf.merge(scenario_df, left_on='code_bss', right_on='Code')
            
#             # Plot points with different notations based on trend and significance
#             for _, row in merged_gdf.iterrows():
#                 if row['Trend'] == 'increasing':
#                     color = 'green'
#                 else:
#                     color = 'red'
                
#                 if row['P_value'] < 0.05:
#                     marker = 'o'  # Significant
#                 else:
#                     marker = 'x'  # Not significant
                
#                 ax.plot(row.geometry.x, row.geometry.y, marker=marker, color=color, markersize=10, label=f"Trend: {row['Trend']}, Sig: {row['P_value'] < 0.05}")
            
#             ax.set_aspect('equal')
#             ax.legend()
        
#         plt.show()
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import geopandas as gpd
import pandas as pd
import os

base_directory = "/media/chidesiv/DATA2/Final_phase/Final_plots/Median_updated_MK_pettitt/map_plots/"

gdf = gpd.read_file("/media/chidesiv/DATA2/Phase2/QGIS/classify/final/1976_final_classes.shp")

# ... (assuming other necessary variables and imports are already defined)

Classes = ['annual', 'inertial', 'mixed']
color_map = {'annual': 'green', 'inertial': 'blue', 'mixed': 'red'}
marker_map = {'increasing': 'o', 'decreasing': 'x', 'no trend': 's'}

# Create legend elements
legend_elements = [
    mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Increasing significant'),
    mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=10, label='Decreasing significant'),
    mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10, label='No trend / Not significant'),
]

# Prepare DataFrame to collect all results
all_results_df = pd.DataFrame()

for Class in Classes:
    results_csv_path = os.path.join(base_directory, f'mann_kendall_test_results{Class}.csv')
    class_results_df = pd.read_csv(results_csv_path)
    all_results_df = pd.concat([all_results_df, class_results_df], ignore_index=True)

# Loop through each model and plot
for model in selected_models:
    fig, axs = plt.subplots(1, len(scenarios), figsize=(24, 8), constrained_layout=True)
    fig.suptitle(f'Mann-Kendall Test Results for {model}', fontsize=16)

    for idx, scenario in enumerate(scenarios):
        ax = axs[idx]
        ax.set_title(scenario)
        filtered_df = all_results_df[(all_results_df['Model'] == model) & (all_results_df['Scenario'] == scenario)]
        merged_gdf = gdf.merge(filtered_df, left_on='code_bss', right_on='Code')

        for _, row in merged_gdf.iterrows():
            # Choose color based on class and marker based on trend and significance
            color = color_map[row['Class_x']]
            marker = marker_map[row['Trend']] if row['P_value'] < 0.05 else 's'
            ax.plot(row.geometry.x, row.geometry.y, marker=marker, color=color, markersize=10, linestyle='None')
        
        ax.set_aspect('equal')
    trend_legend_elements = [
               mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Increasing significant'),
               mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=10, label='Decreasing significant'),
               mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10, label='No trend / Not significant'),
           ]
        
           # Define legend elements for classes
    class_legend_elements = [
               mlines.Line2D([], [], color=color_map[Class], marker='s', linestyle='None', markersize=10, label=Class)
               for Class in Classes
           ]
        
           # Combine both sets of legend elements
    all_legend_elements = trend_legend_elements + class_legend_elements
    
       # Plotting logic remains unchanged
       # ...
    
       # Place the legend below the subplots
    fig.legend(handles=all_legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1),
                  ncol=len(all_legend_elements), fancybox=True, shadow=True)
        
    # Place the legend below the subplots
    # fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fancybox=True, shadow=True)

    # Save the figure to a PNG file
    save_path = os.path.join(base_directory, f'project_{model}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

print("All models have been processed and saved.")



import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import geopandas as gpd
import pandas as pd
import os

base_directory = "/media/chidesiv/DATA2/Final_phase/Final_plots/Median_updated_MK_pettitt/map_plots/"
gdf = gpd.read_file("/media/chidesiv/DATA2/Phase2/QGIS/classify/final/1976_final_classes.shp")

# Assuming other necessary variables and imports are already defined
Classes = ['annual', 'inertial', 'mixed']
selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'GISS-E2-1-G', 'FGOALS-g3', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1']
scenarios = ['ssp245', 'ssp370', 'ssp585']
color_map = {'annual': 'green', 'inertial': 'blue', 'mixed': 'red'}
marker_map = {'increasing': 'o', 'decreasing': 'x', 'no trend': 's'}


# Create legend elements
trend_legend_elements = [
    mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Increasing significant'),
    mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=10, label='Decreasing significant'),
    mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10, label='No trend / Not significant'),
]


# Prepare DataFrame to collect all results
all_results_df = pd.DataFrame()

for Class in Classes:
    results_csv_path = os.path.join(base_directory, f'mann_kendall_test_results{Class}.csv')
    class_results_df = pd.read_csv(results_csv_path)
    all_results_df = pd.concat([all_results_df, class_results_df], ignore_index=True)

# Create one big plot with all models and scenarios (16 models * 3 scenarios)
fig, axs = plt.subplots(len(selected_models), len(scenarios), figsize=(len(scenarios)*5, len(selected_models)*3), squeeze=False)

# Loop through each model and scenario and plot
for i, model in enumerate(selected_models):
    for j, scenario in enumerate(scenarios):
        ax = axs[i, j]
        
        # Set the title for the top row of subplots
        if i == 0:
            ax.set_title(scenario)
        
        # Filter and merge data for the current model and scenario
        filtered_df = all_results_df[(all_results_df['Model'] == model) & (all_results_df['Scenario'] == scenario)]
        merged_gdf = gdf.merge(filtered_df, left_on='code_bss', right_on='Code')

        # Plot each point with the color and marker based on the class, trend, and significance
        for _, row in merged_gdf.iterrows():
            color = color_map[row['Class_x']]
            marker = marker_map[row['Trend']] if row['P_value'] < 0.05 else 's'
            ax.plot(row.geometry.x, row.geometry.y, marker=marker, color=color, markersize=5, linestyle='None')
        
        ax.set_aspect('equal')
        ax.axis('off')

        # Label the y-axis with the model names for the first column
        if j == 0:
            ax.set_ylabel(model, rotation=0, size='large', labelpad=30, verticalalignment='center')

# Adjust subplots layout
plt.subplots_adjust(hspace=0.5, wspace=0.1)

# Add a legend for the entire figure
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_elements), frameon=False)

# Save the figure to a PNG file
save_path = os.path.join(base_directory, 'all_models_scenarios.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print("The comprehensive plot has been saved.")


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
import pandas as pd
import os


base_directory = "/media/chidesiv/DATA2/Final_phase/Final_plots/Median_updated_MK_pettitt/map_plots/"
gdf = gpd.read_file("/media/chidesiv/DATA2/Phase2/QGIS/classify/final/1976_final_classes.shp")

Classes = ['annual', 'inertial', 'mixed']
color_map = {'annual': 'green', 'inertial': 'blue', 'mixed': 'red'}
marker_map = {'increasing': 'o', 'decreasing': 'x', 'no trend': 's'}

# Load the combined results
all_results_df = pd.DataFrame()
for Class in Classes:
    results_csv_path = os.path.join(base_directory, f'mann_kendall_test_results{Class}.csv')
    class_results_df = pd.read_csv(results_csv_path)
    all_results_df = pd.concat([all_results_df, class_results_df], ignore_index=True)

# Set up the figure and GridSpec
n_rows = len(scenarios)
n_cols = len(selected_models)
fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))  # Adjust the size as needed
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.4, hspace=0.4)

# Plot each scenario and model
for i, scenario in enumerate(scenarios):
    for j, model in enumerate(selected_models):
        ax = fig.add_subplot(gs[i, j])
        filtered_df = all_results_df[(all_results_df['Model'] == model) & (all_results_df['Scenario'] == scenario)]
        merged_gdf = gdf.merge(filtered_df, left_on='code_bss', right_on='Code')

        # Plot each point
        for _, row in merged_gdf.iterrows():
            color = color_map[row['Class_x']]
            marker = marker_map[row['Trend']] if row['P_value'] < 0.05 else 's'
            ax.plot(row.geometry.x, row.geometry.y, marker=marker, color=color, markersize=8, linestyle='None')
        
        if i == 0:
            ax.set_title(model)
        if j == 0:
            ax.text(-0.1, 0.5, scenario, va='center', ha='right', fontsize=12, rotation='vertical', transform=ax.transAxes)
        ax.set_aspect('equal')

# Add a legend for the entire figure
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fancybox=True, shadow=True)

# Save the figure to a PNG file
save_path = os.path.join(base_directory, 'all_scenarios_models.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print("Plot with models on columns and scenarios on rows has been saved.")

scenarios = ['ssp245', 'ssp370', 'ssp585']
Classes = ['annual', 'inertial', 'mixed']
selected_models = ['ACCESS-ESM1-5',   'EC-Earth3',  'GISS-E2-1-G',  'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6',  'NorESM2-MM', 'UKESM1-0-LL']
# selected_models = ['CanESM5','CMCC-ESM2', 'CNRM-CM6-1','GFDL-ESM4','FGOALS-g3','MPI-ESM1-2-HR', 'MRI-ESM2-0', 'TaiESM1']
#'CanESM5','CMCC-ESM2', 'CNRM-CM6-1','GFDL-ESM4','FGOALS-g3','MPI-ESM1-2-HR', 'MRI-ESM2-0', 'TaiESM1'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
import pandas as pd
import os
import matplotlib.lines as mlines
import contextily as ctx

base_directory = "/media/chidesiv/DATA2/Final_phase/Final_plots/Median_updated_MK_pettitt/map_plots/final_map_16model_plots/"
gdf = gpd.read_file("/media/chidesiv/DATA2/Phase2/QGIS/classify/final/1976_final_classes.shp")

Classes = ['annual', 'inertial', 'mixed']
trend_color_map = {'increasing': 'blue', 'decreasing': 'red', 'no trend': 'grey'}
class_marker_map = {'annual': 'o', 'inertial': '^', 'mixed': 'D'}  # Triangle, right, and left markers

# Create legend elements
legend_elements = [
    mlines.Line2D([], [], color=trend_color_map['increasing'], marker=class_marker_map['annual'], linestyle='None', markersize=10, label='Annual Increasing'),
    mlines.Line2D([], [], color=trend_color_map['decreasing'], marker=class_marker_map['annual'], linestyle='None', markersize=10, label='Annual Decreasing'),
    mlines.Line2D([], [], color=trend_color_map['increasing'], marker=class_marker_map['inertial'], linestyle='None', markersize=10, label='Inertial Increasing'),
    mlines.Line2D([], [], color=trend_color_map['decreasing'], marker=class_marker_map['inertial'], linestyle='None', markersize=10, label='Inertial Decreasing'),
    mlines.Line2D([], [], color=trend_color_map['increasing'], marker=class_marker_map['mixed'], linestyle='None', markersize=10, label='Mixed Increasing'),
    mlines.Line2D([], [], color=trend_color_map['decreasing'], marker=class_marker_map['mixed'], linestyle='None', markersize=10, label='Mixed Decreasing'),
    mlines.Line2D([], [], color=trend_color_map['no trend'], marker=class_marker_map['annual'], linestyle='None', markersize=10, label='Annual No trend'),
    mlines.Line2D([], [], color=trend_color_map['no trend'], marker=class_marker_map['inertial'], linestyle='None', markersize=10, label='Inertial No trend'),
    mlines.Line2D([], [], color=trend_color_map['no trend'], marker=class_marker_map['mixed'], linestyle='None', markersize=10, label='Mixed No trend'),
]

# Load the combined results
all_results_df = pd.DataFrame()
for Class in Classes:
    results_csv_path = os.path.join(base_directory, f'mann_kendall_test_results{Class}.csv')
    class_results_df = pd.read_csv(results_csv_path)
    all_results_df = pd.concat([all_results_df, class_results_df], ignore_index=True)

# Exclude specific code_bss
all_results_df = all_results_df[all_results_df['Code'] != '08025X0009/P']
# Define the number of rows and columns
n_cols = 3  # Number of scenarios
n_rows = 8  # Number of models
fig = plt.figure(figsize=(n_cols * 9, n_rows * 6))  # Adjust the size as needed
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.05)
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08)
# Plot each scenario for each model
for i, model in enumerate(selected_models):
    for j, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs[i, j])
        ax.set_title(f'{model} - {scenario}' , fontsize=32)
        
        filtered_df = all_results_df[(all_results_df['Model'] == model) & (all_results_df['Scenario'] == scenario)]
        merged_gdf = gdf.merge(filtered_df, left_on='code_bss', right_on='Code')
        
        # Plot OSM background map
        merged_gdf = merged_gdf.to_crs(epsg=3857)
        merged_gdf.plot(ax=ax, marker='o', color='none', edgecolor='none')  # Placeholder for setting extent
        ctx.add_basemap(ax, crs=merged_gdf.crs.to_string())
        # Plot country boundaries
        # gdf.boundary.to_crs(epsg=3857).plot(ax=ax, linewidth=1, edgecolor='black')
        
        # Plot each point
        for _, row in merged_gdf.iterrows():
            color = trend_color_map[row['Trend']] 
            marker = class_marker_map[row['Class_x']]
            ax.plot(row.geometry.x, row.geometry.y, marker=marker, color=color, markersize=10, linestyle='None')
        
        ax.set_aspect('equal')
        ax.axis('off')

# Add a single, common legend to the figure
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.04), ncol=6, fontsize=24)

# Save the figure to a PNG file
save_path = os.path.join(base_directory, 'part1models.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
# plt.close(fig)

print("Plot with models on rows and scenarios on columns has been saved.")

