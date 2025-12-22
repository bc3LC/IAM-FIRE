# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:43:30 2025

@author: theo.rouhette
"""
# Importing Needed Libraries
import os  # For navigating os
import pandas as pd  # Data functions
import xarray as xr  # Reading and manipulating NetCDF data
import argparse
import demeter

# Importing Common Paths/Data/Functions
from IF_setup import IF_PATH, experiment_name, landmask, grid_cell_area

# PATHS
config_dir = os.path.join(IF_PATH, 'demeter_setup/config_files')
raw_outputs_dir = os.path.join(IF_PATH, 'demeter_setup/outputs')
proc_outputs_dir = os.path.join(IF_PATH, "output/land_use")
projected_dir = os.path.join(IF_PATH, f"demeter_setup/inputs/projected")

# DEMETER SET UPS
dates_demeter = ["2015", "2021", "2025", "2030", "2035", 
              "2040", "2045", "2050", "2055", "2060", 
              "2065", "2070", "2075", "2080", "2085", 
              "2090", "2095", "2100"]
PFT_list = ['PFT0', 'PFT1', 'PFT2', 'PFT50', 'PFT51', 'PFT52'] # Plant Functional Types to extract from Demeter 

# GCAM YEARS
start_year = 2021
end_year = 2100
# dates_sim = list(range(start_year, end_year + 1, 5))

# INPUTS
basemap = pd.read_csv(os.path.join(IF_PATH, "demeter_setup/inputs/observed/ESA_0p5_deg_Demeter_basemap_2019.csv"))  

def area_checks(basemap, scenario: str):
    
    scenario_harmonized = pd.read_csv(os.path.join(projected_dir, "Scenario_{scenario}_ESA_H_2019.csv")) 
    basemap = basemap.rename(columns={"Latcoord": "lat", "Loncoord": "lon"})
    basemap["Grid_area"] = [grid_cell_area(lat) / 10000 for lat in basemap["lat"]]
    
    land_uses = ["Forest", "Shrubland"] # Note: Only compare areas of consistent land use type
    for land_use in land_uses:    
        # Step 1: Create the land use column based on landclass
        scenario_harmonized[f"{land_use}"] = scenario_harmonized['landclass'].str.contains(f"{land_use}", case=False, na=False)
        scenario_harmonized[f"{land_use}"] = scenario_harmonized[f"{land_use}"].map({True: f"{land_use}", False: 'other'})
        
        # Step 2: Group by 'forest' and sum numeric columns
        df_summarized = scenario_harmonized.groupby(f"{land_use}").sum(numeric_only=True)
        
        # Step 3: Print the value of the 'forest' group in the '2015' column
        print(f"The {land_use} area (Mha) in GCAM projections in column 2021 is", df_summarized.loc[f"{land_use}", '2021']/10)
        print(f"The {land_use} area (Mha) in basemap is ", (basemap["Grid_area"] * (4*basemap[f"{land_use}"])).sum()) 


def demeter_run(scenario: str):
              
    # Launch Demeter
    print(f"Function start: Launching Demeter for scenario {scenario}")
    config_file = os.path.join(config_dir, f"Scenario_{scenario}_ESA_H_2019.ini")  
    demeter.run_model(config_file=config_file, write_outputs=True)
    
def folder_rename(scenario: str):
    import os
    import re
    import shutil
    # List and filter files related to the scenario
    files = [file for file in os.listdir(raw_outputs_dir) if scenario in file]

    # Loop through each file
    for file in files:
        # Full path of the original file
        old_file_path = os.path.join(raw_outputs_dir, file)
        
        # Extract scenario name: keep text before _2025...
        new_name = re.sub(r"_2025.+", "", file)
        new_file_path = os.path.join(raw_outputs_dir, new_name)
        
        # Check if target exists and delete it 
        if os.path.exists(new_file_path):
            if os.path.isdir(new_file_path):
                shutil.rmtree(new_file_path)
                print(f"Deleted existing folder: {new_name}")
            else:
                os.remove(new_file_path)
                print(f"Deleted existing file: {new_name}")

        # Rename safely
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {file} -> {new_name}")

def data_processing_ESA(scenario: str): 
    
    print(f"Processing Demeter outputs for scenario {scenario}")
    
    # Read NetCDF OR Tabular 
    output_dir = os.path.join(raw_outputs_dir, f'{scenario}', 'spatial_landcover_netcdf')
    
    # Create the directory for results 
    scenario_output_dir = os.path.join(proc_outputs_dir, scenario)
    os.makedirs(scenario_output_dir, exist_ok=True)
    
    dates_sim = slice(f"2015-01-31", f"{end_year}-12-31")

    output_scen = []
    # years = ["2020", "2050"]
    for y in dates_demeter: 
        
        # y = "2020"
        output = xr.open_dataset(os.path.join(output_dir, f'_demeter_{scenario}_{y}.nc'), engine="netcdf4")
        output # One time slice, lat 2160, lon 4320 
        output.dims
        
        # Here I only select the variables of PFT that I am interested in 
        # output = output[PFT_list]
        
        # Sum of prot and unprot 
        forest = output['PFT2'] 
        grassland = output['PFT1'] 
        shrubland = output['PFT0'] 
        cropland = output['PFT5']
        # pasture = output['PFT6']

        output['forest'] = forest
        output['grassland'] = grassland
        output['shrubland'] = shrubland
        output['cropland'] = cropland
        # output["pasture"] = pasture
        
        # Select only the sum of the land uses 
        landuse = ['forest', 'grassland', 'shrubland', 'cropland']
        output = output[landuse]
        output = output.transpose('longitude', 'latitude').rename({'longitude': 'lon', 'latitude': 'lat'}).chunk({'lat': 100, 'lon': 100})

        # Create the dimension year 
        if "2015" in y: 
            y1 = int(y) + 5 
        elif "2021" in y:
            y1 = int(y) + 3
        else:
            y1 = int(y) + 4
        times = pd.date_range(f"{y}/01/31",f"{y1}/12/31",freq='YE')
        time_da = xr.DataArray(times, [('time', times)])
        output = output.expand_dims(time=time_da)

        # Save in scenario list with other years 
        output_scen.append(output)
        
    # Combine all years 
    output_all = xr.concat(output_scen, dim='time')
    
    # Select for historic period 
    output_all = output_all.sel(time=dates_sim)
    
    # Transpose
    output_all = output_all.transpose('time', 'lat', 'lon')
    output_all = output_all.reindex(lat=list(reversed(output_all.lat)))

    print(output_all)
    
    # Landmask
    output_all = output_all.where(landmask.mask == 1)
    
    # Check the forest area
    output_df = output_all.to_dataframe().reset_index()
    output_df["Grid_area"] = [grid_cell_area(lat) / 10000 for lat in output_df["lat"]]
    output_df = output_df.loc[output_df["time"] == "2021-12-31"]
    (output_df["Grid_area"] * output_df["forest"]).sum() # 3000
    
    print(f"The forest area of {scenario} in 2021 is", (output_df["Grid_area"] * output_df["forest"]).sum()) 
    print(f"The grassland area of {scenario} in 2021 is", (output_df["Grid_area"] * output_df["grassland"]).sum()) 
    print(f"The cropland area of {scenario} in 2021 is", (output_df["Grid_area"] * output_df["cropland"]).sum()) 
    print(f"The shrubland area of {scenario} in 2021 is", (output_df["Grid_area"] * output_df["shrubland"]).sum()) 
    # print(f"The pasture area of {scenario} in 2015 is", (output_df["Grid_area"] * output_df["pasture"]).sum()) 

    # Save in results folder in generated_demeter 
    output_all.to_netcdf(os.path.join(proc_outputs_dir, f'Processed_DEM_annual_{scenario}_{start_year}-{end_year}_ESA_2019.nc'))

    print(f"Saved Demeter processed outputs for scenario {scenario}")

if __name__ == "__main__":

# Parse the scenario name from temp_gcam ----------------------------------------

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_id', type=int, help='The number of the current task (row of the run_manager_explicit_list.csv file)')
    parser.add_argument('run_name', type=str, help='name of your experiment directory')
    
    parser.add_argument('--warn', action='store_const', dest='warn',
                        const=True, default=False,
                        help='flag to print warnings in log .out file')
    args = parser.parse_args()

    # Task index from SLURM array to run specific scenario
    task_id = args.task_id
    # Name of run directory
    run_name = args.run_name
    # Extract task details
    task_details = pd.read_csv(os.path.join(IF_PATH, f'input/{experiment_name}/temp_global_trajectory.csv')).iloc[task_id]

    # Name the scenario 
    scenario = task_details.scenario
    print(f'Parse completed: Launching Demeter for {scenario}', flush=True)

    # Run area checks 
    area_checks(basemap, scenario)

    # Launch Demeter for the scenario 
    demeter_run(scenario)
    
    # Rename the files
    folder_rename(scenario)
    
    # Process the date
    data_processing_ESA(scenario)






