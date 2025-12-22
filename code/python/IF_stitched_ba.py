"""
This script will be used for BASD for STITCHED data hosted locally
"""

# Importing Needed Libraries
import os  # For navigating os
import shutil  # Running system commands
import socket  # Running on cluster
import sys  # Getting system details
from datetime import datetime  # Manipulate temporal data

import basd  # Bias adjustment and statistical downscaling
import dask  # Setting Dask config
import numpy as np  # Numerical / array functions
import pandas as pd  # Data functions
import IF_utils  # Utility functions script
import xarray as xr  # Reading and manipulating NetCDF data
from dask.distributed import (Client, LocalCluster)  # Using Dask in parallel

# Importing Common Paths/Data/Functions
from IF_setup import IF_PATH, experiment_name, landmask, grid_cell_area


# Global paths and file names 
temp_intermediate_dir = None
output_ba_path = None
output_day_ba_file_name = None
output_mon_ba_file_name = None
output_basd_path = None
output_day_basd_file_name = None
output_mon_basd_file_name = None
input_ref_data_path = None
input_sim_data_path = None

# Chunk sizes (constants to be set)
time_chunk = None
lat_chunk = None
lon_chunk = None

# Function to manage steps for running bias adjustment and downscaling using data accessed from Pangeo.
def ba_stitches(run_object, run_name):
    """
    Function to manage steps for running bias adjustment and downscaling using data from STITCHES saved locally.
    """
        
    # 1. Name output files and paths
    set_names(run_object)

    # 2. Try to make directories if they don't already exist
    create_directories()

    # 4. Get and extract parameters
    params = IF_utils.get_parameters(run_object, os.path.join(IF_PATH, "input", run_name))

    # 5. Read encoding settings
    encoding, reset_chunksizes = IF_utils.get_encoding(os.path.join(IF_PATH, "input", run_name))

    # 6. Read attributes
    variable_attributes, global_monthly_attributes, global_daily_attributes = IF_utils.get_attributes(run_object.Variable, os.path.join(IF_PATH, "input", run_name))

    # 7. Read Dask settings
    global time_chunk, lat_chunk, lon_chunk
    time_chunk, lat_chunk, lon_chunk, dask_temp_directory = IF_utils.get_chunk_sizes(os.path.join(IF_PATH, "input", run_name))

    # 8. Get Data
    # Load in data over the given periods
    obs_reference_data, sim_reference_data, sim_application_data = load_ba_data(run_object)

    # Reset Chunk sizes
    if reset_chunksizes:
        encoding['chunksizes'] = IF_utils.reset_chunk_sizes(encoding['chunksizes'], sim_application_data.dims)

    # Use global path/file names
    global temp_intermediate_dir, output_ba_path, output_basd_path
    global output_day_ba_file_name, output_mon_ba_file_name, output_day_basd_file_name, output_mon_basd_file_name
    global input_ref_data_path, input_sim_data_path
    
    print(obs_reference_data)
    print(sim_reference_data)
    print(sim_application_data)
    
    # 9. Run Bias Adjustment
    # Initializing Bias Adjustment
    ba = basd.init_bias_adjustment(
        obs_reference_data, sim_reference_data, sim_application_data,
        run_object.Variable, params,
        lat_chunk_size=lat_chunk, lon_chunk_size=lon_chunk,
        temp_path=temp_intermediate_dir, periodic=True
    )

    print("Bias Adjustment Initialization Completed")
    # print(ba)
    # client.close()
    # cluster.close()
        
    # Do / don't save monthly data
    if ~run_object.monthly:
        output_mon_ba_file_name = None
        output_mon_basd_file_name = None
    
    print(obs_reference_data.dims)
    print(sim_reference_data.dims)
    print(sim_application_data.dims)
    
    # Perform adjustment and save at daily resolution
    basd.adjust_bias(
        init_output = ba, output_dir = output_ba_path,
        day_file = output_day_ba_file_name, month_file = output_mon_ba_file_name,
        clear_temp = True, encoding={run_object.Variable: encoding},
        ba_attrs = global_daily_attributes, ba_attrs_mon = global_monthly_attributes, 
        variable_attrs = variable_attributes
    )

    print("Bias Adjustment Completed")
    # client.close()
    # cluster.close()
    
    # Close Bias Adjustment Data
    obs_reference_data.close()
    sim_reference_data.close()
    sim_application_data.close()
    # Clear temp directories
    try:
        shutil.rmtree(temp_intermediate_dir)
    except OSError as e:
        print("Warning: %s : %s" % (temp_intermediate_dir, e.strerror))

# Load in datasets and trims to reference and application periods, and drops extra variables in the dataset
def load_ba_data(run_object):
    """
    Function that loads in datasets and trims to reference and application periods, and drops extra variables in the dataset
    """
    
    # File name patterns
    sim_data_pattern = f'stitched_{run_object.ESM}_{run_object.Variable}_{run_object.Scenario}~~1.nc'
    obs_reference_data_pattern = f'{run_object.Variable}_*.nc'

    print(input_sim_data_path, sim_data_pattern)

    # Open data
    sim_data_path = os.path.join(input_sim_data_path, sim_data_pattern)
    sim_data = xr.open_mfdataset(sim_data_path, chunks={'time': time_chunk})
    obs_reference_data = xr.open_mfdataset(os.path.join(input_ref_data_path, obs_reference_data_pattern), chunks={'time': time_chunk})


    print("Time dimension of simulated data:", sim_data.time)

    # # 1. Convert to datetime 
    # datetimeindex = sim_data.indexes['time'].to_datetimeindex()
    # sim_data['time'] = datetimeindex
    
    # # 2. Convert cftime to pandas datetime safely via string
    # datetimeindex = pd.to_datetime([str(t) for t in sim_data.time.values])
    # sim_data = sim_data.assign_coords(time=datetimeindex)
    
    # 3. Perform calendar correction & convert to datetime64 format        
    sim_data = sim_data.convert_calendar(calendar = 'gregorian', align_on = 'date', missing = np.nan)
    
    # Split simulation data into target and application periods
    sim_application_data = sim_data
    sim_reference_data = sim_data

    # Get application and target periods
    application_start_year, application_end_year = str.split(run_object.application_period, '-')
    target_start_year, target_end_year = str.split(run_object.target_period, '-')

    # Sub-setting desired time
    obs_reference_data = obs_reference_data.sel(time = slice(f'{target_start_year}', f'{target_end_year}')).copy()
    sim_application_data = sim_data.sel(time = slice(f'{application_start_year}', f'{application_end_year}')).copy()
    sim_reference_data = sim_data.sel(time = slice(f'{target_start_year}', f'{target_end_year}')).copy()
 

    # DEALING WITH LEAP YEARS IN ESM #################################
        
    # Extract time variables
    time_sim_ref = sim_reference_data.time
    time_isimip = obs_reference_data.time

    # Convert to pandas datetime for easy comparison
    # dates_sim_ref = pd.to_datetime([str(t) for t in time_sim_ref.values]) # For the ESMs 
    # dates_isimip = pd.to_datetime([str(t) for t in time_isimip.values])
    dates_sim_ref = pd.to_datetime(time_sim_ref.values)
    dates_isimip = pd.to_datetime(time_isimip.values)
    
    # Identify missing February 29th dates
    leap_days = [pd.Timestamp(year, 2, 29) for year in np.unique(dates_isimip.year) if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]
    missing_leap_days = [day for day in leap_days if day not in dates_sim_ref]
    
    print("Missing leap days:", missing_leap_days)
    
    # Create copies of February 28 data for each missing February 29
    if missing_leap_days:
        new_entries = []
        
        for missing_day in missing_leap_days:
            feb_28 = missing_day - pd.Timedelta(days=1)  # Get Feb 28 of the same year
            
            if feb_28 in dates_sim_ref:
                # Extract the Feb 28 data
                feb_28_data = sim_reference_data.sel(time=feb_28, method='nearest', tolerance='1D')
                
                # Create a new DataArray for Feb 29 with the same values as Feb 28
                feb_29_data = feb_28_data.assign_coords(time=[missing_day])
                
                # Store the new entry
                new_entries.append(feb_29_data)
        
        # Combine original data with the new leap day entries
        if new_entries:
            new_entries = xr.concat(new_entries, dim="time")
            sim_reference_data = xr.concat([sim_reference_data, new_entries], dim="time").sortby("time")
    
    print("Updated sim_reference_data now includes leap days.")
    
  
    ###########################################################################

    # Close full time series simulation data
    sim_data.close()

    # Drop unwanted vars
    obs_reference_data = obs_reference_data.drop([x for x in list(obs_reference_data.coords) if x not in ['time', 'lat', 'lon']])
    sim_reference_data = sim_reference_data.drop([x for x in list(sim_reference_data.coords) if x not in ['time', 'lat', 'lon']])
    sim_application_data = sim_application_data.drop([x for x in list(sim_application_data.coords) if x not in ['time', 'lat', 'lon']])
    obs_reference_data = obs_reference_data.drop_vars([x for x in list(obs_reference_data.keys()) if x != run_object.Variable])
    sim_reference_data = sim_reference_data.drop_vars([x for x in list(sim_reference_data.keys()) if x != run_object.Variable])
    sim_application_data = sim_application_data.drop_vars([x for x in list(sim_application_data.keys()) if x != run_object.Variable])

    # Return
    return obs_reference_data, sim_reference_data, sim_application_data


# Function for setting path and file names based on run details
def set_names(run_object):
    """
    Function for setting paths and file names based on the run details
    """
    # We want to edit global variables
    global temp_intermediate_dir, output_ba_path, output_basd_path
    global output_day_ba_file_name, output_mon_ba_file_name, output_day_basd_file_name, output_mon_basd_file_name
    global input_ref_data_path, input_sim_data_path

    # Temporary intermediate results directory
    temp_intermediate_dir = os.path.join(run_object.Output_Location, run_object.Reference_Dataset, 
                                         run_object.ESM, run_object.Scenario, 
                                         f'{run_object.Variable}_STITCHES_temp_intermediate')

    # Full output path for bias adjusted data
    output_ba_path = os.path.join(run_object.Output_Location, run_object.Reference_Dataset,
                                  run_object.ESM, run_object.Scenario, 'ba')
    
    # Start and End years
    start, end = str.split(run_object.application_period, '-')

    # Output file name for daily and monthly bias adjusted data
    output_day_ba_file_name = f'{run_object.ESM}_STITCHES_{run_object.Reference_Dataset}_{run_object.Scenario}_{run_object.Variable}_global_daily_{start}_{end}.nc'
    output_mon_ba_file_name = f'{run_object.ESM}_STITCHES_{run_object.Reference_Dataset}_{run_object.Scenario}_{run_object.Variable}_global_monthly_{start}_{end}.nc'
    
    # Full output path for downscaled data
    output_basd_path = os.path.join(run_object.Output_Location, run_object.Reference_Dataset,
                                    run_object.ESM, run_object.Scenario, 'basd')
    
    # Output file name for daily and monthly downscaled data
    output_day_basd_file_name = f'{run_object.ESM}_STITCHES_{run_object.Reference_Dataset}_{run_object.Scenario}_{run_object.Variable}_global_daily_{start}_{end}.nc'
    output_mon_basd_file_name = f'{run_object.ESM}_STITCHES_{run_object.Reference_Dataset}_{run_object.Scenario}_{run_object.Variable}_global_monthly_{start}_{end}.nc'
    
    # Input location for observational reference dataset
    # input_ref_data_path = os.path.join(run_object.Reference_Input_Location, run_object.Variable)
    input_ref_data_path = os.path.join(run_object.Reference_Input_Location)
    
    # Input location for simulated datasets
    input_sim_data_path = run_object.ESM_Input_Location


# Function that creates new directories
def create_directories():
    """
    Try to create any new directories. If already exist, do nothing.
    """
    try:
        os.makedirs(temp_intermediate_dir)
    except FileExistsError:
        pass
    try:
        os.makedirs(output_ba_path)
    except FileExistsError:
        pass
    try:
        os.makedirs(output_basd_path)
    except FileExistsError:
        pass


# Main function in case someone tries to run this as a script
if __name__ == '__main__':
    # pangeo.py executed as script
    print(f'downloaded.py not intended to be run as a script')
