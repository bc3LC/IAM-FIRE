import os

import basd
import numpy as np
import pandas as pd
import xarray as xr


# Get relevant parameters object
def get_parameters(run_object, input_path):
    """
    Function for reading parameters for the relevant variable and returning a basd.Parameters object
    """
    
    # Read in input parameter data
    param_data = pd.read_csv(os.path.join(input_path, 'variable_parameters.csv'))

    # Get parameter data for relevant variable as dictionary
    param_dict = param_data[param_data.variable == run_object.Variable].dropna(axis=1).to_dict(orient='records')[0]
    del param_dict['variable'] # Parameters object doesn't take 'variable', was only needed to filter data

    # Make n_iterations an integer
    if 'n_iterations' in param_dict:
        param_dict['n_iterations'] = int(param_dict['n_iterations'])
    # Make halfwin_ubc an integer
    if 'halfwin_ubc' in param_dict:
        param_dict['halfwin_ubc'] = int(param_dict['halfwin_ubc'])

    # Create basd.Parameters object
    param_obj = basd.Parameters(**param_dict)
    
    return param_obj


# Function for getting the sizes of chunks to be using while performing dask operations on data
def get_chunk_sizes(input_path):
    """
    Function for getting the sizes of chunks to be using while performing dask operations on data
    """
    
    # Read in input parameter data
    dask_params = pd.read_csv(os.path.join(input_path, 'dask_parameters.csv'))
    
    return dask_params['time_chunk_size'][0], dask_params['lat_chunk_size'][0], dask_params['lon_chunk_size'][0], dask_params['dask_temp_directory'][0]


# Get attributes for given variable, and global
def get_attributes(variable, input_path):
    """
    Function for reading in variable and global attributes from input file
    """

    # Read in input parameter data
    attribute_data = pd.read_csv(os.path.join(input_path, 'attributes.csv'))

    # Get attribute data as dictionaries
    variable_attribute_dict = attribute_data[attribute_data.variable == variable].dropna(axis=1).to_dict(orient='records')[0]
    try:
        global_monthly_attribute_dict = attribute_data[attribute_data.variable == 'global_monthly'].dropna(axis=1).to_dict(orient='records')[0]
        global_daily_attribute_dict = attribute_data[attribute_data.variable == 'global_daily'].dropna(axis=1).to_dict(orient='records')[0]
    except:
        global_monthly_attribute_dict, global_daily_attribute_dict = {}
    
    # Return
    return variable_attribute_dict, global_monthly_attribute_dict, global_daily_attribute_dict


# Setting chunks sizes to dimension sizes
def reset_chunk_sizes(chunk_size_tuple, dims_dict):
    """
    Function for setting chunks sizes to dimension sizes
    """
    time_chunk = chunk_size_tuple[0]
    lat_chunk = chunk_size_tuple[1]
    lon_chunk = chunk_size_tuple[2]

    # If want time dimension to have full chunk size
    if chunk_size_tuple[0] == 'max':
        time_chunk = dims_dict['time']
    # If want lat dimension to have full chunk size
    if chunk_size_tuple[1] == 'max':
        lat_chunk = dims_dict['lat']
    # If want lon dimension to have full chunk size
    if chunk_size_tuple[2] == 'max':
        lon_chunk = dims_dict['lon']

    return (time_chunk, lat_chunk, lon_chunk)


# Function for loading in data for statistical downscaling routine, including trimming to respective periods
def load_sd_data(run_object, input_ref_dir, time_chunk_size, output_ba_path, output_day_ba_file_name):
    """
    Function for loading in data for statistical downscaling routine, including trimming to respective periods
    """
    # Load in data for downscaling
    obs_reference_data = xr.open_mfdataset(os.path.join(input_ref_dir, f'{run_object.Variable}_*.nc'), chunks={'time': time_chunk_size})
    sim_application_data = xr.open_mfdataset(os.path.join(output_ba_path, output_day_ba_file_name), chunks={'time': time_chunk_size})

    # Get application and target periods
    application_start_year, application_end_year = str.split(run_object.application_period, '-')
    target_start_year, target_end_year = str.split(run_object.target_period, '-')

    # Subsetting desired time
    obs_reference_data = obs_reference_data.sel(time = slice(f'{target_start_year}', f'{target_end_year}'))
    sim_application_data = sim_application_data.sel(time = slice(f'{application_start_year}', f'{application_end_year}'))
    
    # Drop unwanted vars
    obs_reference_data = obs_reference_data.drop([x for x in list(obs_reference_data.coords) if x not in ['time', 'lat', 'lon']])
    sim_application_data = sim_application_data.drop([x for x in list(sim_application_data.coords) if x not in ['time', 'lat', 'lon']])

    return obs_reference_data, sim_application_data


# Function for reading in encoding parameters to be passed to xarray.to_netcdf()
def get_encoding(input_path):
    """
    Function for reading in encoding parameters to be passed to xarray.to_netcdf()
    """

    # Read in encoding input file
    encoding_data = pd.read_csv(os.path.join(input_path, 'encoding.csv'))

    # Turn into dictionary
    encoding_data_dict = encoding_data.dropna(axis=1).to_dict(orient='records')[0]

    # Requires all chunksizes to be filled out, or will use defaults
    # If all given, set chunksizes to the tuple
    # If one or more set to "max", set reset_encoding_chunks = True, meaning
    # that the chunksizes will later be set to whatever the data dimensions are
    reset_encoding_chunks = False
    if np.any( [pd.isna(encoding_data['time_chunk']), pd.isna(encoding_data['lat_chunk']), pd.isna(encoding_data['lon_chunk'])] ):
        del encoding_data_dict['time_chunk'], encoding_data_dict['lat_chunk'], encoding_data_dict['lon_chunk']
    else:
        if 'max' in [encoding_data_dict['time_chunk'], encoding_data_dict['lat_chunk'], encoding_data_dict['lon_chunk']]:
            reset_encoding_chunks = True
            encoding_data_dict['chunksizes'] = (encoding_data_dict['time_chunk'], encoding_data_dict['lat_chunk'], encoding_data_dict['lon_chunk'])
        del encoding_data_dict['time_chunk'], encoding_data_dict['lat_chunk'], encoding_data_dict['lon_chunk']
    
    return encoding_data_dict, reset_encoding_chunks
