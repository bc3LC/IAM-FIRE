"""
Script for generated STITCHED datasets
Lots to update here, rough outline
"""

# Import Packages ----------------------------------------
import os
import sys
import importlib.resources as resources
import argparse
import pandas as pd
import numpy as np
import stitches
import stitches.fx_processing as fxp

from IF_setup import IF_PATH, remove_nas


# Define Functions ----------------------------------------

def get_archive():
    """
    Function to get the data archive
    TODO: Add option to specify end_yr_vector somehow
    """
    # # Download data if not already present
    # if not os.path.isfile(pkg_resources.resource_filename('stitches', 'data/matching_archive_staggered.csv')):
    #     stitches.install_pkgdata.install_package_data()

    # # read in the package data of all ESMs-Scenarios-ensemble members avail.
    # path = pkg_resources.resource_filename('stitches', 'data/matching_archive_staggered.csv')
    # data = pd.read_csv(path)
    
    # Download data if not already present
    archive_path = resources.files('stitches') / 'data/matching_archive_staggered.csv'
    
    if not os.path.isfile(archive_path):
        stitches.install_pkgdata.install_package_data()
    
    # read in the package data of all ESMs-Scenarios-ensemble members avail.
    data = pd.read_csv(archive_path)

    # Subset the data to use chunks starting at 2100 and going back in 9 year intervals
    end_yr_vector = np.arange(2100,1800,-9)
    data = stitches.fx_processing.subset_archive(staggered_archive = data, end_yr_vector = end_yr_vector)

    # Return
    return data


def interp(years, values):
    min_year = min(years); max_year = max(years)
    new_years = np.arange(min_year, max_year+1)
    new_values = np.zeros(len(new_years))

    for index, year in enumerate(new_years):
        if np.isin(year, years):
            new_values[index] = values[year == years][0]
        else:
            less_year = max(years[year > years])
            more_year = min(years[year < years])
            less_value = values[np.where(less_year == years)[0][0]]
            more_value = values[np.where(more_year == years)[0][0]]
            p = (year - less_year)/(more_year - less_year)
            new_values[index] = p * more_value + (1-p) * less_value

    return new_years, new_values


def format_data_for_stitches(interped_data, experiment):
    # Variable, model, ensemble, experiment columns
    interped_data['variable'] = 'tas'
    interped_data['model'] = ''
    interped_data['ensemble'] = ''
    interped_data['experiment'] = experiment

    # Convert to tas anomaly
    interped_data.value = interped_data.value - np.mean(interped_data.value[(interped_data.year <= 2014) & (interped_data.year >= 1995)])

    # Sort columns
    formatted_traj = interped_data[['variable', 'experiment', 'ensemble', 'model', 'year', 'value']]
    
    # Return
    return formatted_traj


def get_recipe(target_data, archive_data, variables):
    
    
    # Get recipe (some randomness involved in fit, so try multiple times)
    for i in range(100):
        try:
            target_data['unit'] = "degC change from avg over 1995~2014"
            # print(target_data.unit)
            # print(archive_data.unit)
            stitches_recipe = stitches.make_recipe(target_data, 
                                                   archive_data, 
                                                   tol=0., 
                                                   N_matches=1, 
                                                   res='day', 
                                                   non_tas_variables=[var for var in variables if var != 'tas'])
        except Exception as e:
            print(f"[Attempt {i+1}] Failed to create recipe")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            continue
    
    # Make sure last period has same length in archive and target
    last_period_length = stitches_recipe['target_end_yr'].values[-1] - stitches_recipe['target_start_yr'].values[-1]
    asy = stitches_recipe['archive_start_yr'].values
    asy[-1] = stitches_recipe['archive_end_yr'].values[-1] - last_period_length
    stitches_recipe['archive_start_yr'] = asy.copy()
    
    return stitches_recipe


def generate_stitched(esm, variables, time_series, years, experiment,  output_path, chunk_sizes = 9):
    
    
    # Get full archive data
    data = get_archive()
    # Get archive data for specific model
    model_data = data[(data["model"] == esm) &
                    (data["experiment"].str.contains('ssp'))]

    # Interpolate data
    years, temps = interp(years, time_series)
    interped_data = pd.DataFrame({'year': years, 'value': temps})

    # Format data into STITCHES format
    formatted_data = format_data_for_stitches(interped_data, experiment)

    # Chunk data
    target_chunk = fxp.chunk_ts(formatted_data, n=chunk_sizes)
    target_data = fxp.get_chunk_info(target_chunk)

    # Make Recipe
    stitches_recipe = get_recipe(target_data, model_data, variables)

    print("Make recipe done, entering the gridded stitching function of STITCHES")

    # Make gridded datasets
    outputs = stitches.gridded_stitching(output_path, stitches_recipe)

    return outputs


if __name__ == "__main__":

# Define Constants ----------------------------------------
    # TODO: This stuff needs to be read in from user input

    # Name of the current experiment directory
    task_id = str(sys.argv[1])
    run_name = str(sys.argv[2])
    
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
    inter_files_path = os.path.join(IF_PATH, 'intermediate/', run_name)
    task_details = pd.read_csv(os.path.join(inter_files_path, 'run_manager_explicit_list_stitch.csv')).iloc[task_id]

    # Name the scenario 
    scenario = task_details['Scenario']
    # Name of ESM 
    esm = task_details['ESM']
    # Name of input path
    esm_input_paths = task_details['ESM_Input_Location']
    
    # Input file path
    input_files_path = os.path.join(IF_PATH, 'input/', run_name)

    # Reading the run details to get variables 
    run_manager_df = pd.read_csv(os.path.join(input_files_path, 'run_manager.csv'))

    # Extracting needed infor and formatting the run details
    variables = remove_nas(run_manager_df['Variable'].values)

    # Reading in the tas trajectories data
    trajectories_data = pd.read_csv(os.path.join(input_files_path, 'temp_global_trajectory.csv'))
    trajectories_data = trajectories_data.drop(columns=["variable", "Units"])
    trajectories_data = trajectories_data.melt(id_vars=["scenario"], var_name="year", value_name="value")
    trajectories_data = trajectories_data.pivot(index="year", columns="scenario", values="value").reset_index()
    trajectories_data = trajectories_data[trajectories_data['year'] >= '1980']

    print(f"Trajectories created, entering the STITCHES array for {esm} and {scenario}")

    # Get trajectory data for the given scenario
    time_series_df = trajectories_data[['year', scenario]].dropna()
    tas_time_series = np.array(time_series_df.iloc[:,1].values)
    years = np.array( time_series_df.iloc[:,0].values ).astype(int)
    
    # Generate STITCHED data for ESM and Experiment
    generate_stitched(esm, variables, tas_time_series, years, scenario, esm_input_paths)
    print(f'{esm} with scenario {scenario} being saved')

