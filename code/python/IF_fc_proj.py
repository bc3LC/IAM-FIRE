# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:11:51 2025

@author: theo.rouhette

"""

# Create the biome vector from Ecoregions 
# Importing Needed Libraries
import sys  # Getting system details
import os  # For navigating os
import numpy as np  # Numerical / array functions
import pandas as pd  # Data functions
import xarray as xr  # Reading and manipulating NetCDF data
import shutil

# Importing Common Paths/Data/Functions
from IF_setup import IF_PATH, experiment_name, landmask, regions, grid_cell_area, remove_nas, generate_scaled_coordinates_helper

# PATHS
basd_dir = os.path.join(IF_PATH, "output/basd/W5E5v2/")
climate_dir = os.path.join(IF_PATH, "output/climate")
fc_dir = os.path.join(IF_PATH, "output/fuel_consumption")

# INPUTS
biome_nc = xr.open_dataset(os.path.join(IF_PATH, "input/fuel_consumption/Biome_EF_Full.nc"))
fuel_load = xr.open_dataset(os.path.join(IF_PATH, "input/fuel_consumption/Fuel_Load_ECMWF_05deg.nc"))
gfed_fc = xr.open_dataset(os.path.join(IF_PATH, "input/historic_glm/GLM_Historic_Inputs.nc"))[["FC_for_GFED", "FC_nonf_GFED", "FC_total_GFED"]]

# CONSTANTS
start_year = 2015
end_year = 2100
dates_sim = pd.date_range(start=f"{start_year}-01-31", end=f"{end_year}-12-31", freq='YE')  # 'ME' means Month Start

# Values of combustion factor ranges from Wees 2022 (Table S3)
cc_range_table = {
    "boreal": {
        "stem": (10, 30),
        "leaf": (90, 100),
        "cwd": (30, 70),
        "litter": (70, 100)
    },
    "temperate": {
        "stem": (30, 50),
        "leaf": (90, 100),
        "cwd": (20, 60),
        "litter": (70, 100)
    },
    "tropical": {
        "stem": (20, 40),
        "leaf": (90, 100),
        "cwd": (30, 70),
        "litter": (80, 100)
    }
}

###########################################################################
# COMBUSTION COMPLETENESS/FACTORS BASED ON SOIL MOISTURE 
def cc_scaling(cc_range_table, vpd, scenario, esm):  

    def vpd_normalize(vpd, vpd_min=0.1, vpd_max=5):
        """Compute εSM from soil moisture (in m³/m³)"""
        # Clip VPD to the [vpd_min, vpd_max] range
        vpd_clipped = vpd.clip(min=vpd_min, max=vpd_max)
        
        # Normalize to 0–1 range
        scalar = (vpd_clipped - vpd_min) / (vpd_max - vpd_min)
        return scalar
    
    vpd_scalar = vpd_normalize(vpd)
    
    def linear_scale_cc(scalar, lower, upper):
        """
        Linearly scale soil moisture to a combustion completeness (CC) value.
        
        Given that drier conditions (sm = sm_min) result in higher CC (upper bound)
        and wetter conditions (sm = sm_max) yield lower CC (lower bound), this function
        computes:
        
            CC = lower + (upper - lower) * (1 - (sm - sm_min) / (sm_max - sm_min))
        
        Parameters
        ----------
        sm : xarray.DataArray or numpy.ndarray
            Soil moisture scalar.
        lower : float
            The lower bound CC value for this fuel type.
        upper : float
            The upper bound CC value for this fuel type.
        sm_min : float, optional
            Minimum soil moisture value (default 0.0).
        sm_max : float, optional
            Maximum soil moisture value (default 1.0).
            
        Returns
        -------
        cc : same type as sm
            The computed combustion completeness.
        """
        # Normalize soil moisture to [0, 1]; note that lower soil moisture gives higher CC.
        # norm = (scalar - sm_min) / (sm_max - sm_min)
        cc = lower + (upper - lower) * (scalar)
        return cc

    # CC_test = linear_scale_cc(sm_scalar, 20, 50)
    
    def compute_cc_map(biome_nc, vpd_nc, cc_range_table):
        """
        Compute combustion completeness (CC) for each fuel type and each grid cell.
        
        Parameters
        ----------
        biome_nc : str
            File path to the netCDF file that contains the biome information.
        sm_nc : str
            File path to the netCDF file that contains the soil moisture data.
        cc_range_table : dict
            A nested dictionary where keys are forest biome types (e.g., 'boreal', 'temperate', 'tropical')
            and values are dictionaries mapping a fuel type (e.g., 'stem', 'leaf', 'litter', 'BGB')
            to a tuple (lower_bound, upper_bound).
            Tundra and sparse boreal have been grouped with boreal.)
        sm_min : float, optional
            The minimum soil moisture value expected (default: 0.0).
        sm_max : float, optional
            The maximum soil moisture value expected (default: 1.0).
            
        Returns
        -------
        cc_results : dict
            A dictionary mapping each fuel type to an xarray.DataArray (with the same spatial dimensions as the input),
            containing the computed CC values.
        """
        
        # Assume the variable names in the netCDFs are 'biome' and 'soil_moisture'
        biome_da = biome_nc['biome']
        vpd_da = vpd_scalar['vpd']
        
        # Create a dictionary that will hold the computed CC for each fuel type.
        cc_results = {}
        
        # Retrieve the list of fuel types from one of the biome entries.
        sample_biome = next(iter(cc_range_table))
        fuel_types = list(cc_range_table[sample_biome].keys())
        
        # Initialize an empty (nan-filled) DataArray for each fuel type with the same dimensions as the soil moisture field.
        for fuel in fuel_types:
            cc_results[fuel] = xr.full_like(vpd_da, np.nan, dtype=float)
        
        # Mapping of numeric biome codes to string keys in cc_range_table
        biome_code_map = {
            0: 'boreal',
            1: 'temperate',
            2: 'tropical'
        }
        
        # Loop over each biome code and corresponding string key
        for biome_code, biome_type in biome_code_map.items():
            
            fuels = cc_range_table[biome_type]
            
            # Create a boolean mask for grid cells matching this biome code
            mask = (biome_da == biome_code)
            
            # For each fuel type, compute the CC for the masked grid cells.
            for fuel, (lower, upper) in fuels.items():
                # fuel = "stem"
                # lower = 20
                # upper = 50
                # Apply the linear scaling on soil moisture values where the mask is True.
                # Note: Using .where() retains NaN for cells outside the mask.
                vpd_masked = vpd_da.where(mask)
                cc_values = linear_scale_cc(vpd_masked, lower, upper)
                cc_values.mean(dim="time").plot()
                
                # Combine the computed values into the result array for this fuel type.
                # Here, we use xr.where() to replace only where the mask is True.
                cc_results[fuel] = xr.where(mask, cc_values, cc_results[fuel])
                
        # cc_results["stem"].mean(dim="time").plot()
        return cc_results
    
    cc_maps = compute_cc_map(biome_nc, vpd, cc_range_table)
    cc_final = xr.merge([cc_maps])
    cc_final.to_netcdf(os.path.join(fc_dir, f"CF_{scenario}_{esm}_2015-2100.nc"))
    
def fuel_consumption(scenario, esm, fuel_load, biome_nc, gfed_fc, regions):
    

    ###########################################################################
    # 1. Load the scenario-specific scaled CC 
    cc_final = xr.open_dataset(os.path.join(fc_dir, f"CF_{scenario}_{esm}_2015-2100.nc"))

    # Merge and mask
    fe_param = xr.merge([biome_nc, fuel_load, cc_final])    
    fe_param = fe_param.where(landmask.mask == 1)

    # Get FC specific to each fuel types. FL from Kg to g. CC from % to fraction. EF from DM to C 
    fe_param["FC_Stem"] = fe_param["Live_Wood"] * 1e3 * fe_param["stem"] * 0.01 * fe_param["EF_C"] 
    fe_param["FC_Leaf"] = fe_param["Live_Leaf"] * 1e3 * fe_param["leaf"] * 0.01 * fe_param["EF_C"] 
    fe_param["FC_Litter"] = fe_param["Dead_Foliage"] * 1e3 * fe_param["litter"] * 0.01 * fe_param["EF_C"] 
    fe_param["FC_CWD"] = fe_param["Dead_Wood"] * 1e3 * fe_param["cwd"] * 0.01 * fe_param["EF_C"] 
  
    # Sum the FC for Forests (All fuel types) and for Non-Forests (Leaf and Litter)
    fe_param["FC_NonForest"] = fe_param["FC_Leaf"] + fe_param["FC_Litter"]
    fe_param["FC_Total"] = fe_param["FC_Stem"] + fe_param["FC_Leaf"] + fe_param["FC_Litter"] + fe_param["FC_CWD"]
    
    ###########################################################################
    # 2. Add boreal soil carbon - following Park et al. 2023 default FC for peat fires in boreal forests of 2200 gC.m2 
    boreal_mask = (gfed_fc.lat > 50) & (gfed_fc["FC_for_GFED"].mean(dim="time") > 2200) # Boolean mask: True where condition met, False elsewhere
    fe_param_boreal = fe_param.copy()
    fc_boreal_soil = xr.where(boreal_mask, 2200, 0)
    fc_boreal_soil = fc_boreal_soil.broadcast_like(fe_param_boreal)
    fe_param_boreal["FC_Boreal_Soil"] = fc_boreal_soil
    fe_param_boreal["FC_Forest"] = fe_param_boreal["FC_Total"] + fe_param_boreal[f"FC_Boreal_Soil"]
        
    ###########################################################################
    # 3. Apply the change ratio of dynamic FC to static GFED layer at 2019
    
    # Lighten the final NETCDF
    fe_param_final = fe_param_boreal[["FC_Forest", "FC_NonForest"]]
    
    # Transform the projection to change ratio through time (after 2019 only)
    fc_2019 = fe_param_final.sel(time="2019-12-31")
    fc_2019_safe = fc_2019.where(fc_2019 != 0, np.nan) # Avoid division by zero: replace 0 with NaN in fc_2019
    fc_ratio = fe_param_final / fc_2019_safe
    fc_ratio = fc_ratio.where(fc_ratio.time > np.datetime64("2019-12-31"), 1.0) # Keep ratio = 1 for years <= 2019
    
    # Apply that to FC_GFED at the year 2019
    gfed_fc_2019 = gfed_fc.sel(time="2019-12-31")
    gfed_fc_proj = xr.Dataset({
        "FC_for_GFED_proj": gfed_fc_2019["FC_for_GFED"] * fc_ratio["FC_Forest"],
        "FC_nonf_GFED_proj": gfed_fc_2019["FC_nonf_GFED"] * fc_ratio["FC_NonForest"],
        "FC_total_GFED_proj": gfed_fc_2019["FC_total_GFED"] * fc_ratio["FC_Forest"]

    })
    gfed_fc_proj = gfed_fc_proj.assign_coords(time=fe_param_final.time) # Assign time coordinate from fc_test
    
    # Optional: fill NaNs (from 0 division) with 2019 GFED values
    gfed_fc_proj["FC_for_GFED_proj"] = gfed_fc_proj["FC_for_GFED_proj"].fillna(gfed_fc_2019["FC_for_GFED"])
    gfed_fc_proj["FC_nonf_GFED_proj"] = gfed_fc_proj["FC_nonf_GFED_proj"].fillna(gfed_fc_2019["FC_nonf_GFED"])
    gfed_fc_proj["FC_total_GFED_proj"] = gfed_fc_proj["FC_total_GFED_proj"].fillna(gfed_fc_2019["FC_total_GFED"])

    # Save Netcdf 
    print(f"Saving projected GFED FC for {scenario} and {esm}")
    gfed_fc_proj.to_netcdf(os.path.join(fc_dir, f"FC_{scenario}_{esm}_2019-2100_proj.nc"))
    
if __name__ == "__main__":
    
    ###########################################################################
    # STEP 0. Prepare the runs 
    # Name of the current experiment directory
    run_directory = str(sys.argv[1])
    # Input file path
    input_files_path = os.path.join(IF_PATH, 'input/', run_directory)
    # Reading the run details
    run_manager_df = pd.read_csv(os.path.join(input_files_path, 'run_manager.csv'))
    # Extracting needed infor and formatting the run details
    esms = remove_nas(run_manager_df['ESM'].values)
    scenarios = remove_nas(run_manager_df['Scenario'].values)
    print("ESM: ", esms)
    print("Scenarios: ", scenarios)
    print("Launching the VPD scaling and combustion completeness")
    
    # Run for each pair of scenario-esm 
    for scenario in scenarios: 
        print(f"Processing {scenario}")
        for esm in esms:
            print(f"Processing {esm}")
            vpd_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_vpd_global_monthly_{start_year}_{end_year}.nc'        
            vpd = xr.open_dataset(os.path.join(climate_dir, vpd_pattern))
            vpd = vpd.resample(time="YE").mean()
            vpd = vpd.reindex(lat=list(reversed(vpd.lat)))
            print(vpd)

            # Run the functions to scale CC based on VPD and to estimate FC from the scaled CC and FL dataset
            cc_scaling(cc_range_table, vpd, scenario, esm) # Takes the VPD projection and computes the CC. Output: CC projection 
            fuel_consumption(scenario, esm, fuel_load, biome_nc, gfed_fc, regions) # Takes the CC and the FL dataset and computes FC. Output: FC projection
            print(f"FC Completed for {scenario} and {esm}")
    print("FC Completed for all scenarios and ESMs")
    




