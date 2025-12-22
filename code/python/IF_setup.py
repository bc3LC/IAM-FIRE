# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:02:30 2025

@author: theo.rouhette
"""

import numpy as np 
import xarray as xr
import pandas as pd
import os
from pathlib import Path

# COMMON PATHS 
IF_PATH = "C:/GCAM/Theo/IAM-FIRE"
# IF_PATH = "/scratch/bc3lc/IAM-FIRE"

# HPC CONDA ENVIRONMENT
conda_env = "iam-fire-env"

# EXPERIMENT NAME 
experiment_name = "stitches_experiment"

# COMMON INPUTS
landmask = xr.open_dataset(os.path.join(IF_PATH, "input/static_layers/landseamask_no-ant.nc")).drop_vars("time").sel(time=0)
landmask = landmask.reindex(lat=list(reversed(landmask.lat)))
regions = xr.open_dataset(os.path.join(IF_PATH, "input/static_layers/GFED_Regions.nc"))

# CREATE OUTPUT SUB-FOLDERS 
subfolders = [
    "basd",
    "climate",
    "dask_tmp",
    "figures",
    "fire_impacts",
    "fuel_consumption",
    "historic_glm",
    "land_use",
    "stitches",
    "vegetation",
]

for name in subfolders:
    Path("output", name).mkdir(parents=True, exist_ok=True)

###############################################################################
# FUNCTIONS
###############################################################################


# Function to calculate grid cell area in km²
def grid_cell_area(lat, dlon=0.5, dlat=0.5):
    """Calculate grid cell area given latitude, assuming a 0.5° x 0.5° grid."""
    R = 6371  # Earth radius in km
    lat_rad = np.radians(lat)
    
    # Width of cell (km) varies with latitude
    cell_width = (np.pi / 180) * R * np.cos(lat_rad) * dlon  
    # Height of cell (km)
    cell_height = (np.pi / 180) * R * dlat  
    
    grid_area = cell_width * cell_height 
    
    return grid_area  # Area in km²

def remove_nas(x):
    return x[~pd.isnull(x)]

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_nme(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true))))


def generate_scaled_coordinates_helper(coord_min: float,
                                       coord_max: float,
                                       res: float,
                                       ascending: bool = True,
                                       decimals: int = 3) -> np.array:
    """Generate a list of evenly-spaced coordinate pairs for the output grid based on lat, lon values.

    :param coord_min:                   Minimum coordinate in range.
    :type coord_min:                    float

    :param coord_max:                   Maximum coordinate in range.
    :type coord_max:                    float

    :param ascending:                   Ascend coordinate values if True; descend if False
    :type ascending:                    bool

    :param decimals:                    Number of desired decimals to round to.
    :type decimals:                     int

    :returns:                           Array of coordinate values.

    """

    # distance between centroid and edge of grid cell
    center_spacing = res / 2

    if ascending:
        return np.arange(coord_min + center_spacing, coord_max, res).round(decimals)

    else:
        return np.arange(coord_max - center_spacing, coord_min, -res).round(decimals)