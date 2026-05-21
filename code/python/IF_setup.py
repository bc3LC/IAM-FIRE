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

cwd = os.getcwd() 
print("Current Working Directory:", cwd)
os.chdir('C:/GCAM/Theo/IAM-FIRE/code/python')

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

def make_netcdf_encoding(ds, chunksizes=(1, 360, 720), complevel=1):
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            "zlib": True,
            "complevel": complevel,
            "chunksizes": chunksizes,
            "dtype": "float32"
        }
    return encoding

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
    
def compute_ba_ref(
        ds,
        ba_var="BA_area",
        region=None,
        region_var="region",
        ref_year=None
    ):
    """
    Compute reference burned area for a given variable and region.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing burned area variables.
    ba_var : str
        Name of burned area variable (e.g., "BA_area", "BA_area_for").
    region : str or int or None
        Region identifier. If None → global burned area.
    region_var : str
        Name of region variable in the dataset.
    ref_year : np.datetime64 or str or None
        Reference year to extract. Default: last historical year.

    Returns
    -------
    float
        Reference burned area value.
    """

    # Extract BA variable
    BA = ds[ba_var]

    # If region is specified, filter to that region
    if region is not None:
        if region_var not in ds:
            raise ValueError(f"Region variable '{region_var}' not found in dataset.")
        BA = BA.where(ds[region_var] == region)

    # Sum burned area across spatial dimensions
    BA_hist = BA.sum(dim=["lat", "lon"], skipna=True)

    # Determine reference year
    if ref_year is None:
        ref_year = BA_hist["time"].max().values

    # Extract reference burned area
    BA_ref = BA_hist.sel(time=ref_year).item()

    return BA_ref

def get_ba_ref_field(
    df,
    ba_var="BA_area",
    region=None,
    region_var="region",
    ref_year=None
):
    """
    Return burned-area field fixed at reference year (lat, lon)
    from a pandas DataFrame with MultiIndex (time, lat, lon).
    """

    # Determine reference year
    if ref_year is None:
        ref_year = df.index.get_level_values("time").max()

    # Slice reference year
    BA_ref = df.xs(ref_year, level="time")[ba_var]

    # Optional region filtering
    if region is not None:
        if region_var not in df.columns:
            raise ValueError(f"Region variable '{region_var}' not found in DataFrame.")
        BA_ref = BA_ref[df.xs(ref_year, level="time")[region_var] == region]

    return BA_ref

def weighted_fc_fixed_ba(df, fc_var, BA_ref):
    """
    Fixed burned-area weighting using BA frozen at reference year.
    """

    # Attach BA_ref to every time step via (lat, lon)
    tmp = df[[fc_var]].reset_index().merge(
        BA_ref.rename("BA_ref").reset_index(),
        on=["lat", "lon"],
        how="left"
    )

    # Numerator: sum(FC * BA_ref) per year
    num = (tmp[fc_var] * tmp["BA_ref"]).groupby(tmp["time"]).sum()

    # Denominator: fixed BA
    den = BA_ref.sum()

    return num / den