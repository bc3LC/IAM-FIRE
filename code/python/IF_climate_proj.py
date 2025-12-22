# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:43:30 2025

@author: theo.rouhette
"""
# Importing Needed Libraries
import os  # For navigating os
import sys  # Getting system details
import numpy as np  # Numerical / array functions
import pandas as pd  # Data functions
import xarray as xr  # Reading and manipulating NetCDF data
import shutil

# Importing Common Paths/Data/Functions
from IF_setup import IF_PATH, landmask, remove_nas

# PATHS
basd_dir = os.path.join(IF_PATH, "output/basd/W5E5v2/")
climate_dir = os.path.join(IF_PATH, "output/climate")

# CONSTANTS
start_year = 2015
end_year = 2100
dates_list = list(range(start_year, end_year+1))

def tas_estimator_sim(scenario: str, esm: str):
    
    """
    Estimate monthly near-surface air temperature. 
    Used for GPP by the P-Model
    """
    
    # Load daily tas
    tas_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_tas_global_daily_{start}_{end}.nc'
    stitches_tas = xr.open_dataset(os.path.join(basd_dir, f'{esm}', f'{scenario}', 'basd', tas_pattern), chunks={"time": 365})
    
    stitches_tas = stitches_tas.resample(time="ME").mean()   
    stitches_tas = stitches_tas.tas - 273.15 # Unit from K to C
    stitches_tas = stitches_tas.where(landmask.mask == 1)
    
    stitches_tas.to_netcdf(os.path.join(climate_dir, f'{esm}_STITCHES_W5E5v2_{scenario}_tas_global_monthly_{start}_{end}.nc'))

def ppfd_estimator_sim(scenario: str, esm: str):
    
    """
    Estimate monthly photosynthetic photon density from daily solar radiation
    Used for GPP by the P-Model
    """
    
    # Load daily rsds
    rsds_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_rsds_global_daily_{start}_{end}.nc'
    stitches_rsds = xr.open_dataset(os.path.join(basd_dir, f'{esm}', f'{scenario}', 'basd', rsds_pattern), chunks={"time": 365})
    
    stitches_rsds = stitches_rsds.resample(time="ME").mean()   
    stitches_rsds = stitches_rsds.where(landmask.mask == 1)
    stitches_ppfd = stitches_rsds * 4.6 * 0.5 # Like HAAS, Conversion as factor of 4.6 (from W.m2 to umol.m2.s) and 0.5 (fraction of incoming solar irradiance that is photosynthetically active radiation (PAR); defaults to 0.5)
    
    stitches_ppfd.to_netcdf(os.path.join(climate_dir, f'{esm}_STITCHES_W5E5v2_{scenario}_ppfd_global_monthly_{start}_{end}.nc'))


def pr_estimator_sim(scenario: str, esm: str):
    
    """
    Estimate daily and monthly precipitation metrics. 
    Produces mm/day, 30-day rolling sum, and monthly aggregates.
    """
    # Load daily precipitation
    pr_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_pr_global_daily_{start}_{end}.nc'
    stitches_pr = xr.open_dataset(os.path.join(basd_dir, f'{esm}', f'{scenario}', 'basd', pr_pattern), chunks={"time": 365})
    
    # Convert kg m-2 s-1 → mm/day
    stitches_pr["pr"] = stitches_pr["pr"] * 86400
    stitches_pr["pr"].attrs["units"] = "mm/day"
    
    # 30-day rolling cumulative precipitation (daily)
    stitches_pr["pr_30d_sum"] = stitches_pr["pr"].rolling(time=30, min_periods=1).sum()
    stitches_pr["pr_30d_sum"].attrs["description"] = "30-day rolling sum of precipitation (mm)"
    
    # Monthly aggregates
    stitches_pr_monthly = stitches_pr.resample(time="ME").mean()
    stitches_pr_monthly["pr_sum_month"] = stitches_pr["pr"].resample(time="ME").sum()
    stitches_pr_monthly["pr_30d_sum_monthend"] = stitches_pr["pr_30d_sum"].resample(time="ME").last()
    
    # Apply landmask (broadcast automatically if dims match)
    stitches_pr_monthly = stitches_pr_monthly.where(landmask.mask == 1)
    
    stitches_pr_annual = stitches_pr_monthly.resample(time="YE").mean()
    stitches_pr_annual.to_netcdf(os.path.join(climate_dir, f'{esm}_STITCHES_W5E5v2_{scenario}_pr_global_annual_{start}_{end}.nc'))


def vpd_estimator_sim(scenario: str, esm: str):
          
    # Open the tas and hurs files 
    tas_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_tas_global_monthly_{start}_{end}.nc'
    hurs_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_hurs_global_monthly_{start}_{end}.nc'
        
    stitches_tas = xr.open_dataset(os.path.join(basd_dir, f'{esm}', f'{scenario}', 'basd', tas_pattern))
    stitches_hurs = xr.open_dataset(os.path.join(basd_dir, f'{esm}', f'{scenario}', 'basd', hurs_pattern))
        
    stitches_tas = stitches_tas.tas - 273.15
    
    # stitches_vpd = xr.merge([stitches_tas, stitches_hurs]).sel(time=dates_hist)
    stitches_vpd = xr.merge([stitches_tas, stitches_hurs])
    
    # Compute the SVP + VPD in kPa
    stitches_vpd = stitches_vpd.assign(svp=lambda stitches_vpd: 610.8 * np.exp((17.27 * stitches_vpd.tas) / (stitches_vpd.tas + 237.3)))        
    stitches_vpd = stitches_vpd.assign(vpd = lambda stitches_vpd: (1-stitches_vpd.hurs/100) * stitches_vpd.svp)  
    stitches_vpd = stitches_vpd.assign(vpd = lambda stitches_vpd: stitches_vpd.vpd / 1000)
              
    # Dropping non VPD values
    stitches_vpd = stitches_vpd.drop_vars(["tas", "hurs", "svp"])

    # Landmask 
    stitches_vpd = stitches_vpd.where(landmask.mask == 1)
                    
    # Maximum annual for GLM prediction
    stitches_vpd_annual = stitches_vpd.resample(time="YE").max()
    stitches_vpd_annual.to_netcdf(os.path.join(climate_dir, f'{esm}_STITCHES_W5E5v2_{scenario}_vpd_global_annual_{start}_{end}.nc'))

    # Average monthly for GPP prediction
    stitches_vpd_monthly = stitches_vpd.resample(time="ME").mean()
    stitches_vpd_monthly.to_netcdf(os.path.join(climate_dir, f'{esm}_STITCHES_W5E5v2_{scenario}_vpd_global_monthly_{start}_{end}.nc'))
        
    print(f"Gridded monthly and annual VPD saved for scenario: {scenario} and ESM: {esm}")

    return stitches_vpd, stitches_vpd_annual
    
def ndd_estimator_sim(scenario: str, esm: str):

    # Open the pr files 
    pr_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_pr_global_daily_{start}_{end}.nc'
    stitches_pr = xr.open_dataset(os.path.join(basd_dir, f'{esm}', f'{scenario}', 'basd', pr_pattern), chunks={"time": 365})

    # Convert rain from kg m−2 s−1 to mm day−1 with Quilcaille equation 
    # stitches_pr["pr"] = stitches_pr["pr"] * 24 * 3600  # Keep it as an xarray DataArray
    stitches_pr["pr"] *= 24 * 3600  # This is an in-place operation
    
    # Set dry days threshold to 1 mm day-1
    dry_day_thresh_mm=1.0
    
    # Identify dry days
    dry_days = (stitches_pr["pr"] <= dry_day_thresh_mm).astype("int8")
    
    # Sum at monthly scale 
    dry_days_monthly = dry_days.resample(time="ME").sum()    

    # Landmask 
    dry_days_monthly = dry_days_monthly.where(landmask.mask == 1)
    
    # Rename pr to ndd
    dry_days_monthly = dry_days_monthly.rename("ndd")
    dry_days_monthly = dry_days_monthly.to_dataset()
    print(dry_days_monthly)
    
    # Compute the seasonality index    
    # Check if you have 12 months per year for each year
    # Group by year and apply function
    def compute_seasonality(x):
        monthly = x.groupby("time.month").mean(dim="time")  # (month, lat, lon)
        seasonality = (monthly.max("month") - monthly.min("month")) / monthly.mean("month")
        return seasonality  # (lat, lon)
    
    # Apply this across years
    seasonality_per_year = dry_days_monthly.groupby("time.year").apply(compute_seasonality)
    
    # Rename year to time with mid-year timestamp
    seasonality_per_year = seasonality_per_year.rename({"year": "time"})
    seasonality_per_year["time"] = [np.datetime64(f"{int(y)}-12-31") for y in seasonality_per_year.time.values]    
    print(seasonality_per_year)

    # Store in dataset
    dry_days_annual = dry_days_monthly.resample(time="YE").max()
    dry_days_annual["NDD_seasonality"] = seasonality_per_year["ndd"]
        
    # Save output to NetCDF
    output_path = os.path.join(climate_dir, f'{esm}_STITCHES_W5E5v2_{scenario}_ndd_global_annual_{start}_{end}.nc')
    dry_days_annual.to_netcdf(output_path)
        
    print("Completed. NDD Seasonality estimated + File saved")
    
    return dry_days_monthly, dry_days_annual


def wind_estimator_sim(scenario: str, esm: str):

    tas_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_tas_global_monthly_{start}_{end}.nc'
    wind_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_sfcWind_global_monthly_{start}_{end}.nc'
    
    stitches_tas = xr.open_dataset(os.path.join(basd_dir, f'{esm}', f'{scenario}', 'basd', tas_pattern))
    stitches_wind = xr.open_dataset(os.path.join(basd_dir, f'{esm}', f'{scenario}', 'basd', wind_pattern))
    
    # Merge wind and tas 
    stitches_merge = xr.merge([stitches_wind, stitches_tas])
    
    # Sum at monthly scale 
    stitches_merge_m = stitches_merge.resample(time="ME").mean()    

    # Landmask 
    stitches_merge_m = stitches_merge_m.where(landmask.mask == 1)
    
    # Split dataset by years and process each one
    annual_winds = []

    for year in dates_list:
        
        # year = 2001
        year_ds = stitches_merge_m.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        # year_ds.tas.mean(dim='time').plot()

        # Find month with max temperature for each pixel
        max_month = year_ds['tas'].fillna(0).argmax(dim='time')
        
        # Stack spatial dimensions
        stacked = year_ds.stack(point=('lat', 'lon'))
        
        # Create matching stacked indices
        max_month_stacked = max_month.stack(point=('lat', 'lon'))
        
        # Get wind speeds at max temperature months
        annual_wind_stacked = stacked['sfcWind'].isel(time=max_month_stacked)
        
        # Unstack back to original grid
        annual_wind = annual_wind_stacked.unstack('point')
        
        # Add year coordinate
        annual_wind = annual_wind.assign_coords(year=year)
        annual_wind = annual_wind.expand_dims('year')
        
        annual_winds.append(annual_wind)
    
    # Combine all years into a single dataset
    combined = xr.concat(annual_winds, dim='year')
    combined = combined.to_dataset()
    
    # Rename 'year' to 'time'
    combined = combined.drop_vars("time")
    combined = combined.rename({"year": "time"})
    
    # Convert integer years to datetime64 (assuming yearly data, setting date to January 1st)
    combined["time"] = np.array([np.datetime64(f"{year}-12-31") for year in combined["time"].values])
        
    # Save the final dataset
    combined.to_netcdf(os.path.join(climate_dir, f'{esm}_STITCHES_W5E5v2_{scenario}_sfcWind_global_annual_{start_year}_{end_year}.nc'))
    
    print(f"Gridded annual WIND SPEED saved for scenario: {scenario} and ESM: {esm}")

    
def var_multimodel_mean(scenario: str, esms: list[str], variable: str, frequency: str):
    """
    Compute the multi-model ensemble (MME) mean for a given variable across multiple ESMs.

    Args:
        scenario (list[str]): List of climate scenarios.
        esm (list[str]): List of Earth System Models (ESMs).
        variable (str): Variable name to process.

    Returns:
        dict: A dictionary containing the merged datasets for each scenario.
    """

    datasets = []  # List to hold datasets for merging
    print(f"Processing scenario: {i} and variabe: {variable}")

    for esm in esms:
        file_name = f'{esm}_STITCHES_W5E5v2_{scenario}_{variable}_global_{frequency}_{start}_{end}.nc'
        ds = xr.open_dataset(os.path.join(climate_dir, file_name), engine="netcdf4")
        print(ds)
        datasets.append(ds)

    # Merge all datasets from different ESMs
    if datasets:
        merged_ds = xr.concat(datasets, dim="model") # Concatenating along a new 'model' dimension
        print(merged_ds)
        mme_mean = merged_ds.mean(dim="model")  # Compute the mean across models

        # Save the merged dataset
        save_name = f'MME_STITCHES_W5E5v2_{scenario}_{variable}_global_{frequency}_{start}_{end}.nc'
        mme_mean.to_netcdf(os.path.join(climate_dir, save_name), engine="netcdf4")
    
    print(f"MME Computed and saved for {scenario} and {frequency} {variable}")    


if __name__ == "__main__":


    # Name of the current experiment directory
    run_directory = str(sys.argv[1])

    # Input file path
    input_files_path = os.path.join(IF_PATH, 'input/', run_directory)
    
    # Reading the run details
    run_manager_df = pd.read_csv(os.path.join(input_files_path, 'run_manager.csv'))

    # Extracting needed infor and formatting the run details
    esms = remove_nas(run_manager_df['ESM'].values)
    variables = remove_nas(run_manager_df['Variable'].values)
    scenarios = remove_nas(run_manager_df['Scenario'].values)

    start, end = run_manager_df['application_period'].iloc[0].split('-')
    dates_sim = slice(f"{start}-01-31", f"{end}-12-31")
    dates_sim_df = pd.date_range(start=f"{start}-01-31", end=f"{end}-12-31", freq='ME') 

    print("Runs are prepared, processing with climate estimations")
    
    # STEP 1. Estimate climate projections per Scenario and ESM 
    for i, scenario in enumerate(scenarios):
        for j, esm in enumerate(esms):
            print(f'{esm} with scenario {scenario} being saved to {climate_dir}')
            tas_estimator_sim(scenario, esm)
            ppfd_estimator_sim(scenario, esm)
            pr_estimator_sim(scenario, esm)
            vpd_estimator_sim(scenario, esm)
            ndd_estimator_sim(scenario, esm)
            wind_estimator_sim(scenario, esm)
    print("Climate projections completed")

    