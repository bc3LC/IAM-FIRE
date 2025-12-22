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
from pyrealm.pmodel.pmodel import PModel
from pyrealm.pmodel import PModelEnvironment
from pyrealm.core.pressure import calc_patm    

# Importing Common Paths/Data/Functions
from IF_setup import IF_PATH, experiment_name, landmask, grid_cell_area, remove_nas

# PATHS
basd_dir = os.path.join(IF_PATH, "output/basd/W5E5v2/")
climate_dir = os.path.join(IF_PATH, "output/climate")
vegetation_dir = os.path.join(IF_PATH, "output/vegetation")

# CONSTANTS
start_year = 2015
end_year = 2100
dates_list = list(range(start_year, end_year+1))
dates_sim_m = pd.date_range(f"{start_year}-01-31", f"{end_year}-12-31", freq="ME")

# INPUTS
CO2_trajectory = pd.read_csv(os.path.join(IF_PATH, f"input/{experiment_name}/CO2_global_trajectory.csv"))
elev = xr.open_dataset(os.path.join(IF_PATH, "input/vegetation_productivity/elev.0.5-deg.nc")).drop_vars("time").sel(time=0) # Unit in meters


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


def CO2_rasterize(CO2_trajectory, scenario):
        
    print(f"Processing CO2 for {scenario}")
    
    CO2_trajectory = CO2_trajectory.drop(columns=["variable", "Units"])
    CO2_trajectory = CO2_trajectory.melt(id_vars=["scenario"], var_name="year", value_name="value")
    CO2_trajectory = CO2_trajectory.pivot(index="year", columns="scenario", values="value").reset_index()

    # Get trajectory data for the given scenario
    time_series_df = CO2_trajectory[['year', scenario]].dropna()
    CO2_time_series = np.array(time_series_df.iloc[:,1].values)
    years = np.array(time_series_df.iloc[:,0].values ).astype(int)

    # Interpolate data
    years, temps = interp(years, CO2_time_series)
    interped_data = pd.DataFrame({'year': years, 'value': temps})
    
    # Target monthly dates (end of month)
    target_dates = pd.date_range('2014-12-31', '2100-12-31', freq='YE')

    # Create coordinates
    lon = np.arange(-179.75, 180.25, 0.5)       # 720 points
    lat = np.arange(-89.75, 90.25, 0.5)       # 360 points
    time = target_dates  
    
    # Create 3D array with homogeneous values across the grid
    interped_data = interped_data[(interped_data["year"] >= 2014) & (interped_data["year"] <= 2100)]
    co2_data = np.tile(interped_data['value'].values[:, np.newaxis, np.newaxis], (1, len(lat), len(lon)))
    
    # Create the xarray Dataset
    ds = xr.Dataset(
        {
            "CO2_concentration": (["time", "lat", "lon"], co2_data)
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon
        }
    )
    
    ds = ds.resample(time="ME").interpolate("linear")
    ds = ds.sel(time=slice("2015-01-31", "2100-12-31"))
        
    ds = ds.where(landmask.mask == 1)
    
    # Save to NetCDF (optional)
    ds.to_netcdf(os.path.join(climate_dir, f'CO2_{scenario}_monthly_{start}_{end}.nc'))

    print(f"CO2 for {scenario} Completed")



def GPP_estimator_sim(dates_sim_m, elev, scenario, esm):

    print(f"Processing GPP for {scenario} - {esm}")

    # 1. Monthly temperature 
    tas_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_tas_global_monthly_{start_year}_{end_year}.nc'        
    temp = xr.open_dataset(os.path.join(climate_dir, tas_pattern))
    
    # 2. Monthly VPD 
    vpd_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_vpd_global_monthly_{start_year}_{end_year}.nc'        
    vpd = xr.open_dataset(os.path.join(climate_dir, vpd_pattern))
    vpd = vpd * 1e-3 # Unit from kPa to Pa
    
    # 3. CO2 
    CO2 = xr.open_dataset(os.path.join(climate_dir, f'CO2_{scenario}_monthly_{start}_{end}.nc')) # Unit in ppm
    CO2["CO2_concentration"].mean(dim="time").plot()
    
    # 4. Atmospheric pressure based on elevation
    elev.coords['lon'] = (elev.coords['lon'] + 180) % 360 - 180
    elev = elev.sortby(elev.lon)
    elev = elev.reindex(lat=list(reversed(elev.lat)))
    elev = elev.where(landmask.mask == 1)
    patm = calc_patm(elev) # Convert elevation to atmospheric pressure
    patm = patm.expand_dims(time=dates_sim_m)
    
    # 5. fAPAR - Static for now but could be projected (pending improvement)
    fapar_clim = xr.open_dataset(os.path.join(IF_PATH, 'input/vegetation_productivity/fapar_monthly_climatology.nc'))
    month_numbers = [d.month for d in dates_sim_m]  # list of month numbers
    fapar = fapar_clim.sel(month=xr.DataArray(month_numbers, dims="time"))  # (time, lat, lon)
    fapar = fapar.assign_coords(time=("time", dates_sim_m))
    fapar = fapar.drop_vars("month")  # optional cleanup
    
    # 6. PPFD 
    ppfd_pattern = f'{esm}_STITCHES_W5E5v2_{scenario}_ppfd_global_monthly_{start_year}_{end_year}.nc'     
    ppfd = xr.open_dataset(os.path.join(climate_dir, ppfd_pattern))
    
    print("Step 1. Completed. All datasets prepared")
    
    # Extract the six variables for the two months and convert from
    ds = xr.merge([temp, vpd, CO2, patm, fapar, ppfd])    
    temp_df = ds["tas"].to_numpy()
    co2_df = ds["CO2_concentration"].to_numpy()
    patm_df = ds["data"].to_numpy()
    vpd_df = ds["vpd"].to_numpy()
    fapar_df = ds["fAPAR"].to_numpy()
    ppfd_df = ds["rsds"].to_numpy()
    
    # Mask out temperature values below -25°C
    temp_df[temp_df < -25] = np.nan
    
    # # Clip VPD to force negative VPD to be zero
    # vpd_df = np.clip(vpd_df, 0, np.inf)
    
    # Calculate the photosynthetic environment 
    env = PModelEnvironment(tc=temp_df, co2=co2_df, patm=patm_df, vpd=vpd_df)
    env.summarize()
    
    # Run the P model
    model = PModel(env)
    model.summarize()
    
    # Estimate GPP from the Pmodel providing physical variables (fAPAR + PPFD)
    model.estimate_productivity(fapar=fapar_df, ppfd=ppfd_df)
    gpp_df = model.gpp
    
    # Convert GPP to gC.m2.month from µg C m-2 s-1 
    gpp_df = gpp_df * 1e-6 * 60 * 60 * 24 * 30

    # Convert GPP to xarray and save 
    # Time: monthly values, let's assume it starts in Jan 2000
    time = pd.date_range(start='2015-01-31', end="2100-12-31", freq='ME')  # monthly start
    
    # Latitude: from 89.75 to -89.75 (360 steps at 0.5°)
    lat = np.linspace(-89.75, 89.75, 360)
    
    # Longitude: from -179.75 to 179.75 (720 steps at 0.5°)
    lon = np.linspace(-179.75, 179.75, 720)
    
    # Create the DataArray
    gpp_xr = xr.DataArray(
        data=gpp_df,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="gpp",
        attrs={"units": "gC m⁻² month⁻¹", "long_name": "Gross Primary Production"}
    )
    
    # Optional: wrap into a Dataset if needed and save 
    gpp_ds = gpp_xr.to_dataset()
    # gpp_ds.to_netcdf(os.path.join(vegetation_dir, f'GPP_pModel_{scenario}_2015-2100.nc'))
    
    print("Step 2. Completed. GPP Estimated with p_model")
    
    # Compute the seasonality index (Optional, this is not included in the final GLM)
    gpp_monthly = gpp_ds.gpp
    
    # Check if you have 12 months per year for each year
    # Group by year and apply function
    def compute_seasonality(x):
        monthly = x.groupby("time.month").mean(dim="time")  # (month, lat, lon)
        seasonality = (monthly.max("month") - monthly.min("month")) / monthly.mean("month")
        return seasonality  # (lat, lon)
    
    # Apply this across years
    seasonality_per_year = gpp_monthly.groupby("time.year").apply(compute_seasonality)
    
    # Rename year to time with mid-year timestamp
    seasonality_per_year = seasonality_per_year.rename({"year": "time"})
    seasonality_per_year["time"] = [np.datetime64(f"{int(y)}-12-31") for y in seasonality_per_year.time.values]    

    # Store in dataset
    gpp_ds = gpp_ds.resample(time="YE").sum()
    gpp_ds["GPP_seasonality"] = seasonality_per_year
        
    # Save the GPP annual 
    gpp_ds.to_netcdf(os.path.join(vegetation_dir, f'GPP_pModel_{scenario}_{esm}_season_2015-2100.nc')) # Save the seasonality index
        
    print("Step 3. Completed. GPP Seasonality estimated + File saved")

    
def var_multimodel_mean(scenario: str, esms: list[str], variable: str):
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
        file_name = f'GPP_pModel_{scenario}_{esm}_season_2015-2100.nc'
        ds = xr.open_dataset(os.path.join(vegetation_dir, file_name), engine="netcdf4")
        print(ds)
        datasets.append(ds)

    # Merge all datasets from different ESMs
    if datasets:
        merged_ds = xr.concat(datasets, dim="model") # Concatenating along a new 'model' dimension
        print(merged_ds)
        mme_mean = merged_ds.mean(dim="model")  # Compute the mean across models

        # Save the merged dataset
        save_name = f'GPP_pModel_{scenario}_MME_season_2015-2100.nc'
        mme_mean.to_netcdf(os.path.join(vegetation_dir, save_name), engine="netcdf4")
    
    print(f"MME Computed and saved for {scenario} and annual {variable}")    

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

    print("Runs are prepared, processing GPP estimations")
    
    # STEP 1. Estimate GPP per Scenario and ESM 
    # Iterate through each requested ESM and Experiment/Scenario
    for i, scenario in enumerate(scenarios):
        CO2_rasterize(CO2_trajectory, scenario)
        for j, esm in enumerate(esms):
            # print(f'{esm} with scenario {scenario} being saved to {climate_dir}')
            GPP_estimator_sim(dates_sim_m, elev, scenario, esm)
    print("GPP Completed per Scenario-ESM pair")


    