# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:39:22 2025

@author: theo.rouhette

"""

# Importing Needed Libraries
import os  # For navigating os
import numpy as np  # Numerical / array functions
import pandas as pd  # Data functions
import xarray as xr  # Reading and manipulating NetCDF data
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LogisticGAM, s, f

# Importing Common Paths/Data/Functions
from IF_setup import IF_PATH, landmask, regions, grid_cell_area, compute_rmse, compute_nme

# CONSTANTS 
start_year = 2002
end_year = 2019
dates_obs = slice(f"{start_year}-01-31", f"{end_year}-12-31")
dates_obs_df = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='YE')  
dates_sim = pd.date_range(start=f"{end_year}-01-31", end=f"2100-12-31", freq='YE')  
variable = ['tas', 'pr', 'hurs', "sfcWind", "rsds"]

# INPUTS
nc_annual = xr.open_dataset(os.path.join(IF_PATH, "input/historic_glm/GLM_Historic_Inputs.nc")) 

# PATHS
glm_output_dir = os.path.join(IF_PATH, "output/historic_glm")

def glm_run(nc_annual, save=True):
    
    import numpy as np
    import matplotlib.pyplot as plt

    ##########################################################################
    # 1. DATA PROCESSING
    ##########################################################################

    # Apply land mask 
    nc_annual = nc_annual.where(landmask.mask == 1)
    
    # Drop missing or zero data 
    nc_annual_proc = nc_annual.where(
        (nc_annual["vpd"] != 0) &
        (nc_annual["ndd"] != 0) &
        (nc_annual["sfcWind"] != 0) &
        (nc_annual["GPP"] > 0.1) &
        (nc_annual["BA_frac"] >= 0)
    )
    
    # Add the log transformations
    nc_annual_proc["GPP_log"] = np.log2(nc_annual_proc["GPP"]) 
    nc_annual_proc["GPP_season_log"] = np.log2(nc_annual_proc["GPP_seasonality"]) 
    nc_annual_proc["ndd_log"] = np.log2(nc_annual_proc["ndd"]) 
    nc_annual_proc["ndd_season_log"] = np.log2(nc_annual_proc["NDD_seasonality"]) 
    nc_annual_proc["vpd_log"] = np.log2(nc_annual_proc["vpd"])
    nc_annual_proc["sfcWind_log"] = np.log2(nc_annual_proc["sfcWind"])
    nc_annual_proc['pr_sum_month_log'] = np.log(nc_annual_proc['pr_sum_month'].clip(min=1e-6)) 
    nc_annual_proc['pop_sqrt'] = np.sqrt(nc_annual_proc["pop_density"])
        
    # (Optional) Define a fragmentation index (e.g., 10% coverage)
    threshold = 0.1
    nc_annual_proc["frag_index"] = (
            (nc_annual_proc[['cropland', 'grassland', 'shrubland', 'forest']] > threshold)
            .to_array(dim='land_type')
            .sum(dim='land_type')
        ).astype(float) # Ensure it's a float for the GLM solver
            
    # Loop through the variables to check for breaks and issues in the predictors ... 
    predictors = [
        "BA_frac",
        "BA_frac_for",
        "BA_frac_nonf",
        "vpd_log",
        "ndd_log",
        "ndd_season_log",
        "sfcWind_log",
        "pr_sum_month_log",
        "GPP_log",
        "TPI",
        "VRM",
        "grassland",
        "shrubland",
        "cropland",
        "forest", 
        "HDI",
        "population",
        "pop_density",
        "frag_index",
        "grazing_pressure"
    ]
    emission_param = ["FC_total_GFED", "FC_for_GFED", "FC_nonf_GFED", "CE_total_C", "CE_nonf_C", "CE_for_C"]
        
    # Take mean over time
    nc_mean = nc_annual_proc.mean(dim="time")
    
    # Create the CSV from annual predictor brick
    ds_annual = nc_annual_proc.to_dataframe()
    ds_mean = nc_mean.to_dataframe()
    
    # Drop NAs
    ds_annual = ds_annual.dropna(subset=predictors)
    ds_annual[emission_param] = ds_annual[emission_param].fillna(0)
    ds_mean = ds_mean.dropna(subset=predictors)
    ds_mean[emission_param] = ds_mean[emission_param].fillna(0)

    # Run descriptions 
    des_annual = ds_annual.groupby("time").describe()
    des_mean = ds_mean.describe()

    # SAVE THE ANNUAL CSV OF THE GLM MODEL 
    if save == True:
        nc_annual.to_netcdf(os.path.join(glm_output_dir, "Predictors_Raw_NC_Annual.nc"))
        ds_annual.to_csv(os.path.join(glm_output_dir, "Predictors_CSV_Annual.csv"))
        nc_annual_proc.to_netcdf(os.path.join(glm_output_dir, "Predictors_NC_Annual.nc"))
        ds_mean.to_csv(os.path.join(glm_output_dir, "Predictors_CSV_Mean.csv"))
        nc_mean.to_netcdf(os.path.join(glm_output_dir, "Predictors_NC_Mean.nc"))
    
        
    ##########################################################################
    # 2. RUN GLM MODEL 
    ##########################################################################

    # Fit GLM
    formula = "BA_frac ~ vpd_log +  ndd_log + ndd_season_log + np.power(ndd_season_log, 2) + pr_sum_month_log + GPP_log + VRM + grassland + np.power(grassland, 2) +  cropland + HDI + np.power(HDI, 2) + grazing_pressure"
    model = smf.glm(
        formula=formula,
        data=ds_mean,
        family=sm.families.Binomial()
        ).fit(scale='X2')
    
    print(model.summary())
    print("ESA Model:", formula)
    
    # Compute pseudo R²
    pseudo_r2 = 1 - (model.deviance / model.null_deviance)

    # Predictions
    ds_mean["BA_frac_pred"] = model.predict(ds_mean)	

    # Aggregate predictions by grid cell
    pred_df = (
        ds_mean.groupby(["lat", "lon"])[["BA_frac", "BA_frac_pred", "BA_frac_for", "forest"]]
        .mean()
        .reset_index()
    )

    observed = pred_df["BA_frac"]
    predicted = pred_df["BA_frac_pred"]

    # Compute metrics
    rmse = compute_rmse(observed, predicted)
    nme = compute_nme(observed, predicted)
    
    print(f"✅ Model | Pseudo R² = {pseudo_r2:.3f}, RMSE = {rmse:.4f}, NME = {nme:.3f}")

    # Set the forest proportion 
    pred_df["forest_proportion"] = (pred_df["BA_frac_for"] / pred_df["BA_frac"]).fillna(0)
    pred_df["forest_proportion"] =  pred_df["forest_proportion"].where(
        np.isfinite(pred_df["forest_proportion"]), 0
    ).clip(lower=0,upper=1)
    pred_df["forest_proportion"].hist(bins=50)
    pred_df["BA_frac_for_pred"] = pred_df["BA_frac_pred"] * pred_df["forest_proportion"]    
        
    ##########################################################################
    # 3. PREDICT BURNED FRACTIONS AND AREAS ON 2001-2019
    ##########################################################################
        
    # Predict on the Annual DF (CSV) 
    annual_esa = ds_annual
    y_pred = model.predict(exog=annual_esa)
    annual_esa.loc[:, 'BA_frac_pred'] = y_pred
    
    annual_esa['BA_frac_pred'] = annual_esa['BA_frac_pred'].clip(upper=1)
    annual_esa["BA_frac"] = annual_esa["BA_frac"].clip(upper=1)
    annual_esa["BA_frac_for"] = annual_esa["BA_frac_for"].clip(upper=1)
    annual_esa["BA_frac_nonf"] = annual_esa["BA_frac_nonf"].clip(upper=1)
    
    # Compute total burnt area from fractions
    annual_esa["Grid_area"] = [grid_cell_area(lat) / 10000 for lat in annual_esa.index.get_level_values("lat")]
    print((annual_esa["Grid_area"] * annual_esa["forest"]).groupby("time").sum())
    print((annual_esa["Grid_area"] * annual_esa["grassland"]).groupby("time").sum())
    print((annual_esa["Grid_area"] * annual_esa["cropland"]).groupby("time").sum())
    
    annual_esa["BA_area_pred"] = annual_esa["BA_frac_pred"] * annual_esa["Grid_area"] 
    annual_esa["BA_area"] = annual_esa["BA_frac"] * annual_esa["Grid_area"] 
    annual_esa["BA_area_for"] = annual_esa["BA_frac_for"] * annual_esa["Grid_area"] 
    annual_esa["BA_area_nonf"] = annual_esa["BA_frac_nonf"] * annual_esa["Grid_area"] 
        
    # Compute Forest Burnt area and fraction with proportional approach
    spatial_series = pred_df.set_index(['lat', 'lon'])['forest_proportion']
    annual_esa["forest_proportion"] = annual_esa.index.droplevel('time').map(spatial_series)
    
    # Implement the forest proportion mask
    annual_esa["BA_frac_for_pred"] = annual_esa["BA_frac_pred"] * annual_esa["forest_proportion"]
    annual_esa["BA_area_for_pred"] = annual_esa["BA_frac_for_pred"] * annual_esa["Grid_area"] 
    
    # Compute the non-forest area 
    annual_esa["BA_area_nonf_pred"] = annual_esa["BA_area_pred"] - annual_esa["BA_area_for_pred"]
    annual_esa["BA_frac_nonf_pred"] = annual_esa["BA_frac_pred"] - annual_esa["BA_frac_for_pred"]
      
    # Create netcdf with predictions for figure in HPC
    annual_esa_nc = annual_esa.to_xarray().transpose("time", "lat", "lon").where(landmask.mask == 1)
    annual_esa_nc = annual_esa_nc.sortby(annual_esa_nc.lon)
    annual_esa_nc['time'] = pd.to_datetime(annual_esa_nc['time'].values)

    ##########################################################################
    # 5. PREDICT FIRE EMISSIONS USING GFED FC
    ##########################################################################
    
    # BA - Convert Mha to m2 
    annual_esa_nc["BA_area_for_pred_m2"] = annual_esa_nc["BA_area_for_pred"] * 1e10
    annual_esa_nc["BA_area_for_m2"] = annual_esa_nc["BA_area_for"] * 1e10
    annual_esa_nc["BA_area_nonf_pred_m2"] = annual_esa_nc["BA_area_nonf_pred"] * 1e10
    annual_esa_nc["BA_area_nonf_m2"] = annual_esa_nc["BA_area_nonf"] * 1e10
    annual_esa_nc["BA_area_pred_m2"] = annual_esa_nc["BA_area_pred"] * 1e10
    annual_esa_nc["BA_area_m2"] = annual_esa_nc["BA_area"] * 1e10
            
    # Multiply BA by FC 
    annual_esa_nc[f"CE_for_C_pred_GFED"] = (annual_esa_nc["BA_area_for_pred_m2"] * annual_esa_nc["FC_for_GFED"] ) / 1e12 
    annual_esa_nc[f"CE_nonf_C_pred_GFED"] = (annual_esa_nc["BA_area_nonf_pred_m2"] * annual_esa_nc["FC_nonf_GFED"] ) / 1e12 
    annual_esa_nc[f"CE_total_C_pred_GFED"] = (annual_esa_nc["BA_area_pred_m2"] * annual_esa_nc["FC_total_GFED"] ) / 1e12 
    
    # Multiply BA by FC 
    annual_esa_nc[f"CE_for_C"] = annual_esa_nc[f"CE_for_C"] / 1e12 
    annual_esa_nc[f"CE_nonf_C"] =  annual_esa_nc[f"CE_nonf_C"]/ 1e12 
    annual_esa_nc[f"CE_total_C"] =  annual_esa_nc[f"CE_total_C"] / 1e12 

    # Calculate Weighted FC for Forest -- Formula: Sum(FC * BA) / Sum(BA)
    annual_esa_nc["FC_weighted_for_obs"] = (annual_esa_nc["FC_for_GFED"] * annual_esa_nc["BA_area_for_m2"]).sum(dim=["lat", "lon"]) / \
                                           annual_esa_nc["BA_area_for_m2"].sum(dim=["lat", "lon"])
    
    annual_esa_nc["FC_weighted_for_pred"] = (annual_esa_nc["FC_for_GFED"] * annual_esa_nc["BA_area_for_pred_m2"]).sum(dim=["lat", "lon"]) / \
                                            annual_esa_nc["BA_area_for_pred_m2"].sum(dim=["lat", "lon"])
    
    # Calculate Weighted FC for Non-Forest
    annual_esa_nc["FC_weighted_nonf_obs"] = (annual_esa_nc["FC_nonf_GFED"] * annual_esa_nc["BA_area_nonf_m2"]).sum(dim=["lat", "lon"]) / \
                                            annual_esa_nc["BA_area_nonf_m2"].sum(dim=["lat", "lon"])
    
    annual_esa_nc["FC_weighted_nonf_pred"] = (annual_esa_nc["FC_nonf_GFED"] * annual_esa_nc["BA_area_nonf_pred_m2"]).sum(dim=["lat", "lon"]) / \
                                             annual_esa_nc["BA_area_nonf_pred_m2"].sum(dim=["lat", "lon"])
    
    # Calculate Weighted FC for Total Area
    annual_esa_nc["FC_weighted_total_obs"] = (annual_esa_nc["FC_total_GFED"] * annual_esa_nc["BA_area_m2"]).sum(dim=["lat", "lon"]) / \
                                             annual_esa_nc["BA_area_m2"].sum(dim=["lat", "lon"])
    
    annual_esa_nc["FC_weighted_total_pred"] = (annual_esa_nc["FC_total_GFED"] * annual_esa_nc["BA_area_pred_m2"]).sum(dim=["lat", "lon"]) / \
                                              annual_esa_nc["BA_area_pred_m2"].sum(dim=["lat", "lon"])
   
    # Merge FE to the previous dataframe
    annual_esa = annual_esa_nc.to_dataframe()
                
    # Compute mean predictions and forest proportion for simulated period 
    pred_nc = pred_df.set_index(["lat", "lon"]).to_xarray()
    pred_nc["BA_frac"] = pred_nc["BA_frac"] * 12      
    pred_nc["BA_frac_pred"] = pred_nc["BA_frac_pred"] * 12   
    pred_nc["BA_frac_for"] = pred_nc["BA_frac_for"] * 12  
    forest_prop_sim = pred_nc['forest_proportion'].expand_dims({"time": dates_sim})
    
    # Save GLM outputs
    if save == True: 
        annual_esa.to_csv(os.path.join(glm_output_dir, "Predictions_CSV_Annual.csv"))
        annual_esa_nc.to_netcdf(os.path.join(glm_output_dir, "Predictions_NC_Annual.nc"))
        
        annual_esa_sum = annual_esa.groupby(["time"]).sum()   
        annual_esa_sum.to_csv(os.path.join(glm_output_dir, "Predictions_CSV_Sum.csv"))

        pred_nc.to_netcdf(os.path.join(glm_output_dir, "Predictions_NC_Mean.nc"))
        forest_prop_sim.to_netcdf(os.path.join(glm_output_dir, f"BA_Forest_Proportion_{end_year}-2100.nc"))    

    print("GLM Model Completed. All important files saved in GLM_MODEL")        
    return model, formula 

def run_rolling_hindcast(ds_annual, formula, window_size=3):
    """
    Performs a rolling out-of-sample validation. 
    Trains the GLM on all years EXCEPT a sliding window, then predicts that window.
    """
    
    ds_annual = ds_annual.set_index(["lat", "lon", "time"])
    years = sorted(ds_annual.index.get_level_values('time').unique())
    hindcast_results = []

    print(f"\n--- Starting Rolling Hindcast ({window_size}-year windows) ---")

    # Loop through years to create sliding windows
    for i in range(len(years) - window_size + 1):
        test_years = years[i : i + window_size]
        train_years = [y for y in years if y not in test_years]

        # Split Data
        train_df = ds_annual.loc[ds_annual.index.get_level_values('time').isin(train_years)]
        test_df = ds_annual.loc[ds_annual.index.get_level_values('time').isin(test_years)]

        # Fit Model on Training set
        model = smf.glm(
            formula=formula,
            data=train_df,
            family=sm.families.Binomial()
        ).fit(scale='X2')

        # Compute pseudo R²
        pseudo_r2 = 1 - (model.deviance / model.null_deviance)

        # Predict on "Future/Unseen" Test set
        test_df = test_df.copy()
        test_df["BA_frac_pred"] = model.predict(test_df)

        # Calculate Metrics for this window
        obs = test_df["BA_frac"]
        pred = test_df["BA_frac_pred"]
        
        # Use your imported metric functions
        rmse = compute_rmse(obs, pred)
        nme = compute_nme(obs, pred)
        
        hindcast_results.append({
            "Window": f"{test_years[0]}-{test_years[-1]}",
            "Pseudo-R2": pseudo_r2,
            "RMSE": rmse,
            "NME": nme
            # "Temp_Corr": window_corr
        })
        
        print(f"Window {test_years[0]}-{test_years[-1]} | R2: {pseudo_r2:.3f} | RMSE: {rmse:.5f}")

    # Summary
    hindcast_df = pd.DataFrame(hindcast_results)
    print("\n--- HINDCAST SUMMARY ---")
    print(hindcast_df.describe())
    
    return hindcast_df


def run_full_gam_test(ds_mean):
    # 1. Define the exact columns used in your GLM
    # Note: Using the raw versions (not log) can also be interesting 
    # as the GAM will find the log-shape itself if it exists!
    feature_cols = [
        "vpd_log", "ndd_log", "ndd_season_log",
        "pr_sum_month_log", "GPP_log", "TPI", "VRM", 
        "grassland", "shrubland", "cropland", "HDI", "grazing_pressure"
    ]
    
    X = ds_mean[feature_cols].values
    y = ds_mean['BA_frac'].values

    # 2. Build the GAM terms dynamically
    # We apply a spline s() to every single variable
    terms = s(0)
    for i in range(1, len(feature_cols)):
        terms += s(i)
        
    # 3. Fit the model
    # Using 'Binomial' to match your GLM family
    gam_full = LogisticGAM(terms).fit(X, y)
    
    # 4. Extract Performance
    explained_deviance = gam_full.statistics_['pseudo_r2']['explained_deviance']
    
    print(f"--- Full GAM Stress Test ---")
    print(f"Variables included: {len(feature_cols)}")
    print(f"GAM Explained Deviance: {explained_deviance:.4f}")
    
    gam_full.summary()
    return gam_full, explained_deviance


def factorial_decomposition(model, annual_pred, option: int, save = True):


    # Create the five groups of variables 
    factor_groups = {
        "climate": [
            "vpd_log",
            "ndd_log",
            "ndd_season_log",
            # "sfcWind_log",
            "pr_sum_month_log"
        ],
        "land_use": [
            "grassland",
            # "shrubland",
            "cropland",
            "grazing_pressure"
        ],
        "vegetation": [
            "GPP_log"
        ],
        "socioeconomic": [
            "HDI"
        ]
    }
    
    decomp_BA = annual_pred[["lat", "time", "lon", "BA_area", "BA_area_pred"]].set_index(["lat", "lon", "time"])
    annual_pred = annual_pred.set_index(["lat", "lon", "time"])
    annual_pred.index = annual_pred.index.set_levels(pd.to_datetime(annual_pred.index.levels[2]), level="time")
    
    # OPTION 1. FIX ONE DRIVER ALL THE REST FREE 
    if option == 1: 

        for group_name, variables in factor_groups.items():
            print(f"Processing: {group_name}")
            # group_name = "land_use"
            # variables = ["grassland", "shrubland", "cropland", "grazing_pressure"]
            
            # Start fresh from base predictors
            pred_df = annual_pred.copy()
            
            # Mask for 2020-12-31
            time_mask = pred_df.index.get_level_values("time") == pd.Timestamp("2002-12-31")
            # time_mask.min()
            # Get 2020 values for these variables
            values_2002 = pred_df.loc[time_mask, list(variables)].copy()
            
            # Map lat/lon -> 2020 value
            idx_latlon = pred_df.index.droplevel("time")
            map_2002 = values_2002.set_index(idx_latlon[time_mask])
            
            # Apply mapping to all time steps
            for variable in variables:
                pred_df[variable] = idx_latlon.map(map_2002[variable])
                
            # Predict BAF (annual)
            # y_pred = model.predict(exog=pred_df) * 12
            y_pred = model.predict(exog=pred_df)
            pred_df[f"BAF_Fixed_{group_name}"] = y_pred
            
            # Compute grid area (ha)
            pred_df["Grid_area"] = [
                grid_cell_area(lat) / 10000
                for lat in pred_df.index.get_level_values("lat")
            ]
            
            # Convert to burned area (ha)
            pred_df[f"BA_Fixed_{group_name}"] = pred_df[f"BAF_Fixed_{group_name}"] * pred_df["Grid_area"]
            
            # Store in output
            decomp_BA[f"BA_Fixed_{group_name}"] = pred_df[f"BA_Fixed_{group_name}"]
                
        # Aggregate across lat/lon
        # decomp_BA = decomp_BA.reset_index().merge(regions_csv, how="outer", on=["lat", "lon"])
        # decomp_BA = decomp_BA.set_index(["lat", "lon", "time", "basisregions"])
        decomp_BA_agg = decomp_BA.groupby(level=["time"]).sum()
        
        # Save results
        if save == True:
            # decomp_BA.to_csv(os.path.join(bace_dir, f"BA_Factorial_Decomp_LimitingF_{scenario}_{esm}.csv"))
            decomp_BA_agg.to_csv(os.path.join(glm_output_dir, f"BA_Factorial_Decomp_LimitingF_Agg_HistGLM.csv"))
        
    
    # OPTION 2. FIX ALL ONLY ONE DRIVER FREE 
    if option == 2: 
        all_vars = [v for vars_list in factor_groups.values() for v in vars_list]
        print(all_vars)
        for group_name, variables in factor_groups.items():
            print(f"Processing: {group_name} (free driver mode)")
            # group_name = "climate"
            # variables = ["vpd_log", "ndd_log"]
            
            # Start fresh from base predictors
            pred_df = annual_pred.copy()
            
            # Mask for 2020-12-31
            time_mask = pred_df.index.get_level_values("time") == pd.Timestamp("2002-12-31")

            # --- FIX all variables NOT in this group ---
            free_vars = [v for v in all_vars if v not in variables]
            values_2002 = pred_df.loc[time_mask, free_vars].copy()
            idx_latlon = pred_df.index.droplevel("time")
            map_2002 = values_2002.set_index(idx_latlon[time_mask])
            
            # Apply mapping to all time steps
            for variable in free_vars:
                pred_df[variable] = idx_latlon.map(map_2002[variable])
                
            # Predict BAF (annual)
            # y_pred = model.predict(exog=pred_df) * 12
            y_pred = model.predict(exog=pred_df) 
            pred_df[f"BAF_Free_{group_name}"] = y_pred
            
            # Compute grid area (ha)
            pred_df["Grid_area"] = [
                grid_cell_area(lat) / 10000
                for lat in pred_df.index.get_level_values("lat")
            ]
            
            # Convert to burned area (ha)
            pred_df[f"BA_Free_{group_name}"] = pred_df[f"BAF_Free_{group_name}"] * pred_df["Grid_area"]
            
            # Store in output
            decomp_BA[f"BA_Free_{group_name}"] = pred_df[f"BA_Free_{group_name}"]
                
        # Aggregate across lat/lon
        # decomp_BA = decomp_BA.reset_index().merge(regions_csv, how="outer", on=["lat", "lon"])
        # decomp_BA = decomp_BA.set_index(["lat", "lon", "time", "basisregions"])
        decomp_BA_agg = decomp_BA.groupby(level=["time"]).sum()

        # Save results
        if save == True:
            # decomp_BA.to_csv(os.path.join(bace_dir, f"BA_Factorial_Decomp_Drivers_{scenario}_{esm}.csv"))
            decomp_BA_agg.to_csv(os.path.join(glm_output_dir, f"BA_Factorial_Decomp_Drivers_Agg_HistGLM.csv"))
    

if __name__ == "__main__":
    
    
    # Main GLM function
    model, formula = glm_run(nc_annual, save = False)
    
    # GLM out-of-sample tests
    ds_annual = pd.read_csv(os.path.join(glm_output_dir, "Predictors_CSV_Annual.csv"))
    hindcast_df = run_rolling_hindcast(ds_annual, formula, window_size=3)

    # GLM non-linear tests
    ds_mean = pd.read_csv(os.path.join(glm_output_dir, "Predictors_CSV_Mean.csv"))
    gam_model, gam_score = run_full_gam_test(ds_mean)
    gam_model.summary()
    
    # GLM factorial decomposition
    annual_esa = pd.read_csv(os.path.join(glm_output_dir, "Predictions_CSV_Annual.csv"))
    factorial_decomposition(model, annual_esa, option = 1, save = True)
    factorial_decomposition(model, annual_esa, option = 2, save = True)

