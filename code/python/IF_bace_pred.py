# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:39:22 2025

@author: theo.rouhette

"""

# Importing Needed Libraries
import os  # For navigating os
import sys  # Getting system details
import numpy as np  # Numerical / array functions
import pandas as pd  # Data functions
import xarray as xr  # Reading and manipulating NetCDF data
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Importing Common Paths/Data/Functions
from IF_setup import IF_PATH, experiment_name, landmask, grid_cell_area, regions, remove_nas

# PATHS
basd_dir = os.path.join(IF_PATH, "output/basd/W5E5v2/")
climate_dir = os.path.join(IF_PATH, "output/climate")
land_dir = os.path.join(IF_PATH, "output/land_use")
vegetation_dir = os.path.join(IF_PATH, "output/vegetation")
fc_dir = os.path.join(IF_PATH, "output/fuel_consumption")
bace_dir = os.path.join(IF_PATH, "output/fire_impacts")

# CONSTANTS
start = 2019
end = 2100

# INPUTS
pred_df = pd.read_csv(os.path.join(IF_PATH, "output/historic_glm/Predictors_CSV_Mean.csv"))    
hist_nc = xr.open_dataset(os.path.join(IF_PATH, "output/historic_glm/Predictions_NC_Annual.nc"))
regions_csv = regions.to_dataframe().reset_index()


def glm_train(pred_df, formula: str):
                        
    # Fit the model
    model = smf.glm(formula=formula[0], data=pred_df, family=sm.families.Binomial()).fit(scale='X2')
    
    # Model summary 
    print(model.summary())
  
    # Make predictions on validation set
    y_pred = model.predict(exog=pred_df)
    # pred_df['BA_frac_pred'] = y_pred
    pred_df.loc[:, 'BA_frac_pred'] = y_pred
    
    # Aggregate predictions
    pred_df_sum = pred_df.groupby(['lat', 'lon'])[['BA_frac', 'BA_frac_pred']].mean().reset_index()

    def compute_rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def compute_nme(y_true, y_pred):
        return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true))))

    # Compute pseudo R²
    pseudo_r2 = 1 - (model.deviance / model.null_deviance)

    # Calculate NME
    observed = pred_df_sum['BA_frac']
    predicted = pred_df_sum['BA_frac_pred']
    
    # Compute metrics
    rmse = compute_rmse(observed, predicted)
    nme = compute_nme(observed, predicted)

    print(f"Model done | Pseudo R² = {pseudo_r2:.3f}, RMSE = {rmse:.4f}, NME = {nme:.3f}")

    return model 
    
def glm_predict(model, regions, scenario: str, esm: str, years: list):
    
    ##########################################################################
    # 1. OPEN ALL THE PREDICTORS
    print(f"Running the GLM prediction for {scenario} and {esm}")
    
    # Open climate projections
    pr = xr.open_dataset(os.path.join(climate_dir, f"{esm}_STITCHES_W5E5v2_{scenario}_pr_global_annual_2015_{end}.nc"))
    vpd = xr.open_dataset(os.path.join(climate_dir, f"{esm}_STITCHES_W5E5v2_{scenario}_vpd_global_annual_2015_{end}.nc"))
    ndd = xr.open_dataset(os.path.join(climate_dir, f"{esm}_STITCHES_W5E5v2_{scenario}_ndd_global_annual_2015_{end}.nc"))
    wind = xr.open_dataset(os.path.join(climate_dir, f"{esm}_STITCHES_W5E5v2_{scenario}_sfcWind_global_annual_2015_{end}.nc"))
        
    # Open Demeter projections based on the ESA basemap 
    land = xr.open_dataset(os.path.join(land_dir, f"Processed_DEM_annual_{scenario}_2021-{end}_ESA_2019.nc"))    
    
    # Open the forest parameters from historic data (proportion)
    forest_prop = xr.open_dataset(os.path.join(IF_PATH, f"output/historic_glm/BA_Forest_Proportion_{start}-{end}.nc"))
    
    # Open the fuel consumption data from VPD scaling 
    fc_param = xr.open_dataset(os.path.join(fc_dir, f"FC_{scenario}_{esm}_2019-{end}_proj.nc"))
    
    # Open the dynamic GPP
    gpp = xr.open_dataset(os.path.join(vegetation_dir, f"GPP_pModel_{scenario}_{esm}_season_2015-{end}.nc"))
    gpp["GPP"] = gpp["gpp"]
    gpp = gpp.drop_vars("gpp")
        
    # Open static layers and HDI 
    tpi = xr.open_dataset(os.path.join(IF_PATH, f"input/static_layers/TPI_2019-{end}.nc"))
    vrm = xr.open_dataset(os.path.join(IF_PATH, f"input/static_layers/VRM_2019-{end}.nc"))        
    grazing = xr.open_dataset(os.path.join(IF_PATH, f"input/static_layers/Grazing_pressure_{start}-{end}.nc")).fillna(0)
    
    if "SSP1" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP1_{start}-{end}.nc"))
    if "SSP2" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP2_{start}-{end}.nc"))
    if "SSP3" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP3_{start}-{end}.nc"))
    if "SSP4" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP4_{start}-{end}.nc"))
    if "SSP5" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP5_{start}-{end}.nc"))
    if "SSP" not in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP2_{start}-{end}.nc"))
        
    # Merge NetCDF + Transform to dataframe
    pred_nc = xr.merge([hdi, tpi, vrm, gpp, land, vpd, ndd, wind, pr, grazing, fc_param])
    pred_nc = pred_nc.sel(time=pred_nc.time.dt.year > 2019) # Cover 2020 - 2100 and match the 2001-2019 historic period.
    print(pred_nc)
    pred_df = pred_nc.to_dataframe()
    pred_df["Grid_area"] = [grid_cell_area(lat) / 10000 for lat in pred_df.index.get_level_values("lat")]
    
    print("Forest", (pred_df["forest"]* pred_df["Grid_area"]).groupby("time").sum())
    print("Grassland", (pred_df["grassland"]* pred_df["Grid_area"]).groupby("time").sum())
    print("Cropland", (pred_df["cropland"]* pred_df["Grid_area"]).groupby("time").sum())
    print("COUNT:", pred_df.groupby("time").forest.count())

    ##########################################################################
    # 2. FILTER & TRANSFORM THE PREDICTORS
    
    # Delete 0s from GPP and VPD  
    pred_df = pred_df.loc[pred_df["vpd"] != 0]
    pred_df = pred_df.loc[pred_df["ndd"] != 0]
    pred_df = pred_df.loc[pred_df["sfcWind"] != 0]
    pred_df = pred_df.loc[pred_df["GPP"] > 0.1]
    
    # Perform transformations x on variables y 
    pred_df["GPP_log"] = np.log2(pred_df["GPP"]) # Important  
    pred_df["GPP_season_log"] = np.log2(pred_df["GPP_seasonality"]) # NO, INCREASES NME ... 
    pred_df["ndd_log"] = np.log2(pred_df["ndd"]) # not so skewed so no need 
    pred_df["ndd_season_log"] = np.log2(pred_df["NDD_seasonality"]) # not so skewed so no need 
    pred_df["vpd_log"] = np.log2(pred_df["vpd"])
    pred_df["sfcWind_log"] = np.log2(pred_df["sfcWind"])
    pred_df['pr_sum_month_log'] = np.log(pred_df['pr_sum_month'].clip(lower=1e-6))
    
    print("Forest", (pred_df["forest"]* pred_df["Grid_area"]).groupby("time").sum())
    print("Grassland", (pred_df["grassland"]* pred_df["Grid_area"]).groupby("time").sum())
    print("Cropland", (pred_df["cropland"]* pred_df["Grid_area"]).groupby("time").sum())
    print("COUNT:", pred_df.groupby("time").forest.count())
    
    ##########################################################################
    # 3. PREDICT TOTAL BA AND SPLIT FOREST 

    # Predict Burned Areas with the model
    y_pred = model.predict(exog=pred_df)
    pred_df.loc[:, 'BA_frac_pred'] = y_pred      
        
    # Estimate the Predicted AREA from the Frac Pred and the area per grid cell 
    pred_df["BA_area_pred"] = pred_df["BA_frac_pred"] * pred_df["Grid_area"] 
    print(f"Predicted total BA for {scenario}:", pred_df["BA_area_pred"].groupby("time").sum())

    # Save CSV with BA prediction    
    pred_df.to_csv(os.path.join(bace_dir, f"BA_Prediction_{scenario}_{esm}.csv"))
    # print(pred_df)
    
    # Read it again to avoid the index issue 
    pred_df = pd.read_csv(os.path.join(bace_dir, f"BA_Prediction_{scenario}_{esm}.csv"))
    
    # Convert back to NetCDF 
    # annual_dt_summed = pred_df.groupby(['lat', 'lon', 'time'])[['BA_frac_pred']].mean().reset_index()
    pred_df = pred_df.set_index(["lon", "lat", "time"])
    pred_netcdf = pred_df.to_xarray()
    pred_netcdf["time"] = pred_netcdf["time"].astype("datetime64[ns]")
    
    # Split total BA into Forests and Non-Forests 
        # Merge with pred_netcdf 
    pred_netcdf = xr.merge([pred_netcdf, forest_prop])
        # Compute the forest proportion following method in "glm_model.py" 
    pred_netcdf["BA_frac_for_pred"] = pred_netcdf["BA_frac_pred"] * pred_netcdf["forest_proportion"] 
    pred_netcdf["BA_area_for_pred"] = pred_netcdf["BA_frac_for_pred"] * pred_netcdf["Grid_area"] 
        # Compute the non-forest area and fraction
    pred_netcdf["BA_area_nonf_pred"] = pred_netcdf["BA_area_pred"] - pred_netcdf["BA_area_for_pred"]
    pred_netcdf["BA_frac_nonf_pred"] = pred_netcdf["BA_frac_pred"] - pred_netcdf["BA_frac_for_pred"]

    
    ##########################################################################
    # 4. ESTIMATE FIRE EMISSIONS 
    # BA - Convert Mha to m2 
    pred_netcdf["BA_area_for_pred_m2"] = pred_netcdf["BA_area_for_pred"] * 1e10
    pred_netcdf["BA_area_nonf_pred_m2"] = pred_netcdf["BA_area_nonf_pred"] * 1e10

    # GFED DYNAMIC PROJECTION
    # Convert ha to m2 and multiple by FC from GFED 
    pred_netcdf["CE_for_C_pred_GFED"] = (
        pred_netcdf["BA_area_for_pred_m2"] 
        * pred_netcdf["FC_for_GFED_proj"] ) / 1e12 # g to Tg 
    pred_netcdf["CE_nonf_C_pred_GFED"] = (
        pred_netcdf["BA_area_nonf_pred_m2"] 
        * pred_netcdf["FC_nonf_GFED_proj"] ) / 1e12 # g to Tg 
    pred_netcdf["CE_total_C_pred_GFED"] = pred_netcdf["CE_for_C_pred_GFED"].fillna(0) + pred_netcdf["CE_nonf_C_pred_GFED"].fillna(0)
    
    ##########################################################################
    # 5. SAVE THE FINAL NETCDF OF BA_FE PER SCENARIO AND ESM  
    pred_netcdf = xr.merge([pred_netcdf, regions])
    pred_netcdf.to_netcdf(os.path.join(bace_dir, f"BA_CE_Prediction_{scenario}_{esm}.nc"))
    print(f"GLM Prediction completed for {scenario} and {esm}")
    
def master_csv(hist_nc, scenarios: list, esms: list):

    ##########################################################################
    # 1.A Open the historical data (NC without predictions)    
    hist_nc["BA_frac"] *= 12
    
    # Convert to dataframe and transform/filter 
    hist_annual = hist_nc.to_dataframe() # Convert to DF + Add Land use area + Filter     
    hist_annual["Grid_area"] = [grid_cell_area(lat) / 10000 for lat in hist_annual.index.get_level_values("lat")]
    hist_annual["forest"] = hist_annual["forest"] * hist_annual["Grid_area"] 
    hist_annual["grassland"] = hist_annual["grassland"] * hist_annual["Grid_area"] 
    hist_annual["shrubland"] = hist_annual["shrubland"] * hist_annual["Grid_area"] 
    hist_annual["cropland"] = hist_annual["cropland"] * hist_annual["Grid_area"]
        
    hist_annual = hist_annual.loc[hist_annual["vpd"] != 0]
    hist_annual = hist_annual.loc[hist_annual["ndd"] != 0]
    hist_annual = hist_annual.loc[hist_annual["sfcWind"] != 0]
    hist_annual = hist_annual.loc[hist_annual["GPP"] > 1]
    
    # Compute the mean/sum value 
    hist_mean = hist_annual.groupby("time").mean()
    hist_mean["BA_area"] = hist_annual["BA_area"].groupby("time").sum()
    hist_mean["BA_area_for"] = hist_annual["BA_area_for"].groupby("time").sum()
    
    hist_mean["BA_area_pred"] = hist_annual["BA_area_pred"].groupby("time").sum()
    hist_mean["BA_area_for_pred"] = hist_annual["BA_area_for_pred"].groupby("time").sum()
    
    hist_mean["BA_area_nonf"] = hist_annual["BA_area_nonf"].groupby("time").sum()
    hist_mean["BA_area_nonf_pred"] = hist_annual["BA_area_nonf_pred"].groupby("time").sum()
    
    # Sums and weighted averages 
    hist_mean["FC_for_GFED"] = (hist_annual["FC_for_GFED"] * hist_annual["BA_area_for"]).groupby("time").sum() / hist_annual["BA_area_for"].groupby("time").sum()
    hist_mean["FC_nonf_GFED"] = (hist_annual["FC_nonf_GFED"] * hist_annual["BA_area_nonf"]).groupby("time").sum() / hist_annual["BA_area_nonf"].groupby("time").sum()
    hist_mean["FC_total_GFED"] = (hist_annual["FC_total_GFED"] * hist_annual["BA_area"]).groupby("time").sum() / hist_annual["BA_area"].groupby("time").sum()

    hist_mean["CE_for_C_pred_GFED"] = hist_annual["CE_for_C_pred_GFED"].groupby("time").sum() 
    hist_mean["CE_for_C"] = hist_annual["CE_for_C"].groupby("time").sum() 

    hist_mean["CE_nonf_C_pred_GFED"] = hist_annual["CE_nonf_C_pred_GFED"].groupby("time").sum()
    hist_mean["CE_nonf_C"] = hist_annual["CE_nonf_C"].groupby("time").sum() 

    hist_mean["CE_total_C_pred_GFED"] = hist_annual["CE_total_C_pred_GFED"].groupby("time").sum() 
    hist_mean["CE_total_C"] = hist_annual["CE_total_C"].groupby("time").sum()

    hist_mean["forest"] = hist_annual["forest"].groupby("time").sum()
    hist_mean["grassland"] = hist_annual["grassland"].groupby("time").sum()
    hist_mean["shrubland"] = hist_annual["shrubland"].groupby("time").sum() 
    hist_mean["cropland"] = hist_annual["cropland"].groupby("time").sum()
    # hist_mean["pasture"] = hist_annual["pasture"].groupby("time").sum()
    hist_mean["Source"] = "Historic"

    ###########################################################################
    # Open the files of BA projections 
    master_df = []
    master_nc = []
    for scenario in scenarios: 
        for esm in esms: 
            print(f"Processing {scenario} - {esm}")
            pred_nc = xr.open_dataset(os.path.join(bace_dir, f"BA_CE_Prediction_{scenario}_{esm}.nc"))
            pred_nc["forest"] = pred_nc["forest"] * pred_nc["Grid_area"] 
            pred_nc["grassland"] = pred_nc["grassland"] * pred_nc["Grid_area"] 
            pred_nc["shrubland"] = pred_nc["shrubland"] * pred_nc["Grid_area"] 
            pred_nc["cropland"] = pred_nc["cropland"] * pred_nc["Grid_area"]
            # pred_nc["FC_total_GFED_proj"] = pred_nc["FC_for_GFED_proj"] + pred_nc["FC_nonf_GFED_proj"]
            
            pred_mean = pred_nc.mean(dim=["lat", "lon"])
            pred_mean["BA_area_pred"] = pred_nc["BA_area_pred"].sum(dim=["lat", "lon"])
            pred_mean["BA_area_for_pred"] = pred_nc["BA_area_for_pred"].sum(dim=["lat", "lon"])
            pred_mean["BA_area_nonf_pred"] = pred_nc["BA_area_nonf_pred"].sum(dim=["lat", "lon"])
            
            pred_mean["FC_for_GFED_proj"] = (pred_nc["FC_for_GFED_proj"] * pred_nc["BA_area_for_pred"]).sum(dim=["lat", "lon"]) / pred_nc["BA_area_for_pred"].sum(dim=["lat", "lon"])
            pred_mean["FC_nonf_GFED_proj"] = (pred_nc["FC_nonf_GFED_proj"] * pred_nc["BA_area_nonf_pred"]).sum(dim=["lat", "lon"]) / pred_nc["BA_area_nonf_pred"].sum(dim=["lat", "lon"])
            pred_mean["FC_total_GFED_proj"] = (pred_nc["FC_total_GFED_proj"] * pred_nc["BA_area_pred"]).sum(dim=["lat", "lon"]) / pred_nc["BA_area_pred"].sum(dim=["lat", "lon"])

            pred_mean["CE_for_C_pred_GFED"] = pred_nc["CE_for_C_pred_GFED"].sum(dim=["lat", "lon"]) 
            pred_mean["CE_nonf_C_pred_GFED"] = pred_nc["CE_nonf_C_pred_GFED"].sum(dim=["lat", "lon"]) 
            pred_mean["CE_total_C_pred_GFED"] = pred_nc["CE_total_C_pred_GFED"].sum(dim=["lat", "lon"]) 
            
            pred_mean["forest"] = pred_nc["forest"].sum(dim=["lat", "lon"])
            pred_mean["grassland"] = pred_nc["grassland"].sum(dim=["lat", "lon"])
            pred_mean["shrubland"] = pred_nc["shrubland"].sum(dim=["lat", "lon"])
            pred_mean["cropland"] = pred_nc["cropland"].sum(dim=["lat", "lon"])   
            # pred_mean["pasture"] = pred_nc["pasture"].sum(dim=["lat", "lon"])   
           
            pred_nc = pred_nc[["basisregions",
                                "BA_area_pred", "BA_area_for_pred", "BA_area_nonf_pred", 
                                "BA_frac_pred", "BA_frac_for_pred", "BA_frac_nonf_pred", 
                                "FC_total_GFED_proj", "FC_for_GFED_proj", "FC_nonf_GFED_proj",
                                "CE_for_C_pred_GFED", "CE_nonf_C_pred_GFED", "CE_total_C_pred_GFED"]]
            master_nc.append(pred_nc)
            
            pred_df = pred_mean.to_dataframe()
            pred_df["Source"] = f"{scenario} - {esm}"
            pred_df["Scenario"] = f"{scenario}"
            pred_df["ESM"] = f"{esm}"

            master_df.append(pred_df)
            
    # Concat projections and historical data and save 
    master_df = pd.concat(master_df)
    master_full = [hist_mean, master_df]
    time_series = pd.concat(master_full)
    # master_nc = xr.concat(master_nc, dim = "scen_esm") 
    # Select final results 
    # master_nc = master_nc[["basisregions",
    #                         "BA_area_pred", "BA_area_for_pred", "BA_area_nonf_pred", 
    #                        "BA_frac_pred", "BA_frac_for_pred", "BA_frac_nonf_pred", 
    #                        "FE_for_C_pred_ECMWF", "FE_nonf_C_pred_ECMWF", "FE_total_C_pred_ECMWF"]]
    print(master_nc)
    print(time_series)    
    
    if "MME" in esms: 
        time_series.to_csv(os.path.join(bace_dir, f"BA_CE_Prediction_AllScen_MME.csv"))
        # master_nc.to_netcdf(os.path.join(bace_dir, f"BA_CE_Prediction_AllScen_MME.nc"))
    else:
        time_series.to_csv(os.path.join(bace_dir, f"BA_CE_Prediction_AllScen.csv"))
        # master_nc.to_netcdf(os.path.join(bace_dir, f"BA_CE_Prediction_AllScen.nc"))

def regional_csv(hist_nc, regions, scenarios: list, esms: list):

    ##########################################################################
    # 1.A Open the historical data (NC without predictions)
    
    # Open the GFED regions
    region_ids = np.unique(regions['basisregions'].values.ravel())
    region_ids = region_ids[~np.isnan(region_ids)]  # Remove any NaNs
    region_ids = region_ids.astype(int)
    region_ids = region_ids[region_ids > 0]  # Skip 0 if it means "no region"
    print(region_ids)
    region_ids = np.unique(region_ids)
    print(len(region_ids))
    print("Unique region IDs after rounding:", region_ids)
    
    hist_nc["BA_frac"] *= 12
    
    # Convert to dataframe and transform/filter 
    hist_annual = hist_nc.to_dataframe() # Convert to DF + Add Land use area + Filter     
    hist_annual["Grid_area"] = [grid_cell_area(lat) / 10000 for lat in hist_annual.index.get_level_values("lat")]
    hist_annual["forest"] = hist_annual["forest"] * hist_annual["Grid_area"] 
    hist_annual["grassland"] = hist_annual["grassland"] * hist_annual["Grid_area"] 
    hist_annual["shrubland"] = hist_annual["shrubland"] * hist_annual["Grid_area"] 
    hist_annual["cropland"] = hist_annual["cropland"] * hist_annual["Grid_area"]
    hist_annual = hist_annual.loc[hist_annual["vpd"] != 0]
    hist_annual = hist_annual.loc[hist_annual["ndd"] != 0]
    hist_annual = hist_annual.loc[hist_annual["sfcWind"] != 0]
    hist_annual = hist_annual.loc[hist_annual["GPP"] > 1]
        
    # Compute the mean/sum value 
    hist_annual = hist_annual.reset_index().set_index(["time", "lat", "lon", "basisregions"])
    hist_mean = hist_annual.groupby(["basisregions", "time"]).mean()
    hist_mean["BA_area"] = hist_annual["BA_area"].groupby(["basisregions", "time"]).sum()
    hist_mean["BA_area_for"] = hist_annual["BA_area_for"].groupby(["basisregions", "time"]).sum()
    hist_mean["BA_area_pred"] = hist_annual["BA_area_pred"].groupby(["basisregions", "time"]).sum()
    hist_mean["BA_area_for_pred"] = hist_annual["BA_area_for_pred"].groupby(["basisregions", "time"]).sum()
    hist_mean["BA_area_nonf"] = hist_annual["BA_area_nonf"].groupby(["basisregions", "time"]).sum()
    hist_mean["BA_area_nonf_pred"] = hist_annual["BA_area_nonf_pred"].groupby(["basisregions", "time"]).sum()
    
    # Sums and weighted averages 
    hist_mean["FC_total_GFED"] = (hist_annual["FC_total_GFED"] * hist_annual["BA_area"]).groupby(["basisregions", "time"]).sum() / hist_annual["BA_area"].groupby(["basisregions", "time"]).sum()
    hist_mean["FC_for_GFED"] = (hist_annual["FC_for_GFED"] * hist_annual["BA_area_for"]).groupby(["basisregions", "time"]).sum() / hist_annual["BA_area_for"].groupby(["basisregions", "time"]).sum()
    hist_mean["FC_nonf_GFED"] = (hist_annual["FC_nonf_GFED"] * hist_annual["BA_area_nonf"]).groupby(["basisregions", "time"]).sum() / hist_annual["BA_area_nonf"].groupby(["basisregions", "time"]).sum()
    
    hist_mean["CE_for_C_pred_GFED"] = hist_annual["CE_for_C_pred_GFED"].groupby(["basisregions", "time"]).sum() 
    hist_mean["CE_nonf_C_pred_GFED"] = hist_annual["CE_nonf_C_pred_GFED"].groupby(["basisregions", "time"]).sum() 
    hist_mean["CE_total_C_pred_GFED"] = hist_annual["CE_total_C_pred_GFED"].groupby(["basisregions", "time"]).sum()  
    
    hist_mean["CE_for_C"] = hist_annual["CE_for_C"].groupby(["basisregions", "time"]).sum() 
    hist_mean["CE_nonf_C"] = hist_annual["CE_nonf_C"].groupby(["basisregions", "time"]).sum()
    hist_mean["CE_total_C"] = hist_annual["CE_total_C"].groupby(["basisregions", "time"]).sum() 

    hist_mean["forest"] = hist_annual["forest"].groupby(["basisregions", "time"]).sum()
    hist_mean["grassland"] = hist_annual["grassland"].groupby(["basisregions", "time"]).sum()
    hist_mean["shrubland"] = hist_annual["shrubland"].groupby(["basisregions", "time"]).sum() 
    hist_mean["cropland"] = hist_annual["cropland"].groupby(["basisregions", "time"]).sum()
    # hist_mean["pasture"] = hist_annual["pasture"].groupby(["basisregions", "time"]).sum()
    hist_mean["Source"] = "Historic"
    hist_mean = hist_mean.reset_index()
    print("Historic mean processed")

    hist_mean["FC_total_GFED"].plot()

    ###########################################################################
    # Open the files of BA projections 
    master_df = []    
    for scenario in scenarios: 
        scenario_df = []
        for esm in esms: 
            pred_nc = xr.open_dataset(os.path.join(bace_dir, f"BA_CE_Prediction_{scenario}_{esm}.nc"))
            
            pred_nc["forest"] = pred_nc["forest"] * pred_nc["Grid_area"] 
            pred_nc["grassland"] = pred_nc["grassland"] * pred_nc["Grid_area"] 
            pred_nc["shrubland"] = pred_nc["shrubland"] * pred_nc["Grid_area"] 
            pred_nc["cropland"] = pred_nc["cropland"] * pred_nc["Grid_area"]
            pred_nc["FC_total_GFED_proj"] = pred_nc["FC_for_GFED_proj"] + pred_nc["FC_nonf_GFED_proj"]

            print(f"Processing {scenario} - {esm}")
            # Dictionary to store results
            region_df = []
            for region_id in region_ids:
                # region_id = 1
                print(f"Processing {region_id}")
                pred_filter = pred_nc.where(pred_nc["basisregions"] == region_id) #  Filter per region      
                
                pred_mean = pred_filter.mean(dim=["lat", "lon"])
                pred_mean["BA_area_pred"] = pred_filter["BA_area_pred"].sum(dim=["lat", "lon"])
                pred_mean["BA_area_for_pred"] = pred_filter["BA_area_for_pred"].sum(dim=["lat", "lon"])
                pred_mean["BA_area_nonf_pred"] = pred_filter["BA_area_nonf_pred"].sum(dim=["lat", "lon"])
                
                pred_mean["FC_for_GFED_proj"] = (pred_filter["FC_for_GFED_proj"] * pred_filter["BA_area_for_pred"]).sum(dim=["lat", "lon"]) / pred_filter["BA_area_for_pred"].sum(dim=["lat", "lon"])
                pred_mean["FC_nonf_GFED_proj"] = (pred_filter["FC_nonf_GFED_proj"] * pred_filter["BA_area_nonf_pred"]).sum(dim=["lat", "lon"]) / pred_filter["BA_area_nonf_pred"].sum(dim=["lat", "lon"])
                pred_mean["FC_total_GFED_proj"] = (pred_filter["FC_total_GFED_proj"] * pred_filter["BA_area_pred"]).sum(dim=["lat", "lon"]) / pred_filter["BA_area_pred"].sum(dim=["lat", "lon"])
                
                pred_mean["CE_for_C_pred_GFED"] = pred_filter["CE_for_C_pred_GFED"].sum(dim=["lat", "lon"]) 
                pred_mean["CE_nonf_C_pred_GFED"] = pred_filter["CE_nonf_C_pred_GFED"].sum(dim=["lat", "lon"]) 
                pred_mean["CE_total_C_pred_GFED"] = pred_filter["CE_total_C_pred_GFED"].sum(dim=["lat", "lon"])
                
                pred_mean["forest"] = pred_filter["forest"].sum(dim=["lat", "lon"])
                pred_mean["grassland"] = pred_filter["grassland"].sum(dim=["lat", "lon"])
                pred_mean["shrubland"] = pred_filter["shrubland"].sum(dim=["lat", "lon"])
                pred_mean["cropland"] = pred_filter["cropland"].sum(dim=["lat", "lon"])   
                # pred_mean["pasture"] = pred_filter["pasture"].sum(dim=["lat", "lon"])   
                
                pred_df = pred_mean.to_dataframe()
                pred_df["Source"] = f"{scenario} - {esm}"
                pred_df["Scenario"] = f"{scenario}"
                pred_df["ESM"] = f"{esm}"
                pred_df["basisregions"] = f"{region_id}"
                pred_df = pred_df.reset_index()
                
                region_df.append(pred_df)
                
            # Concat across regions for this esm
            esm_df = pd.concat(region_df)
            scenario_df.append(esm_df)
        
        # Concat across esms for this scenario
        scenario_df = pd.concat(scenario_df)
        master_df.append(scenario_df)
            
    # Concat projections and historical data and save 
    master_df = pd.concat(master_df)
    master_full = [hist_mean, master_df]
    time_series = pd.concat(master_full)

    print(time_series)    
    
    if "MME" in esms: 
        time_series.to_csv(os.path.join(bace_dir, f"BA_CE_Prediction_AllScen_Reg_MME.csv"))
    else:
        time_series.to_csv(os.path.join(bace_dir, f"BA_CE_Prediction_AllScen_Reg.csv"))
  
def factorial_decomp(model, regions, scenario: str, esm: str, years: list, option: int, save = True):
    
    ##########################################################################
    # 1. OPEN ALL THE PREDICTORS
    print(f"Running the Factorial Decomposition for {scenario} and {esm}")
    
    # Open climate projections
    pr = xr.open_dataset(os.path.join(climate_dir, f"{esm}_STITCHES_W5E5v2_{scenario}_pr_global_annual_2015_{end}.nc"))
    vpd = xr.open_dataset(os.path.join(climate_dir, f"{esm}_STITCHES_W5E5v2_{scenario}_vpd_global_annual_2015_{end}.nc"))
    ndd = xr.open_dataset(os.path.join(climate_dir, f"{esm}_STITCHES_W5E5v2_{scenario}_ndd_global_annual_2015_{end}.nc"))
    wind = xr.open_dataset(os.path.join(climate_dir, f"{esm}_STITCHES_W5E5v2_{scenario}_sfcWind_global_annual_2015_{end}.nc"))
        
    # Open Demeter projections based on the ESA basemap 
    land = xr.open_dataset(os.path.join(land_dir, f"Processed_DEM_annual_{scenario}_2021-{end}_ESA_2019.nc"))    
    
    # Open the forest parameters from historic data (proportion)
    forest_prop = xr.open_dataset(os.path.join(IF_PATH, f"output/historic_glm/BA_Forest_Proportion_{start}-{end}.nc"))
    
    # Open the fuel consumption data from VPD scaling 
    fc_param = xr.open_dataset(os.path.join(fc_dir, f"FC_{scenario}_{esm}_2019-{end}_proj.nc"))
    
    # Open the dynamic GPP
    gpp = xr.open_dataset(os.path.join(vegetation_dir, f"GPP_pModel_{scenario}_{esm}_season_2015-{end}.nc"))
    gpp["GPP"] = gpp["gpp"]
    gpp = gpp.drop_vars("gpp")
        
    # Open static layers and HDI 
    tpi = xr.open_dataset(os.path.join(IF_PATH, f"input/static_layers/TPI_2019-{end}.nc"))
    vrm = xr.open_dataset(os.path.join(IF_PATH, f"input/static_layers/VRM_2019-{end}.nc"))        
    grazing = xr.open_dataset(os.path.join(IF_PATH, f"input/static_layers/Grazing_pressure_{start}-{end}.nc")).fillna(0)
    
    if "SSP1" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP1_{start}-{end}.nc"))
    if "SSP2" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP2_{start}-{end}.nc"))
    if "SSP3" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP3_{start}-{end}.nc"))
    if "SSP4" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP4_{start}-{end}.nc"))
    if "SSP5" in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP5_{start}-{end}.nc"))
    if "SSP" not in scenario: 
        hdi = xr.open_dataset(os.path.join(IF_PATH, f"input/human_development/HDI_SSP2_{start}-{end}.nc"))
        
    # Merge NetCDF + Transform to dataframe
    pred_nc = xr.merge([hdi, tpi, vrm, gpp, land, vpd, ndd, wind, pr, fc_param])
    pred_nc = pred_nc.sel(time=pred_nc.time.dt.year > 2019) 
    pred_df = pred_nc.to_dataframe()
    
    ##########################################################################
    # 2. PROCESS/TRANSFORM ALL THE PREDICTORS
    
    # Filter and transform the variables  (equal to the original DF) 
    
    # Delete 0s from GPP and VPD  
    pred_df = pred_df.loc[pred_df["vpd"] != 0]
    pred_df = pred_df.loc[pred_df["ndd"] != 0]
    pred_df = pred_df.loc[pred_df["sfcWind"] != 0]
    pred_df = pred_df.loc[pred_df["GPP"] > 1]
    # print(pred_df)
    
    # Perform transformations x on variables y 
    pred_df["GPP_log"] = np.log2(pred_df["GPP"]) # Important  
    pred_df["GPP_season_log"] = np.log2(pred_df["GPP_seasonality"]) # NO, INCREASES NME ... 
    pred_df["ndd_log"] = np.log2(pred_df["ndd"]) # not so skewed so no need 
    pred_df["ndd_season_log"] = np.log2(pred_df["NDD_seasonality"]) # not so skewed so no need 
    pred_df["vpd_log"] = np.log2(pred_df["vpd"])
    pred_df["sfcWind_log"] = np.log2(pred_df["sfcWind"])
    pred_df["pr_sum_month_log"] = np.log2(pred_df["pr_sum_month"])

    ##########################################################################
    # 3. FIX THE FACTORS AND RUN BA PREDICT PER GROUP 
    '''
    The goal here is to have a new total BA projection per scenario-esm with fixed factors
    I can fix groups of factors: climate (VPD, NDD, Wind) / land-use (all of them) / vegetation (GPP) / topographic (TPI, VRM) / socio-economic (HDI) 
    The first year of my projection is 2015, so copy-paste 2015 values all the way to 2100
    Hence, for each scenario-esm I will have BA for the 5 groups and the default 
    Main output should be a trend figure and maps of main factors (that would be great but more challenging)
    '''

    # Create the five groups of variables 
    factor_groups = {
        "climate": [
            "vpd_log",
            "ndd_log",
            "ndd_season_log",
            "sfcWind_log",
            "pr_sum_month_log"
        ],
        "land_use": [
            "grassland",
            "shrubland",
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
    
    # Read default BA and predictors
    pred_nc = xr.open_dataset(os.path.join(bace_dir, f"BA_CE_Prediction_{scenario}_{esm}.nc"))
    
    pred_df_base = pred_nc.to_dataframe()
    decomp_BA = pd.DataFrame(index=pred_df_base.index)

    # OPTION 1. FIX ONE DRIVER ALL THE REST FREE 
    if option == 1: 

        for group_name, variables in factor_groups.items():
            print(f"Processing: {group_name}")
            # group_name = "climate"
            # variables = ["vpd_log", "ndd_log"]
            
            # Start fresh from base predictors
            pred_df = pred_df_base.copy()
            
            # Mask for 2020-12-31
            time_mask = pred_df.index.get_level_values("time") == pd.Timestamp("2020-12-31")
            
            # Get 2020 values for these variables
            values_2020 = pred_df.loc[time_mask, list(variables)].copy()
            
            # Map lat/lon -> 2020 value
            idx_latlon = pred_df.index.droplevel("time")
            map_2020 = values_2020.set_index(idx_latlon[time_mask])
            
            # Apply mapping to all time steps
            for variable in variables:
                
                pred_df[variable] = idx_latlon.map(map_2020[variable])
                
                
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
        
        # Add default BA
        decomp_BA["BA_area_pred"] = pred_df_base["BA_area_pred"]
        
        # Aggregate across lat/lon
        decomp_BA = decomp_BA.merge(regions_csv, how="outer", on=["lat", "lon"])
        decomp_BA = decomp_BA.set_index(["lat", "lon", "time", "basisregions"])
        decomp_BA_agg = decomp_BA.groupby(level=["time", "basisregions"]).sum()
        
        # Save results
        if save == True:
            # decomp_BA.to_csv(os.path.join(bace_dir, f"BA_Factorial_Decomp_LimitingF_{scenario}_{esm}.csv"))
            decomp_BA_agg.to_csv(os.path.join(bace_dir, f"BA_Factorial_Decomp_LimitingF_Agg_{scenario}_{esm}.csv"))
        
    
    # OPTION 2. FIX ALL ONLY ONE DRIVER FREE 
    if option == 2: 
        all_vars = [v for vars_list in factor_groups.values() for v in vars_list]
        print(all_vars)
        for group_name, variables in factor_groups.items():
            print(f"Processing: {group_name} (free driver mode)")
            # group_name = "climate"
            # variables = ["vpd_log", "ndd_log"]
            
            # Start fresh from base predictors
            pred_df = pred_df_base.copy()
            
            # Mask for 2020-12-31
            time_mask = pred_df.index.get_level_values("time") == pd.Timestamp("2020-12-31")

            # --- FIX all variables NOT in this group ---
            free_vars = [v for v in all_vars if v not in variables]
            values_2020 = pred_df.loc[time_mask, free_vars].copy()
            idx_latlon = pred_df.index.droplevel("time")
            map_2020 = values_2020.set_index(idx_latlon[time_mask])
            
            # Apply mapping to all time steps
            for variable in free_vars:

                
                pred_df[variable] = idx_latlon.map(map_2020[variable])
                

                
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
        
        # Add default BA
        decomp_BA["BA_area_pred"] = pred_df_base["BA_area_pred"]
        
        # Aggregate across lat/lon
        decomp_BA = decomp_BA.merge(regions_csv, how="outer", on=["lat", "lon"])
        decomp_BA = decomp_BA.set_index(["lat", "lon", "time", "basisregions"])
        decomp_BA_agg = decomp_BA.groupby(level=["time", "basisregions"]).sum()

        # Save results
        if save == True:
            # decomp_BA.to_csv(os.path.join(bace_dir, f"BA_Factorial_Decomp_Drivers_{scenario}_{esm}.csv"))
            decomp_BA_agg.to_csv(os.path.join(bace_dir, f"BA_Factorial_Decomp_Drivers_Agg_{scenario}_{esm}.csv"))

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
    
    start, end = run_manager_df['application_period'].iloc[0].split('-')
    start = 2019
    end = 2100
    dates_pred = list(range(start, end + 1, 5))    
    dates_sim = slice(f"{start}-01-31", f"{end}-12-31")
    dates_sim_df = pd.date_range(start=f"{start}-01-31", end=f"{end}-12-31", freq='ME') 
    print("Runs are prepared, processing with BA predictions")
    
    ###########################################################################
    # STEP 1. Train the Model on the historical DF 
    ###########################################################################

    formula = "BA_frac ~ vpd_log + ndd_log + ndd_season_log + sfcWind_log + pr_sum_month_log + GPP_log + TPI + VRM + HDI + grazing_pressure + grassland + shrubland + cropland", # HAAS + HDI 
    print("The GLM is", formula)
    # Open all the predictors and the BA - CSV from the server
    model = glm_train(pred_df, formula)
    print("STEP 1. GLM Training completed")
    
    # #########################################################################
    # STEP 2. Estimate BA and CE per Scenario and ESM 
    # #########################################################################

    # Iterate through each requested ESM and Experiment/Scenario
    for i, scenario in enumerate(scenarios):
        for j, esm in enumerate(esms): 
            glm_predict(model, regions, scenario, esm, dates_pred)
        # glm_predict(model, scenario, "MME", dates_pred, glm_lu_source)
    print("STEP 2. BA Prediction Completed")
    
    # #########################################################################
    # STEP 3. Create the final products (CSV) for tables and figures    
    # #########################################################################
    
    # STEP 3.A Create the aggregated files and figures (paper-ready) - GLOBAL
    master_csv(hist_nc, scenarios, esms)
    print("STEP 3.A BAFE Final Results Global Completed + Saved")
    
    # STEP 3.B Create the aggregated files and figures (paper-ready) - REGION
    regional_csv(hist_nc, regions, scenarios, esms)
    print("STEP 3.B BAFE Final Results Regional Completed + Saved")

    ###########################################################################
    # STEP 4. Factorial Decomposition
    ###########################################################################
    for i, scenario in enumerate(scenarios):
        for j, esm in enumerate(esms):
            factorial_decomp(model, regions, scenario, esm, dates_pred, option=1, save = True) # FIX ONE ALL THE REST FREE
            factorial_decomp(model, regions, scenario, esm, dates_pred, option=2, save = True) # FIX ALL ONLY ONE DRIVER FREE
        # factorial_decomp(model, scenario, "MME", dates_pred)
    
    print("STEP 4. BA Factor Decomposition Completed + Saved")






