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

# Importing Common Paths/Data/Functions
from IF_setup import IF_PATH, landmask, regions, grid_cell_area, compute_rmse, compute_nme

# CONSTANTS 
start_year = 2002
end_year = 2019
dates_obs = slice(f"{start_year}-01-31", f"{end_year}-12-31")
dates_obs_df = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='YE')  # 'ME' means Month Start
dates_sim = pd.date_range(start=f"{end_year}-01-31", end=f"2100-12-31", freq='YE')  # 'ME' means Month Start
variable = ['tas', 'pr', 'hurs', "sfcWind", "rsds"]

# INPUTS
nc_annual = xr.open_dataset(os.path.join(IF_PATH, "input/historic_glm/GLM_Historic_Inputs.nc")) # HAS ANNUAL TOTAL IN FRAC 

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
    nc_annual_proc["GPP_log"] = np.log2(nc_annual_proc["GPP"]) # Important  
    nc_annual_proc["GPP_season_log"] = np.log2(nc_annual_proc["GPP_seasonality"]) # NO, INCREASES NME ... 
    nc_annual_proc["ndd_log"] = np.log2(nc_annual_proc["ndd"]) # not so skewed so no need 
    nc_annual_proc["ndd_season_log"] = np.log2(nc_annual_proc["NDD_seasonality"]) # not so skewed so no need 
    nc_annual_proc["vpd_log"] = np.log2(nc_annual_proc["vpd"])
    nc_annual_proc["sfcWind_log"] = np.log2(nc_annual_proc["sfcWind"])
    nc_annual_proc['pr_sum_month_log'] = np.log(nc_annual_proc['pr_sum_month'].clip(min=1e-6)) 
    nc_annual_proc['pop_sqrt'] = np.sqrt(nc_annual_proc["pop_density"])
    # nc_annual_proc["biome"] = nc_annual_proc["biome"].astype("category")
    
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

    # OPTIONAL: Check trends of global sums of predictors 
    # group by year (change "year" to your time column)
    # grouped = ds_annual.groupby("time")
    
    # for var in predictors:
    #     # compute mean (or replace with grouped[var].sum())
    #     temporal_series = grouped[var].sum()
    
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(temporal_series.index,
    #              temporal_series.values,
    #              marker='o')
    
    #     plt.title(f"Temporal Trend of {var}")
    #     plt.xlabel("Year")
    #     plt.ylabel(f"Mean {var}")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()


    # SAVE THE ANNUAL CSV OF THE GLM MODEL 
    if save == True:
        nc_annual.to_netcdf(os.path.join(glm_output_dir, "Predictors_Raw_NC_Annual.nc"))
        ds_annual.to_csv(os.path.join(glm_output_dir, "Predictors_CSV_Annual.csv"))
        nc_annual_proc.to_netcdf(os.path.join(glm_output_dir, "Predictors_NC_Annual.nc"))
        ds_mean.to_csv(os.path.join(glm_output_dir, "Predictors_CSV_Mean.csv"))
        nc_mean.to_netcdf(os.path.join(glm_output_dir, "Predictors_NC_Mean.nc"))
    
        
    ##########################################################################
    # 2. RUN EXPLORATORY GLM MODELS
    ##########################################################################

    formulas = {
        "HDI_only": (
        "BA_frac ~ vpd_log + ndd_log + ndd_season_log + sfcWind_log + pr_sum_month_log + GPP_log + TPI + VRM + grassland + shrubland + cropland + HDI"
        ),
        "HDI_grazing": (
        "BA_frac ~ vpd_log + ndd_log + ndd_season_log + sfcWind_log + pr_sum_month_log + GPP_log + TPI + VRM + grassland + shrubland + cropland + HDI + grazing_pressure"
        ),
        "grazing_only": (
        "BA_frac ~ vpd_log + ndd_log + ndd_season_log + sfcWind_log + pr_sum_month_log + GPP_log + TPI + VRM + grassland + shrubland + cropland + grazing_pressure"
        ),
        }
    
    results = []

    for name, formula in formulas.items():
        print(f"\n--- Running model: {name} ---")
        # Fit GLM
        model = smf.glm(
            formula=formula,
            data=ds_mean,
            family=sm.families.Binomial()
            ).fit(scale='X2')
    
            
        # Compute pseudo R²
        pseudo_r2 = 1 - (model.deviance / model.null_deviance)
    
        # Predictions
        ds_mean["BA_frac_pred"] = model.predict(ds_mean)
    
        # Aggregate predictions by grid cell
        aggregated = (
            ds_mean.groupby(["lat", "lon"])[["BA_frac", "BA_frac_pred"]]
            .mean()
            .reset_index()
        )
    
        observed = aggregated["BA_frac"]
        predicted = aggregated["BA_frac_pred"]
    
        # Compute metrics
        rmse = compute_rmse(observed, predicted)
        nme = compute_nme(observed, predicted)
        
        print(f"Model {name} done | Pseudo R² = {pseudo_r2:.3f}, RMSE = {rmse:.4f}, NME = {nme:.3f}")
    
        # Store results
        results.append({
            "Model": name,
            "Pseudo_R2": pseudo_r2,
            "RMSE": rmse,
            "NME": nme
        })
        
    results_df = (
        pd.DataFrame(results)
        .sort_values(by="Pseudo_R2", ascending=False)
        .reset_index(drop=True)
        )
    
    print("MODEL PERFORMANCE SUMMARY")    
    print(results_df.to_string(index=False))

    
    ##########################################################################
    # 3. RUN FINAL MODEL 
    ##########################################################################

    # 3.A. GLM MODEL (complete formula: HDI + grazing)
    formula = "BA_frac ~ vpd_log + ndd_log + ndd_season_log + sfcWind_log + pr_sum_month_log + GPP_log + TPI + VRM + grassland + shrubland + cropland + HDI + grazing_pressure"
    
    # Fit GLM
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
        ds_mean.groupby(["lat", "lon"])[["BA_frac", "BA_frac_pred", "BA_frac_for"]]
        .mean()
        .reset_index()
    )

    observed = pred_df["BA_frac"]
    predicted = pred_df["BA_frac_pred"]

    # Compute metrics
    rmse = compute_rmse(observed, predicted)
    nme = compute_nme(observed, predicted)
    
    print(f"✅ Model | Pseudo R² = {pseudo_r2:.3f}, RMSE = {rmse:.4f}, NME = {nme:.3f}")

    #########################################################################
    # 3.B. ADD FOREST PROPORTION
    pred_df["forest_proportion"] = (pred_df["BA_frac_for"] / pred_df["BA_frac"]).fillna(0)
    pred_df["forest_proportion"] =  pred_df["forest_proportion"].where(
        np.isfinite(pred_df["forest_proportion"]), 0
    ).clip(lower=0,upper=1)
    pred_df["forest_proportion"].hist(bins=50)
    pred_df["BA_frac_for_pred"] = pred_df["BA_frac_pred"] * pred_df["forest_proportion"]
    # annual_dt_summed = pred_df.groupby(['lat', 'lon'])[['BA_frac', 'BA_frac_for', 'BA_frac_pred', "BA_frac_for_pred"]].mean().reset_index()

    #########################################################################
    # # 3.C OPTIONAL: RUN PERFORMANCE TESTS
    
    # # 1. Variance Inflation Factor (VIF) 
    # dftypes = ds_mean.dtypes
    # X = ds_mean[['GPP_log', "HDI", "grazing_pressure", 'TPI', 'VRM', 'grassland', 'shrubland', 'cropland', 'ndd_log', 'vpd_log', "pr_sum_month_log", 'sfcWind_log' , 'ndd_season_log']]
    # X = add_constant(X)
    # X = X.dropna()
    # # VIF dataframe
    # vif_data = pd.DataFrame()
    # vif_data["feature"] = X.columns
    # # calculating VIF for each feature
    # vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    # print(vif_data)

    # # 2. Correlation Heat Map  
    # corr = X.corr().abs()
    # high_corr = [col for col in corr.columns if any(corr[col] > 0.95)]
    # print(high_corr)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={'label': 'Absolute correlation'})
    # plt.title("Correlation Heatmap of merge_mean Variables", fontsize=14, pad=12)
    # plt.tight_layout()
    # plt.show()
    
    # #3. T-values plots
    # t_values = model.tvalues
    # coeff_names = t_values.index
    # t_vals = t_values.values
    # sorted_idx = np.argsort(np.abs(t_vals))[::-1]
    # sorted_coeffs = coeff_names[sorted_idx]
    # sorted_tvals = t_vals[sorted_idx]
    # plt.figure(figsize=(6, 6))
    # plt.barh(sorted_coeffs, sorted_tvals, color='steelblue')
    # plt.axvline(0, color='black', linewidth=1)
    # plt.title('T-values per variable', fontsize=14)
    # plt.xlabel('T-value')
    # plt.ylabel('Variable')
    # plt.tight_layout()
    # tvalues_figure = plt.gcf()
    # plt.show()
    # tvalues_figure.savefig(os.path.join(glm_output_dir, "SMX_GLM_T-values.png"))


    # # 4. Partial Residual Plots (PRP)
    # PRP_figure = True
    # if PRP_figure == True: 
    #     # FIGURE 1. Partial Residuals Plots (PRP)    
    #     predictors = ["vpd_log", "ndd_log", "ndd_season_log", "sfcWind_log", "pr_sum_month_log",
    #                   "grassland", "shrubland", "cropland", "grazing_pressure", "GPP_log", "TPI", "VRM", "HDI"]
        
    #     title = "Partial Residual Plots of GLM Predictors"
        
    #     # Number of rows and columns for the panel
    #     n_cols = 4
    #     n_rows = 4
        
    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    #     axes = axes.flatten()
    
    #     for i, pred in enumerate(predictors):
    #         # Partial residuals: response residual + coefficient * predictor
    #         partial_resid = model.resid_response + model.params[pred] * ds_mean[pred]
            
    #         sns.scatterplot(x=ds_mean[pred], y=partial_resid, ax=axes[i])
    #         sns.lineplot(x=ds_mean[pred], y=model.params[pred] * ds_mean[pred], color='red', ax=axes[i])
            
    #         axes[i].set_xlabel(pred)
    #         axes[i].set_ylabel('Partial Residuals')
    #         # axes[i].set_title(f'PRP')
        
    #     # Remove empty subplots
    #     for j in range(i+1, len(axes)):
    #         fig.delaxes(axes[j])
        
    #     plt.tight_layout(rect=[0, 0, 1, 0.95])
    #     fig.suptitle(title, fontsize=16)  # Figure-level title
    #     prp_figure = plt.gcf()
    #     plt.show()    
    #     prp_figure.savefig(os.path.join(glm_output_dir, "SM10_GLM_Partial_Residual_Plot.png"))

    
    ##########################################################################
    # 4. PREDICT BURNED FRACTIONS AND AREAS ON 2001-2019
    ##########################################################################
        
    # 4.A. Predict on the Annual DF (CSV) 
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
    annual_esa["forest_proportion"] = (annual_esa["BA_frac_for"] / annual_esa["BA_frac"]).fillna(0)
    annual_esa["forest_proportion"] = annual_esa["forest_proportion"].where(
        np.isfinite(annual_esa["forest_proportion"]), 0
    )
    annual_esa["forest_proportion"] = annual_esa["forest_proportion"].clip(lower=0,upper=1)
    
    # # Check the difference through time 
    # annual_pred_global = annual_esa["BA_area_pred"].groupby("time").sum()
    # annual_true_global = annual_esa["BA_area"].groupby("time").sum()
    # plt.plot(annual_true_global, label="Observed")
    # plt.plot(annual_pred_global, label="Predicted")
    # plt.legend()
            
    # # Group by year and compute global mean forest proportion
    # global_trend = annual_esa.groupby("time")["forest_proportion"].mean()
    # plt.figure(figsize=(10,6))
    # plt.plot(global_trend.index, global_trend.values, marker='o', color='green')
    # plt.title("Global Temporal Trend of Forest Proportion")
    # plt.xlabel("Year")
    # plt.ylabel("Mean Forest Proportion")
    # plt.ylim(0, 0.2)
    # plt.grid(True)
    # prop_figure = plt.gcf()
    # plt.show()
    # prop_figure.savefig(os.path.join(glm_output_dir, "BA_Forest_Proportion_5.1.png"))
    
    #########################################################################
    # 4.B. Forest proportion mask 
    annual_esa["BA_frac_for_pred"] = annual_esa["BA_frac_pred"] * annual_esa["forest_proportion"]
    annual_esa["BA_area_for_pred"] = annual_esa["BA_frac_for_pred"] * annual_esa["Grid_area"] 
    
    # Compute the non-forest area 
    annual_esa["BA_area_nonf_pred"] = annual_esa["BA_area_pred"] - annual_esa["BA_area_for_pred"]
    annual_esa["BA_frac_nonf_pred"] = annual_esa["BA_frac_pred"] - annual_esa["BA_frac_for_pred"]
    
    #########################################################################
    # 4.C. Checks (optional) and NC conversion
    
    # # Check the latitudinal distribution 
    # annual_esa["lat_band"] = (annual_esa.index.get_level_values("lat") // 2) * 2
    # grouped = annual_esa.groupby("lat_band")[["BA_frac_for", "BA_frac_for_pred"]].mean()
    # grouped.plot()
    
    # # Check the longiotude distribution 
    # annual_esa["lon_band"] = (annual_esa.index.get_level_values("lon") // 2) * 2
    # grouped = annual_esa.groupby("lon_band")[["BA_frac_for", "BA_frac_for_pred"]].mean()
    # grouped.plot()
    
    # # Check the temporal trend
    # annual_pred_global = annual_esa["BA_area_for_pred"].groupby("time").sum()
    # annual_true_global = annual_esa["BA_area_for"].groupby("time").sum()
    # plt.plot(annual_true_global, label="Observed")
    # plt.plot(annual_pred_global, label="Predicted")
    # plt.legend()
    
    # Create netcdf with predictions for figure in DIPC 
    annual_esa_nc = annual_esa.to_xarray().transpose("time", "lat", "lon").where(landmask.mask == 1)
    annual_esa_nc = annual_esa_nc.sortby(annual_esa_nc.lon)
    annual_esa_nc['time'] = pd.to_datetime(annual_esa_nc['time'].values)

    ##########################################################################
    # 5. PREDICT FIRE EMISSIONS GFED FC
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

    # Merge FE to the previous dataframe
    annual_esa = annual_esa_nc.to_dataframe()

    # Save annual_esa CSV and annual_esa_nc 
    if save == True:
        annual_esa.to_csv(os.path.join(glm_output_dir, "Predictions_CSV_Annual.csv"))
        annual_esa_nc.to_netcdf(os.path.join(glm_output_dir, "Predictions_NC_Annual.nc"))
        
        annual_esa_sum = annual_esa.groupby(["time"]).sum()   
        annual_esa_sum.to_csv(os.path.join(glm_output_dir, "Predictions_CSV_Sum.csv"))
            
    # Compute mean predictions and forest proportion for simulated period 
    # TODO: Why not the frac_for_pred here as well? 
    pred_nc = pred_df.set_index(["lat", "lon"]).to_xarray()
    pred_nc["BA_frac"] = pred_nc["BA_frac"] * 12      
    pred_nc["BA_frac_pred"] = pred_nc["BA_frac_pred"] * 12   
    pred_nc["BA_frac_for"] = pred_nc["BA_frac_for"] * 12  
    forest_prop_sim = pred_nc['forest_proportion'].expand_dims({"time": dates_sim})

    if save == True: 
        pred_nc.to_netcdf(os.path.join(glm_output_dir, "Predictions_NC_Mean.nc"))
        forest_prop_sim.to_netcdf(os.path.join(glm_output_dir, f"BA_Forest_Proportion_{end_year}-2100.nc"))    
        
    print("GLM Model Completed. All important files saved in GLM_MODEL")

if __name__ == "__main__":
    
    print("Launching the GLM")
    glm_run(nc_annual, save = False)


