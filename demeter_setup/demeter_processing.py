import demeter
import sys

config_file = 'config_files/' + str(sys.argv[1])+'.ini'

# run all time steps
demeter.run_model(config_file=config_file,
                  write_outputs=True)


# # quick check of GCAM Harmonization 
# import xarray as xr 

# demeter_harmonized = xr.open_dataset("C:/GCAM/Theo/GCAM_7.2_Impacts/demeter-2.0.2/demeter/impact_scenarios/outputs/SSP2-6p0_2025-05-08_10h13m29s/spatial_landcover_netcdf/_demeter_SSP2-6p0_2015.nc")
# demeter_default = xr.open_dataset("C:/GCAM/Theo/GCAM_7.2_Impacts/demeter-2.0.2/demeter/impact_scenarios/outputs/SSP2-6p0_2025-04-30_14h33m49s/spatial_landcover_netcdf/_demeter_SSP2-6p0_2015.nc")
# esa_hyde_basemap = xr.open_dataset("C:/GCAM/Theo/GCAM_7.2_Impacts/data/DEMETER/ESA_HYDE_0p5_annual_vegetation.nc")

# demeter_harmonized["PFT2"].plot()
# demeter_default["PFT2"].plot()
# esa_hyde_basemap["forest"].plot()

# demeter_harmonized["PFT2"].sum()
# demeter_default["PFT2"].sum()
# esa_hyde_basemap["forest"].sum()


# demeter_harmonized["PFT1"].sum()
# demeter_default["PFT1"].sum()
# esa_hyde_basemap["grassland"].sum()

# demeter_harmonized["PFT6"].sum()
# demeter_default["PFT6"].sum()
# esa_hyde_basemap["pasture"].sum()

# demeter_harmonized["PFT0"].sum()
# demeter_default["PFT0"].sum()
# esa_hyde_basemap["shrubland"].sum()
