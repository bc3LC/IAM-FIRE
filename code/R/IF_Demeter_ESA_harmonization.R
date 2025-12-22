# Demeter projected files 

# Set the Working Directory - adjust to local vs. HPC contexts
setwd("C:/GCAM/Theo/IAM-FIRE")

print(paste0("Current working directory is - ",getwd()))

# Load packages ----
print("Loading packages")

library(dplyr)
library(tidyr)
library(xlsx)
library(readxl)
library(openxlsx)
library(stringr)

# Load inputs 

# Upload basin mapping
basin.id <- read_excel("input/land_use/mappings/basin_to_country_mapping.xlsx",col_names = TRUE)
region_mapping <- read.csv("input/land_use/mappings/gcam_regions_32.csv")
gcam_long <- read.csv("input/land_use/basemap/GCAM_harmonization_basemap_ESA_2019.csv")
detailed_land_allocation <- read.csv("input/land_use/raw_land_allocation/GCAM_detailed_land_alloc_SSP2-4p5.csv")

print("Creating Demeter inputs from GCAM land allocation query")

# Format GCAM land use outputs to fit as Demeter inputs ----
det.LU <- detailed_land_allocation %>% 
  # Need to comment out the four next lines if using GCAM 7.0 
  mutate(landleaf = gsub("Hardwood_Forest", "Forest", landleaf)) %>%
  mutate(landleaf = gsub("Softwood_Forest", "Forest", landleaf)) %>%
  group_by(Units, scenario, region, landleaf, year) %>%
  summarize(value = sum(value)) %>% ungroup() %>%
  separate(landleaf, into = c("landclass", "GLU_name", "irrtype", "hiORlo"), sep = "_") %>%
  mutate(landclass = case_when (!is.na(irrtype) ~ paste0(landclass,irrtype, hiORlo),TRUE ~ landclass)) %>%
  merge(basin.id,by="GLU_name") %>% dplyr::select(region, landclass, GCAM_basin_ID, year, value, scenario)  %>%
  rename("metric_id"="GCAM_basin_ID") %>% tidyr::spread(year, value) %>% dplyr::select(-"1975")

add_broad_landclass <- function(df) {
  df$broad_landclass <- with(df, ifelse(grepl("RFD", landclass, ignore.case = TRUE), "Cropland",
                                        ifelse(grepl("Forest", landclass, ignore.case = TRUE), "Forest",
                                               ifelse(grepl("Pasture", landclass, ignore.case = TRUE), "Grassland",
                                                      ifelse(grepl("Grassland", landclass, ignore.case = TRUE), "Grassland",
                                                             ifelse(grepl("Shrubland", landclass, ignore.case = TRUE), "Shrubland",
                                                                    ifelse(grepl("Rock", landclass, ignore.case = TRUE), "RockIceDesert",
                                                                           ifelse(grepl("Urban", landclass, ignore.case = TRUE), "Urbanland", 
                                                                                  ifelse(grepl("IRR", landclass, ignore.case = TRUE), "Cropland", 
                                                                                         ifelse(grepl("Tundra", landclass, ignore.case = TRUE), "RockIceDesert", 
                                                                                                ifelse(grepl("Arable", landclass, ignore.case = TRUE), "Cropland",NA)))))))))))
                             return(df) }

det.LU <- add_broad_landclass(det.LU)


det_sum = det.LU %>% 
  group_by(broad_landclass, scenario) %>%
  summarise(across(where(is.numeric), sum, na.rm = TRUE)) 

det_sum = det.LU %>% 
  group_by(scenario) %>%
  summarise(across(where(is.numeric), sum, na.rm = TRUE)) 




# Get unique scenario names
scenario_names <- unique(det.LU$scenario)

# Read and process the GCAM_long to harmonize GCAM projections based on the Demeter basemap
gcam_long = merge(gcam_long, region_mapping, by="region_id") %>% 
  rename("region"="gcam_region_name", "metric_id"="basin_id") %>% 
  filter(broad_landclass != "Grid_area") %>% 
  filter(broad_landclass != "Water") 

gcam_sum = gcam_long %>% 
  group_by(broad_landclass) %>% 
  mutate(sum = sum(value)) %>% 
  ungroup()

# Loop to create one file per scenario in "input/projected" demeter folder and configuration files ----
for (scenario_name in scenario_names) {
  
  # # DEBUG
  # scenario_name = "SSP2-4p5"
  
  # Filter the dataframe for the current scenario
  filtered_df <- det.LU[det.LU$scenario == scenario_name, ]
  filtered_df <- filtered_df %>% dplyr::select(!scenario)
  
  # Harmonize with ESA_HYDE basemap with gcam_long (TODO) 
  test = merge(filtered_df, gcam_long, by=c("metric_id", "region", "broad_landclass")) # Leads to 2924 Mha forest 
  
  test = merge(filtered_df, gcam_long, by=c("metric_id", "region", "broad_landclass"), all = T) %>%
    replace(is.na(.), 0) %>%
    mutate(scenario = scenario_name) %>%
    mutate(
      landclass = if_else(
        landclass == "0",
        str_to_title(broad_landclass),
        landclass
      )) 
  
  test <- test %>%
    mutate(value = value * 10) %>% # From the Mha of Python to the thous km2 of GCAM
    group_by(metric_id, region, broad_landclass) %>%
    mutate(broad_share = `2021` / sum(`2021`, na.rm = TRUE)) %>%
    ungroup() %>% 
    mutate('2021_H' = value * broad_share) 
    # Harmonize the total sum, or not, since before I had perfect match without doing it ... 
    # mutate(`2015_H` = `2015_H` * sum(`2015`, na.rm = TRUE) / sum(`2015_H`, na.rm = TRUE))
    
  
  # Years you want to compute projections for
  years <- seq(2025, 2100, by = 5)
  
  # Initialize by setting 2015_H if not already present (optional fallback)
  test <- test %>%
    mutate(`2021_H` = ifelse(is.na(`2021_H`), 0, `2021_H`))  # or use real values
  
  # Fix the issue of biomass by summing them up into OtherArableLand and modify Demeter accordingly 
  for (i in seq_along(years)) {
    curr_year <- years[i]
    prev_year <- if (curr_year == 2025) 2021 else years[i - 1]
    
    g_curr <- as.character(curr_year)
    g_prev <- as.character(prev_year)
    h_curr <- paste0(curr_year, "_H")
    h_prev <- paste0(prev_year, "_H")
    
    # Step 1: preliminary harmonization logic
    test <- test %>%
      mutate(
        !!h_curr := case_when(
          !!sym(g_prev) > 0 ~ !!sym(h_prev) * (!!sym(g_curr) / !!sym(g_prev)),
          !!sym(g_prev) == 0 ~ !!sym(g_curr)
        )
      )
    
    # Step 2: compute scaling factor f for each group (can be region or region + metric_id)
    f_df <- test %>%
      group_by(metric_id, region) %>%
      filter(!!sym(g_prev) > 0) %>%
      summarize(
        f = sum(!!sym(g_curr), na.rm = TRUE) / sum(!!sym(h_curr), na.rm = TRUE),
        .groups = "drop"
      )
    
    # Step 3: join back and scale H values where g_prev > 0
    test <- test %>%
      left_join(f_df, by = c("metric_id", "region")) %>%
      mutate(
        !!h_curr := ifelse(!!sym(g_prev) > 0, !!sym(h_curr) * f, !!sym(h_curr))
      ) %>%
      dplyr::select(-f)
    
    # Step 4: Scale the total to the total of 2021
    test <- test %>%
      group_by(region) %>%
      mutate(
        total_area = sum(`2021_H`, na.rm = TRUE),
        !!h_curr := !!sym(h_curr) * total_area / sum(!!sym(h_curr), na.rm = TRUE)
      ) %>%
      ungroup() %>%
      dplyr::select(-total_area)
    
    # Step 5: Copy paste 2021 to previous years (won't be downscaled)
    test <- test %>%
      mutate(
        `1990_H` = `2021_H`,
        `2005_H` = `2021_H`,
        `2010_H` = `2021_H`,
        `2015_H` = `2021_H`,
      )
    
  }
  
  # Final cleanup: replace NaN with 0
  test <- test %>%
    mutate(across(everything(), ~ ifelse(is.nan(.), 0, .)))
  
  # Clean the final dataframe
  test <- test %>% 
    dplyr::select(-c("broad_landclass", "value", "region_id", "broad_share",  "1990", "2005", "2010", "2015", "2021", "2025", "2030", "2035", "2040",
                     "2045", "2050", "2055", "2060", "2065", "2070", "2075", "2080", "2085", "2090", "2095", "2100")) 
  names(test) <- sub("_H$", "", names(test))
  
  # Define all year columns in correct order
  year_cols <- c("1990", "2005", "2010", "2015", "2021", as.character(seq(2025, 2100, by = 5)))
  
  # Select and reorder columns: non-year columns first, then ordered year columns
  test <- test %>%
    dplyr::select(
      setdiff(names(test), year_cols),  # all other columns
      all_of(year_cols)                # then ordered year columns
    ) %>% 
    filter(landclass != "Cropland")

  
  filtered_df <- test 
    

  #Define the file paths for saving the filtered dataframe and the configuration file - HARMONIZED
  file_path <- paste0("demeter_setup/inputs/projected/Scenario_", scenario_name, "_ESA_H_2019.csv")
  config_path <- paste0("demeter_setup/config_files/Scenario_", scenario_name, "_ESA_H_2019.ini")
  
  # Config file parameters
  projected_file <- paste0("Scenario_", scenario_name, "_ESA_H_2019.csv")
  
  # Create the content for the config file
  config_file <- paste0(
    "[STRUCTURE]\n",
    "run_dir =                       /scratch/bc3lc/IAM-FIRE/demeter_setup\n",
    "in_dir =                        inputs\n",
    "out_dir =                       outputs\n\n",
    "[INPUTS]\n",
    "allocation_dir =                allocation\n",
    "observed_dir =                  observed\n",
    "constraints_dir =               constraints\n",
    "projected_dir =                 projected\n\n",
    "[[ALLOCATION]]\n",
    "spatial_allocation_file =       gcam_regbasin_moirai_v3_type5_5arcmin_observed_alloc.csv\n",
    "gcam_allocation_file =          gcam_regbasin_moirai_v3_type5_5arcmin_projected_alloc.csv\n",
    "kernel_allocation_file =        gcam_regbasin_moirai_v3_type5_5arcmin_kernel_weighting.csv\n",
    "transition_order_file =         gcam_regbasin_moirai_v3_type5_5arcmin_transition_alloc.csv\n",
    "treatment_order_file =          gcam_regbasin_moirai_v3_type5_5arcmin_order_alloc.csv\n",
    "constraints_file =              gcam_regbasin_moirai_v3_type5_5arcmin_constraint_alloc.csv\n\n",
    "[[OBSERVED]]\n",
    "observed_lu_file =              ESA_0p5_deg_Demeter_basemap_2019.zip\n\n",
    "[[PROJECTED]]\n",
    "projected_lu_file =             ", projected_file, "\n\n",
    "[[MAPPING]]\n",
    "region_mapping_file =           gcam_regions_32.csv\n",
    "basin_mapping_file =            gcam_basin_lookup.csv\n\n",
    "[PARAMS]\n",
    "# scenario name\n",
    "scenario =                      ", scenario_name, "\n\n",
    "# run description\n",
    "run_desc =                      ", scenario_name, "\n\n",
    "# spatial base layer id field name\n",
    "observed_id_field =             fid\n\n",
    "# first year to process\n",
    "start_year =                    2015\n\n",
    "# last year to process\n",
    "end_year =                      2100\n\n",
    "# enter 1 to use non-kernel density constraints, 0 to ignore non-kernel density constraints\n",
    "use_constraints =               1\n\n",
    "# the spatial resolution of the observed spatial data layer in decimal degrees\n",
    "spatial_resolution =            0.5\n\n",
    "# error tolerance in km2 for PFT area change not completed\n",
    "errortol =                      0.001\n\n",
    "# time step in years\n",
    "timestep =                      5\n\n",
    "# factor to multiply the projected land allocation by\n",
    "proj_factor =                   1000\n\n",
    "# from 0 to 1; ideal fraction of LUC that will occur during intensification, the remainder will be expansion\n",
    "intensification_ratio =         0.8\n\n",
    "# activates the stochastic selection of grid cells for expansion of any PFT\n",
    "stochastic_expansion =          1\n\n",
    "# threshold above which grid cells are selected to receive a given land type expansion; between 0 and 1, where 0 is all\n",
    "#     land cells can receive expansion and set to 1 only the grid cell with the maximum likelihood will expand.  For\n",
    "#     a 0.75 setting, only grid cells with a likelihood >= 0.75 x max_likelihood are selected.\n",
    "selection_threshold =           0.75\n\n",
    "# radius in grid cells to use when computing the kernel density; larger is smoother but will increase run-time\n",
    "kernel_distance =               30\n\n",
    "# create kernel density maps; 1 is True\n",
    "map_kernels =                   1\n\n",
    "# create land change maps per time step per land class\n",
    "map_luc_pft =                   0\n\n",
    "# create land change maps for each intensification and expansion step\n",
    "map_luc_steps =                 0\n\n",
    "# creates maps of land transitions for each time step\n",
    "map_transitions =               0\n\n",
    "# years to save data for, default is all; otherwise a semicolon delimited string e.g, 2005;2050\n",
    "target_years_output =           all\n\n",
    "# save tabular spatial landcover as CSV; define tabular_units below (default sqkm)\n",
    "save_tabular =                  0\n\n",
    "# untis to output tabular data in (sqkm or fraction)\n",
    "tabular_units =                 fraction\n\n",
    "# exports CSV files of land transitions for each time step in km2\n",
    "save_transitions =              0\n\n",
    "# create a NetCDF file of land cover percent for each year by grid cell containing each land class\n",
    "save_netcdf_yr =                1\n"
  )
  
  # Save the filtered dataframe to CSV as "projected" Demeter input 
  write.csv(filtered_df, file = file_path, row.names = FALSE)
  write(config_file, file = config_path)
  
}

print(paste0("Demeter runs completed for all scenarios of database"))
