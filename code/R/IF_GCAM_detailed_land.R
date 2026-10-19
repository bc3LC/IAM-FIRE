# Demeter projected files

setwd("C:/GCAM/Theo/GCAM_8.2_Impacts")
print(paste0("Current working directory is - ",getwd()))

# Load packages ----
print("Loading packages")
library(devtools)
# install_github('JGCRI/rgcam', build_vignettes=TRUE)
library(rgcam)
library(dplyr)

# List of scenarios
scen_name <- c("SSP1-2p6o", "SSP2-4p5", "SSP3-6p6", "SSP5-7p6")

# Database and Land Allocation query paths -- GCAM 8.2

# Option 1. Create project file from a GCAM DB
db_path <- "output/"
db_name <- "database_basexdb"
query_name <- "data/DEMETER/DetailedLU_query.xml"
# Define the project name
prj_name <- "projects/Impact_SSP-RCP_2019_v2.dat"
# Extract scenario data from database
conn <- localDBConn(db_path, db_name)
# Extract information specified by XML query for each scenario
proj <- addScenario(conn, prj_name, scen_name, query_name)

# Option 2. Load an existing project file / Read an existing .dat project
# proj <- loadProject("projects/Impact_SSP-RCP_2019.dat")


# Export as CSV the query results for al lscenarios
detailed_land <- getQuery(proj, "detailed land allocation")
write.csv(detailed_land, paste0("C:/GCAM/Theo/IAM-FIRE/input/land_use/raw_land_allocation/GCAM_detailed_land_alloc.csv"), row.names = TRUE)

# (Optional) Save one CSV per scenario
for (i in scen_name) {
  detailed_land_i = detailed_land %>%
    dplyr::filter(scenario == i)
  write.csv(detailed_land_i, paste0("C:/GCAM/Theo/IAM-FIRE/input/land_use/raw_land_allocation/GCAM_detailed_land_alloc_", i, ".csv"), row.names = TRUE)
}


