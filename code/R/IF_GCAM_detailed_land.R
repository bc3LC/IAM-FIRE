# Demeter projected files 

setwd("C:/GCAM/Theo/GCAM_8.2_Impacts")
print(paste0("Current working directory is - ",getwd()))

# Load packages ----
print("Loading packages")
library(devtools)
# install_github('JGCRI/rgcam', build_vignettes=TRUE)
library(rgcam)

# Database and Land Allocation query paths -- GCAM 8.2

# Option 1. Create project file from a GCAM DB
db_path <- "output/"
db_name <- "database_basexdb"
query_name <- "data/DEMETER/DetailedLU_query.xml"
# Define the project name
prj_name <- "projects/Impact_SSP-RCP_2019.dat"
# Update scenario name
scen_name <- c("SSP1-2p6o", "SSP2-4p5", "SSP3-6p6", "SSP5-7p6")
# Extract scenario data from database
conn <- localDBConn(db_path, db_name)
# Extract information specified by XML query for each scenario
proj <- addScenario(conn, prj_name, scen_name, query_name)

# # Option 2. Load an existing project file / Read an existing .dat project
# proj <- loadProject("projects/Impact_SSP-RCP_2019.dat")

detailed_land <- getQuery(proj, "detailed land allocation")
detailed_land = detailed_land %>% 
  dplyr::filter(scenario == "SSP2-4p5")
write.csv(detailed_land,"C:/GCAM/Theo/IAM-FIRE/input/land_use/GCAM_detailed_land_alloc_SSP2-4p5.csv", row.names = TRUE)

