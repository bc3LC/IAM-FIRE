#! /bin/sh

# UPLOAD IAM-FIRE FOLDER 
# scp -r ../IAM-FIRE bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/

# UPLOAD CODE 
scp -r code/python/* bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/IAM-FIRE/code/python

# UPLOAD BASH SCRIPTS 
# scp -r intermediate/stitches_experiment/* bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/IAM-FIRE/intermediate/stitches_experiment/

# UPLOAD INPUTS 
# scp -r input/historic_glm/GLM_Historic_Inputs.nc bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/IAM-FIRE/input/historic_glm/

