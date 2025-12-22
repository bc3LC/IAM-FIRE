#! /bin/sh

# UPLOAD THE IAM-FIRE FOLDER TO THE HPC
# scp -r ../IAM-FIRE bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/

# UPLOAD ALL PYTHON AND BASH SCRIPTS TO THE HPC
# scp -r code/python/* bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/IAM-FIRE/code/python
# scp -r intermediate/stitches_experiment/* bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/IAM-FIRE/intermediate/stitches_experiment/

# UPLOAD SPECIFIC SCRIPTS TO THE HPC
# scp -r code/python/IF_setup.py bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/IAM-FIRE/code/python

# UPLOAD GLM INPUTS TO THE HPC
# scp -r input/historic_glm/GLM_Historic_Inputs_v5.1_v2.nc bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/IAM-FIRE/input/historic_glm/