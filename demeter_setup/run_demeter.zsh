#!/bin/zsh
#SBATCH -A GCIMS
#SBATCH -t 15:00:00
#SBATCH -N 1
#SBATCH -p short,slurm
 
job=$SLURM_JOB_NAME

 
 
module load python
 
date
time python demeter_processing.py $1
date