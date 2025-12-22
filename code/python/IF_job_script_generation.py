"""
Description: This script will look at the input run manager file and create a bash script to submit all jobs 
             in parallel using the slurm manager, and a csv file that explicitly lists all combinations of 
             runs requested.
Author: Noah Prime
Modified: August 7, 2023
Input:
    - input/<run manager>.csv - file that specifies all the runs requested
    - input/slurm_parameters.csv - parameters to be used for slurm scheduler
Output:
    - intermediate/<run_manager>_explicit_list.csv - file that explicitly lists out the details of each run requested
    - intermediate/<run_manager>.job - bash file for submitting jobs to slurm scheduler
    

conda activate xesmf
cd C:\GCAM\Theo\IAM-FIRE
py code\python\IF_job_script_generation.py “stitches_experiment”


"""

# Import Libraries
import os
import sys
import numpy as np
import pandas as pd

from IF_setup import IF_PATH, conda_env, experiment_name

if __name__ == "__main__":

    # Read in desired run
    run_name = str(sys.argv[1])

    # Define paths
    input_files_path = 'input'
    intermediate_path = 'intermediate'
    run_manager_file = 'run_manager.csv'

    # Helper function to remove nans
    def remove_nas(x):
        return x[~pd.isnull(x)]

    # Read in user defined job requests
    run_manager_df = pd.read_csv(os.path.join(IF_PATH, input_files_path, run_name, run_manager_file))

    esms = remove_nas(run_manager_df['ESM'].values)
    esm_input_paths = remove_nas(run_manager_df['ESM_Input_Location'].values)
    output_paths = remove_nas(run_manager_df['Output_Location'].values)
    ref_datasets = remove_nas(run_manager_df['Reference_Dataset'].values)
    ref_datasets_paths = remove_nas(run_manager_df['Reference_Input_Location'].values)
    variables = remove_nas(run_manager_df['Variable'].values)
    scenarios = remove_nas(run_manager_df['Scenario'].values)
    ensembles = remove_nas(run_manager_df['Ensemble'].values)
    target_periods = remove_nas(run_manager_df['target_period'].values)
    application_periods = remove_nas(run_manager_df['application_period'].values)
    daily = remove_nas(run_manager_df['daily'].values)
    monthly = remove_nas(run_manager_df['monthly'].values)
    stitched = remove_nas(run_manager_df['stitched'].values)

    # If no ensembles given (this happens when we're using STITCHED data)
    if len(ensembles) == 0:
        # Get all combinations (as an array, each row represents a single job)
        mesh_array = np.array(np.meshgrid(esms, 
                                        variables, 
                                        scenarios, 
                                        ref_datasets, 
                                        target_periods, 
                                        application_periods)).T.reshape(-1,6)
        # Convert to pandas DataFrame and add back in columns that didn't need extra enumeration
        mesh_df = pd.DataFrame(mesh_array, columns = ['ESM', 'Variable', 'Scenario', 'Reference_Dataset',
                                                'target_period', 'application_period'])
    # When using ensemble members (using standard CMIP data)
    else:
        # Get all combinations (as an array, each row represents a single job)
        mesh_array = np.array(np.meshgrid(esms, 
                                    variables, 
                                    scenarios, 
                                    ensembles,
                                    ref_datasets, 
                                    target_periods, 
                                    application_periods)).T.reshape(-1,7)
        # Convert to pandas DataFrame and add back in columns that didn't need extra enumeration
        mesh_df = pd.DataFrame(mesh_array, columns = ['ESM', 'Variable', 'Scenario', 'Ensemble', 'Reference_Dataset',
                                                'target_period', 'application_period'])


    # Merge in esm input locations
    mesh_df = mesh_df.merge(run_manager_df[['Reference_Dataset', 'ESM_Input_Location']], on='Reference_Dataset', how='inner')
    # Merge in reference dataset input locations
    mesh_df = mesh_df.merge(run_manager_df[['Reference_Dataset', 'Reference_Input_Location']], on='Reference_Dataset', how='inner')
    # Merge in output paths
    mesh_df = mesh_df.merge(run_manager_df[['Reference_Dataset', 'Output_Location']], on='Reference_Dataset', how='inner')
    # Add daily and monthly bools
    mesh_df['daily'] = daily[0]
    mesh_df['monthly'] = monthly[0]
    mesh_df['stitched'] = stitched[0]
    
    # Make new directory if not already created
    os.makedirs(os.path.join(IF_PATH, intermediate_path, run_name), exist_ok=True)
    # Save dataframe of every run to csv
    mesh_df.to_csv(os.path.join(IF_PATH, intermediate_path, run_name, f'run_manager_explicit_list.csv'), index=False)

    # For STITCH in Array mode in Slurm 

    # If no ensembles given (this happens when we're using STITCHED data)
    if len(ensembles) == 0:
        # Get all combinations (as an array, each row represents a single job)
        mesh_array = np.array(np.meshgrid(esms, 
                                        scenarios, 
                                        ref_datasets, 
                                        target_periods, 
                                        application_periods)).T.reshape(-1,5)
        # Convert to pandas DataFrame and add back in columns that didn't need extra enumeration
        mesh_stitch = pd.DataFrame(mesh_array, columns = ['ESM', 'Scenario', 'Reference_Dataset',
                                                'target_period', 'application_period'])
    # When using ensemble members (using standard CMIP data)
    else:
        # Get all combinations (as an array, each row represents a single job)
        mesh_array = np.array(np.meshgrid(esms, 
                                    scenarios, 
                                    ensembles,
                                    ref_datasets, 
                                    target_periods, 
                                    application_periods)).T.reshape(-1,6)
        # Convert to pandas DataFrame and add back in columns that didn't need extra enumeration
        mesh_stitch = pd.DataFrame(mesh_array, columns = ['ESM',  'Scenario', 'Ensemble', 'Reference_Dataset',
                                                'target_period', 'application_period'])


    # Merge in esm input locations
    mesh_stitch = mesh_stitch.merge(run_manager_df[['Reference_Dataset', 'ESM_Input_Location']], on='Reference_Dataset', how='inner')
    # Merge in reference dataset input locations
    mesh_stitch = mesh_stitch.merge(run_manager_df[['Reference_Dataset', 'Reference_Input_Location']], on='Reference_Dataset', how='inner')
    # Merge in output paths
    mesh_stitch = mesh_stitch.merge(run_manager_df[['Reference_Dataset', 'Output_Location']], on='Reference_Dataset', how='inner')
    # Add daily and monthly bools
    mesh_stitch['daily'] = daily[0]
    mesh_stitch['monthly'] = monthly[0]
    mesh_stitch['stitched'] = stitched[0]
    
    # Make new directory if not already created
    os.makedirs(os.path.join(IF_PATH, intermediate_path, run_name), exist_ok=True)
    # Save dataframe of every run to csv
    mesh_stitch.to_csv(os.path.join(IF_PATH, intermediate_path, run_name, f'run_manager_explicit_list_stitch.csv'), index=False)


    # Read in parameters relating to slurm
    slurm_params = pd.read_csv(os.path.join(IF_PATH, input_files_path, run_name, 'slurm_parameters.csv'))
    # account = slurm_params[slurm_params['parameter'] == 'account']['value'].values[0]
    time = slurm_params[slurm_params['parameter'] == 'time']['value'].values[0]
    partition = slurm_params[slurm_params['parameter'] == 'partition']['value'].values[0]
    qos = slurm_params[slurm_params['parameter'] == 'qos']['value'].values[0]
    max_concurrent = slurm_params[slurm_params['parameter'] == 'max_concurrent']['value'].values[0]
    email = slurm_params[slurm_params['parameter'] == 'email']['value'].values[0]
    mail_type = slurm_params[slurm_params['parameter'] == 'mail-type']['value'].values[0]
    cpus_per_task = slurm_params[slurm_params['parameter'] == 'cpus-per-task']['value'].values[0]
    nodes = slurm_params[slurm_params['parameter'] == 'nodes']['value'].values[0]
    ntasks_per_node = slurm_params[slurm_params['parameter'] == 'ntasks-per-node']['value'].values[0]
    mem = slurm_params[slurm_params['parameter'] == 'mem']['value'].values[0]
    mem_ba = slurm_params[slurm_params['parameter'] == 'mem_ba']['value'].values[0]
    mem_sd = slurm_params[slurm_params['parameter'] == 'mem_sd']['value'].values[0]
    mem_bace = slurm_params[slurm_params['parameter'] == 'mem_bace']['value'].values[0]
    conda_env = slurm_params[slurm_params['parameter'] == 'conda_env']['value'].values[0]
    
    # Create bash file for running the GLM on historic predictors
    with open(os.path.join(IF_PATH, intermediate_path, run_name, 'glm_model.job'), 'w') as job_file:
        job_file.writelines(f"#!/bin/bash\n\n\n")
        job_file.writelines('# Slurm Settings\n')
        # job_file.writelines(f"#SBATCH --account={account}\n")
        job_file.writelines(f"#SBATCH --partition={partition}\n")
        job_file.writelines(f"#SBATCH --qos={qos}\n")
        job_file.writelines(f"#SBATCH --job-name=GLM_{run_name}\n")
        job_file.writelines(f"#SBATCH --time={time}\n")
        job_file.writelines(f"#SBATCH --mail-type={mail_type}\n")
        job_file.writelines(f"#SBATCH --mail-user={email}\n")
        job_file.writelines(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        job_file.writelines(f"#SBATCH --nodes={nodes}\n")
        job_file.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        job_file.writelines(f"#SBATCH --mem={mem}\n")
        job_file.writelines("#SBATCH --output=%x_%a.out\n")
        job_file.writelines("#SBATCH --error=%x_%a.err\n\n\n")

        job_file.writelines('# Load Modules\n')
        job_file.writelines('module load GCC/11.3.0\n')
        job_file.writelines('module load Python\n\n')
        job_file.writelines('module load Anaconda3/2020.02\n')
        job_file.writelines('source /scicomp/EasyBuild/CentOS/7.4.1708/Skylake/software/Anaconda3/2020.02/bin/conda.sh\n\n')
        job_file.writelines('# activate conda environment\n')
        job_file.writelines(f'conda activate {conda_env}\n\n')
        job_file.writelines('# Timing\n')
        job_file.writelines('start=`date +%s.%N`\n\n')
        job_file.writelines('# Run script\n')
        job_file.writelines(f"python ../../code/python/IF_glm_model.py\n\n")
        job_file.writelines('# End timing and print runtime\n')
        job_file.writelines('end=`date +%s.$N`\n')
        job_file.writelines('runtime=$( echo "($end - $start) / 60" | bc -l )\n')
        job_file.writelines('echo "Run completed in $runtime minutes"\n')
    
    # Create bash file for generating STITCHED data
    if stitched:
        with open(os.path.join(IF_PATH, intermediate_path, run_name, 'stitch_array.job'), 'w') as job_file:
            job_file.writelines(f"#!/bin/bash\n\n\n")
            job_file.writelines('# Slurm Settings\n')
            # job_file.writelines(f"#SBATCH --account={account}\n")
            job_file.writelines(f"#SBATCH --partition={partition}\n")
            job_file.writelines(f"#SBATCH --qos={qos}\n")
            job_file.writelines(f"#SBATCH --job-name=STITCH_{run_name}\n")
            job_file.writelines(f"#SBATCH --time={time}\n")
            job_file.writelines(f"#SBATCH --mail-type={mail_type}\n")
            job_file.writelines(f"#SBATCH --mail-user={email}\n")
            job_file.writelines(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
            job_file.writelines(f"#SBATCH --nodes={nodes}\n")
            job_file.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
            job_file.writelines(f"#SBATCH --mem={mem}\n")
            job_file.writelines("#SBATCH --output=%x_%a.out\n")
            job_file.writelines("#SBATCH --error=%x_%a.err\n\n\n")
            job_file.writelines(f"#SBATCH --array=0-{mesh_stitch.shape[0]-1}%{max_concurrent}\n\n\n")

            job_file.writelines('# Load Modules\n')
            job_file.writelines('module load GCC/11.3.0\n')
            job_file.writelines('module load Python\n\n')
            job_file.writelines('module load Anaconda3/2020.02\n')
            job_file.writelines('source /scicomp/EasyBuild/CentOS/7.4.1708/Skylake/software/Anaconda3/2020.02/bin/conda.sh\n\n')
            job_file.writelines('# activate conda environment\n')
            job_file.writelines(f'conda activate {conda_env}\n\n')
            job_file.writelines('# Timing\n')
            job_file.writelines('start=`date +%s.%N`\n\n')
            job_file.writelines('# Run script\n')
            job_file.writelines(f"python ../../code/python/IF_generate_stitched_data_array.py $SLURM_ARRAY_TASK_ID {run_name}\n\n")
            job_file.writelines('# End timing and print runtime\n')
            job_file.writelines('end=`date +%s.$N`\n')
            job_file.writelines('runtime=$( echo "($end - $start) / 60" | bc -l )\n')
            job_file.writelines('echo "Run completed in $runtime minutes"\n')

    # Create bash file for submitting all BASD jobs to slurm
    with open(os.path.join(IF_PATH, intermediate_path, run_name, 'basd.job'), 'w') as job_file:
        job_file.writelines(f"#!/bin/bash\n\n\n")
        job_file.writelines('# Slurm Settings\n')
        # job_file.writelines(f"#SBATCH --account={account}\n")
        job_file.writelines(f"#SBATCH --partition={partition}\n")
        job_file.writelines(f"#SBATCH --qos={qos}\n")
        job_file.writelines(f"#SBATCH --job-name=BASD_{run_name}\n")
        job_file.writelines(f"#SBATCH --time={time}\n")
        job_file.writelines(f"#SBATCH --mail-type={mail_type}\n")
        job_file.writelines(f"#SBATCH --mail-user={email}\n")
        job_file.writelines(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        job_file.writelines(f"#SBATCH --nodes={nodes}\n")
        job_file.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        job_file.writelines(f"#SBATCH --mem={mem}\n")
        job_file.writelines("#SBATCH --output=%x_%a.out\n")
        job_file.writelines("#SBATCH --error=%x_%a.err\n")
        job_file.writelines(f"#SBATCH --array=0-{mesh_df.shape[0]-1}%{max_concurrent}\n\n\n")
        job_file.writelines('# Load Modules\n')
        job_file.writelines('module load GCC/11.3.0\n')
        job_file.writelines('module load Python\n\n')
        job_file.writelines('module load Anaconda3/2020.02\n')
        job_file.writelines('source /scicomp/EasyBuild/CentOS/7.4.1708/Skylake/software/Anaconda3/2020.02/bin/conda.sh\n\n')
        job_file.writelines('# activate conda environment\n')
        job_file.writelines(f'conda activate {conda_env}\n\n')
        job_file.writelines('# Timing\n')
        job_file.writelines('start=`date +%s.%N`\n\n')
        job_file.writelines('# Run script\n')
        job_file.writelines(f"python ../../code/python/IF_basd_main.py $SLURM_ARRAY_TASK_ID {run_name}\n\n")
        job_file.writelines('# End timing and print runtime\n')
        job_file.writelines('end=`date +%s.$N`\n')
        job_file.writelines('runtime=$( echo "($end - $start) / 60" | bc -l )\n')
        job_file.writelines('echo "Run completed in $runtime minutes"\n')

    # Create bash file for generating STITCHED data
    with open(os.path.join(IF_PATH, intermediate_path, run_name, 'climate_proj.job'), 'w') as job_file:
        job_file.writelines(f"#!/bin/bash\n\n\n")
        job_file.writelines('# Slurm Settings\n')
        # job_file.writelines(f"#SBATCH --account={account}\n")
        job_file.writelines(f"#SBATCH --partition={partition}\n")
        job_file.writelines(f"#SBATCH --qos={qos}\n")
        job_file.writelines(f"#SBATCH --job-name=climate_{run_name}\n")
        job_file.writelines(f"#SBATCH --time={time}\n")
        job_file.writelines(f"#SBATCH --mail-type={mail_type}\n")
        job_file.writelines(f"#SBATCH --mail-user={email}\n")
        job_file.writelines(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        job_file.writelines(f"#SBATCH --nodes={nodes}\n")
        job_file.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        job_file.writelines(f"#SBATCH --mem={mem}\n")
        job_file.writelines("#SBATCH --output=%x.out\n")
        job_file.writelines("#SBATCH --error=%x.err\n\n\n")
        job_file.writelines('# Load Modules\n')
        job_file.writelines('module load GCC/11.3.0\n')
        job_file.writelines('module load Python\n\n')
        job_file.writelines('module load Anaconda3/2020.02\n')
        job_file.writelines('source /scicomp/EasyBuild/CentOS/7.4.1708/Skylake/software/Anaconda3/2020.02/bin/conda.sh\n\n')
        job_file.writelines('# activate conda environment\n')
        job_file.writelines(f'conda activate {conda_env}\n\n')
        job_file.writelines('# Timing\n')
        job_file.writelines('start=`date +%s.%N`\n\n')
        job_file.writelines('# Run script\n')
        job_file.writelines(f"python ../../code/python/IF_climate_proj.py {run_name}\n\n")
        job_file.writelines('# End timing and print runtime\n')
        job_file.writelines('end=`date +%s.$N`\n')
        job_file.writelines('runtime=$( echo "($end - $start) / 60" | bc -l )\n')
        job_file.writelines('echo "Run completed in $runtime minutes"\n')

    # Create bash file for generating STITCHED data
    with open(os.path.join(IF_PATH, intermediate_path, run_name, 'vegetation_proj.job'), 'w') as job_file:
        job_file.writelines(f"#!/bin/bash\n\n\n")
        job_file.writelines('# Slurm Settings\n')
        # job_file.writelines(f"#SBATCH --account={account}\n")
        job_file.writelines(f"#SBATCH --partition={partition}\n")
        job_file.writelines(f"#SBATCH --qos={qos}\n")
        job_file.writelines(f"#SBATCH --job-name=veg_{run_name}\n")
        job_file.writelines(f"#SBATCH --time={time}\n")
        job_file.writelines(f"#SBATCH --mail-type={mail_type}\n")
        job_file.writelines(f"#SBATCH --mail-user={email}\n")
        job_file.writelines(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        job_file.writelines(f"#SBATCH --nodes={nodes}\n")
        job_file.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        job_file.writelines(f"#SBATCH --mem={mem}\n")
        job_file.writelines("#SBATCH --output=%x.out\n")
        job_file.writelines("#SBATCH --error=%x.err\n\n\n")
        job_file.writelines('# Load Modules\n')
        job_file.writelines('module load GCC/11.3.0\n')
        job_file.writelines('module load Python\n\n')
        job_file.writelines('module load Anaconda3/2020.02\n')
        job_file.writelines('source /scicomp/EasyBuild/CentOS/7.4.1708/Skylake/software/Anaconda3/2020.02/bin/conda.sh\n\n')
        job_file.writelines('# activate conda environment\n')
        job_file.writelines(f'conda activate {conda_env}\n\n')
        job_file.writelines('# Timing\n')
        job_file.writelines('start=`date +%s.%N`\n\n')
        job_file.writelines('# Run script\n')
        job_file.writelines(f"python ../../code/python/IF_vegetation_proj.py {run_name}\n\n")
        job_file.writelines('# End timing and print runtime\n')
        job_file.writelines('end=`date +%s.$N`\n')
        job_file.writelines('runtime=$( echo "($end - $start) / 60" | bc -l )\n')
        job_file.writelines('echo "Run completed in $runtime minutes"\n')
            

    # Create bash file for generating STITCHED data
    with open(os.path.join(IF_PATH, intermediate_path, run_name, 'fc_proj.job'), 'w') as job_file:
        job_file.writelines(f"#!/bin/bash\n\n\n")
        job_file.writelines('# Slurm Settings\n')
        # job_file.writelines(f"#SBATCH --account={account}\n")
        job_file.writelines(f"#SBATCH --partition={partition}\n")
        job_file.writelines(f"#SBATCH --qos={qos}\n")
        job_file.writelines(f"#SBATCH --job-name=fc_{run_name}\n")
        job_file.writelines(f"#SBATCH --time={time}\n")
        job_file.writelines(f"#SBATCH --mail-type={mail_type}\n")
        job_file.writelines(f"#SBATCH --mail-user={email}\n")
        job_file.writelines(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        job_file.writelines(f"#SBATCH --nodes={nodes}\n")
        job_file.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        job_file.writelines(f"#SBATCH --mem={mem}\n")
        job_file.writelines("#SBATCH --output=%x.out\n")
        job_file.writelines("#SBATCH --error=%x.err\n\n\n")
        job_file.writelines('# Load Modules\n')
        job_file.writelines('module load GCC/11.3.0\n')
        job_file.writelines('module load Python\n\n')
        job_file.writelines('module load Anaconda3/2020.02\n')
        job_file.writelines('source /scicomp/EasyBuild/CentOS/7.4.1708/Skylake/software/Anaconda3/2020.02/bin/conda.sh\n\n')
        job_file.writelines('# activate conda environment\n')
        job_file.writelines(f'conda activate {conda_env}\n\n')
        job_file.writelines('# Timing\n')
        job_file.writelines('start=`date +%s.%N`\n\n')
        job_file.writelines('# Run script\n')
        job_file.writelines(f"python ../../code/python/IF_fc_proj.py {run_name}\n\n")
        job_file.writelines('# End timing and print runtime\n')
        job_file.writelines('end=`date +%s.$N`\n')
        job_file.writelines('runtime=$( echo "($end - $start) / 60" | bc -l )\n')
        job_file.writelines('echo "Run completed in $runtime minutes"\n')
        
        
    # Create bash file for generating STITCHED data
    with open(os.path.join(IF_PATH, intermediate_path, run_name, 'land_use_proj.job'), 'w') as job_file:
        job_file.writelines(f"#!/bin/bash\n\n\n")
        job_file.writelines('# Slurm Settings\n')
        # job_file.writelines(f"#SBATCH --account={account}\n")
        job_file.writelines(f"#SBATCH --partition={partition}\n")
        job_file.writelines(f"#SBATCH --qos={qos}\n")
        job_file.writelines(f"#SBATCH --job-name=land_use_{run_name}\n")
        job_file.writelines(f"#SBATCH --time={time}\n")
        job_file.writelines(f"#SBATCH --mail-type={mail_type}\n")
        job_file.writelines(f"#SBATCH --mail-user={email}\n")
        job_file.writelines(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        job_file.writelines(f"#SBATCH --nodes={nodes}\n")
        job_file.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        job_file.writelines(f"#SBATCH --mem={mem}\n")
        job_file.writelines("#SBATCH --output=%x_%a.out\n")
        job_file.writelines("#SBATCH --error=%x_%a.err\n\n\n")
        job_file.writelines(f"#SBATCH --array=0-{mesh_df.shape[0]-1}%{max_concurrent}\n\n\n")
        job_file.writelines('# Load Modules\n')
        job_file.writelines('module load GCC/11.3.0\n')
        job_file.writelines('module load Python\n\n')
        job_file.writelines('module load Anaconda3/2020.02\n')
        job_file.writelines('source /scicomp/EasyBuild/CentOS/7.4.1708/Skylake/software/Anaconda3/2020.02/bin/conda.sh\n\n')
        job_file.writelines('# activate conda environment\n')
        job_file.writelines(f'conda activate {conda_env}\n\n')
        job_file.writelines('# Timing\n')
        job_file.writelines('start=`date +%s.%N`\n\n')
        job_file.writelines('# Run script\n')
        job_file.writelines(f"python ../../code/python/IF_land_use_proj.py $SLURM_ARRAY_TASK_ID {run_name}\n\n")
        job_file.writelines('# End timing and print runtime\n')
        job_file.writelines('end=`date +%s.$N`\n')
        job_file.writelines('runtime=$( echo "($end - $start) / 60" | bc -l )\n')
        job_file.writelines('echo "Run completed in $runtime minutes"\n')
        
        
    # Create bash file for generating STITCHED data
    with open(os.path.join(IF_PATH, intermediate_path, run_name, 'bace_pred.job'), 'w') as job_file:
        job_file.writelines(f"#!/bin/bash\n\n\n")
        job_file.writelines('# Slurm Settings\n')
        # job_file.writelines(f"#SBATCH --account={account}\n")
        job_file.writelines(f"#SBATCH --partition={partition}\n")
        job_file.writelines(f"#SBATCH --qos={qos}\n")
        job_file.writelines(f"#SBATCH --job-name=bace_pred_{run_name}\n")
        job_file.writelines(f"#SBATCH --time={time}\n")
        job_file.writelines(f"#SBATCH --mail-type={mail_type}\n")
        job_file.writelines(f"#SBATCH --mail-user={email}\n")
        job_file.writelines(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        job_file.writelines(f"#SBATCH --nodes={nodes}\n")
        job_file.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        job_file.writelines(f"#SBATCH --mem={mem_bace}\n")
        job_file.writelines("#SBATCH --output=%x.out\n")
        job_file.writelines("#SBATCH --error=%x.err\n\n\n")
        # job_file.writelines(f"#SBATCH --array=0-{mesh_df.shape[0]-1}%{max_concurrent}\n\n\n")
        job_file.writelines('# Load Modules\n')
        job_file.writelines('module load GCC/11.3.0\n')
        job_file.writelines('module load Python\n\n')
        job_file.writelines('module load Anaconda3/2020.02\n')
        job_file.writelines('source /scicomp/EasyBuild/CentOS/7.4.1708/Skylake/software/Anaconda3/2020.02/bin/conda.sh\n\n')
        job_file.writelines('# activate conda environment\n')
        job_file.writelines(f'conda activate {conda_env}\n\n')
        job_file.writelines('# Timing\n')
        job_file.writelines('start=`date +%s.%N`\n\n')
        job_file.writelines('# Run script\n')
        job_file.writelines(f"python ../../code/python/IF_bace_pred.py {run_name}\n\n")
        job_file.writelines('# End timing and print runtime\n')
        job_file.writelines('end=`date +%s.$N`\n')
        job_file.writelines('runtime=$( echo "($end - $start) / 60" | bc -l )\n')
        job_file.writelines('echo "Run completed in $runtime minutes"\n')
        
        
    # Create bash file for managing the above files in order
    with open(os.path.join(intermediate_path, run_name, 'manager.job'), 'w') as job_file:
        job_file.writelines(f"#!/bin/bash\n\n\n")
        job_file.writelines('# Slurm Settings\n')
        # job_file.writelines(f"#SBATCH --account={account}\n")
        job_file.writelines(f"#SBATCH --partition={partition}\n")
        job_file.writelines(f"#SBATCH --qos={qos}\n")
        job_file.writelines(f"#SBATCH --job-name=manager_{run_name}\n")
        job_file.writelines(f"#SBATCH --time={time}\n")
        job_file.writelines(f"#SBATCH --mail-type={mail_type}\n")
        job_file.writelines(f"#SBATCH --mail-user={email}\n")
        job_file.writelines(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        job_file.writelines(f"#SBATCH --nodes={nodes}\n")
        job_file.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        job_file.writelines(f"#SBATCH --mem={mem_ba}\n")
        job_file.writelines("#SBATCH --output=%x.out\n")
        job_file.writelines("#SBATCH --error=%x.err\n")
        job_file.writelines('# Load Modules\n')
        job_file.writelines('# module load GCC/11.3.0\n')
        job_file.writelines('# module load Python\n\n')
        job_file.writelines('# module load Anaconda3/2020.02\n')
        job_file.writelines('# source /scicomp/EasyBuild/CentOS/7.4.1708/Skylake/software/Anaconda3/2020.02/bin/conda.sh\n\n')
        job_file.writelines('# activate conda environment\n')
        job_file.writelines(f'# conda activate {conda_env}\n\n')
        job_file.writelines('# Timing\n')
        job_file.writelines("start=$(date +%s.%N)\n\n")
        
        job_file.writelines('# Run GLM model generation script\n')
        job_file.writelines(f"glm_id=$(sbatch --parsable glm_model.job)\n\n")
        
        job_file.writelines('# Run land use indicator job\n')
        job_file.writelines(f"land_use_id=$(sbatch --parsable land_use_proj.job '{run_name}')\n\n")
        
        job_file.writelines('# Run STITCHED data generation script\n')
        job_file.writelines(f"stitch_id=$(sbatch --parsable --dependency=afterok:$glm_id stitch_array.job)\n\n")
            
        job_file.writelines('# Run bias adjustment and downscaling\n')
        job_file.writelines(f"basd_id=$(sbatch --parsable --dependency=afterok:$stitch_id basd.job '{run_name}')\n\n")
        
        job_file.writelines('# Run climate projections creation job\n')
        job_file.writelines(f"climate_id=$(sbatch --parsable --dependency=afterok:$basd_id climate_proj.job '{run_name}')\n\n")
        
        job_file.writelines('# Run vegetation projections creation job\n')
        job_file.writelines(f"vegetation_id=$(sbatch --parsable --dependency=afterok:$climate_id vegetation_proj.job '{run_name}')\n\n")
        
        job_file.writelines('# Run fuel consumption projections creation job\n')
        job_file.writelines(f"fc_id=$(sbatch --parsable --dependency=afterok:$vegetation_id fc_proj.job '{run_name}')\n\n")
    
        job_file.writelines('# Run BACE predictor creation job\n')
        # With Demeter run (new scenario added)
        job_file.writelines(f"bace_pred_id=$(sbatch --parsable --dependency=afterok:$fc_id:$land_use_id bace_pred.job '{run_name}')\n\n")
        
        # Without Demeter run (e.g. new ESM added)
        # job_file.writelines(f"predictor_id=$(sbatch --parsable --dependency=afterok:$fc_id bace_predictor.job '{run_name}')\n\n")
                
        job_file.writelines('# End timing and print runtime\n')
        job_file.writelines('end=$(date +%s.%N)\n')
        job_file.writelines('runtime=$( echo "($end - $start) / 60" | bc -l )\n')
        job_file.writelines('echo "Run completed in $runtime minutes"\n')


        
        