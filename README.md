# IAM-FIRE Repository 

The repo provides the code and instructions to run the Integrated Assessment Model - Fire Impacts and Risks Emulator (IAM-FIRE) for arbitrary IAM scenarios. It provides IAM outputs for the scenario SSP2-RCP4.5 from the Global Change Analysis Model v8.2 for testing purposes. 

The inputs required to run IAM-FIRE are provided in Zenodo here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20312037.svg)](https://doi.org/10.5281/zenodo.20312037). See instructions below for climate inputs. 

The configuration is designed to run on the High Performance Computer (HPC) of the Donostia International Physics Center (DIPC). Hence the HPC configuration should be adapted to computational capacities and setups. 

# Set up the Python environment 

In the HPC environment, set up a Conda virtual environment ([Conda user guide](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)). **Specifically use Conda to manage this environment as we will use a Python package which is only available through Conda.**

Install the package dependencies by using the yml file `<iam-fire-env-file.yml>` provided in the repository which will install the package versions required to run IAM-FIRE. Replace the `envs_path` to place it in the desired folder - preferably in a filesystem visible to the computing nodes of the HPC. Then activate the environment:

```
module load Python
conda env create --prefix <envs_path>/iam-fire-env --file iam-fire-env.yml
conda activate <env>
```

Next, install the following Python packages, [STITCHES](https://jgcri.github.io/stitches/), [BASD](https://jgcri.github.io/basd/) and [DEMETER](https://jgcri.github.io/demeter/build/html/index.html). The three packages are developed and maintained by the Joint Global Change Research Institute (JGCRI). 

You can install from github and specify the branch you want to use if needed:

```
pip install git+https://github.com/LinoHub/demeter
pip install git+https://github.com/JGCRI/basd.git
pip install git+https://github.com/JGCRI/stitches.git@daily-data-refinements

```

The recommended branch Demeter works with GCAM v8 (and superior) with a baseline update from 2015 to 2021. For previous versions of GCAM outputs, use the default Demeter branch: 

```
pip install git+http://github.com/JGCRI/demeter.git#egg=demeter
```

Additionnally, you will need to install the package pyrealm ([pyrealm main page](https://pyrealm.readthedocs.io/en/latest/users/getting_started.html)) to compute Gross Primary Productivity (GPP): 

```
pip install pyrealm
```

# Local set-up and inputs 

Clone this repository onto a local computer. 

```
git clone https://github.com/bc3LC/IAM-FIRE
```

Download the input folder from Zenodo (link above) and add the uncompressed `input` folder to the `IAM-FIRE` main folder. This contains generic inputs used for all IAM-FIRE runs, except for climate data (see below). 

The framework needs three inputs from any IAM scenario to run three streams: climate, vegetation and land use. For convenience, all the inputs (generic and IAM-specific) should be added first in the IAM-FIRE folder, before being uploaded to the HPC where it will run. 

In the input folder, the `stitches_experiment` provides inputs for running IAM-FIRE with SSP2-4.5 inputs for illustrative purposes. For specific studies, duplicate the folder and modify its content accordingly.  

The **Climate stream** requires the GSAT (Global Surface Air Temperature) time series for each scenario. 

The **Vegetation stream** requires the CO2 time series for each IAM scenario while other inputs (VPD and solar radiation) are provided by the Climate stream. 

The **Land use stream** requires gridded land use projections for each scenario. There are several options to provide the data depending on the IAM and downscaling tool that are used, as described in the section below.  

## Set up the climate stream (STITCHES and BASD) 

To run the climate stream of IAM-FIRE you need to create a folder for the experiment you want to run. The folder contains 6 input files that define your experiment details and how you want the software to run. 

The `input/stitches-experiment` directory is an example of such a folder. For a new experiment, duplicate the `stitches-experiment` folder and name it something informative to your experiment. Then you can edit the input files inside to set the configuration.  

1. Update the experiment name in the `code/python/IF_setup.py`
   * The script includes paths, environment and experiment names, common inputs and functions shared across the workflow.
   * For new experiment, modify the `experiment_name` variable with the new name. 
2. In the `input/<experiment_name>` folder add the temperature data in the `temp_global_trajectory.csv` file.
   * If using GCAM, copy-paste the temperature trajectories of the GCAM scenarios directly as new rows of the file. 
   * Make sure the scenario column includes the same names provided in the `run_manager.csv` file described below.
3. Open `run_manager.csv` and edit the available columns:
    * ESM
        * List the names of all the ESMs you want to use.
        * Note: IAM-FIRE has been successfully run with CanESM5 and MPI-ESM1-2-LR and has not been implemented with other ESMs.
    * ESM_Input_Location
        * Set the path to where the data for the ESM is stored. 
    * Output_Location
        * Set the path where the output data should be stored.
    * Reference Dataset
        * List the names of all the reference dataset you want to use (normally just one)
        * Most likely you will need to use W5E5v2, which is the standard for impact studies - see specific instructions below 
    * Reference_Input_Location
        * Set the path where the reference dataset is stored
        * This data must be stored locally on the HPC to be used directly by BASD
    * Variable
        * List the climate variable short names that you want to use 
        * IAM-FIRE climate stream requires air temperature (tas), precipitation (pr), relative humidyt (hurs), wind speed (sfcWind) for the climate stream and solar radiation (rsds) for the vegetation stream.
        * Note: Definitions of basic climate variables can be found in [Juckes et al. (2025)](https://gmd.copernicus.org/articles/18/2639/2025/gmd-18-2639-2025.pdf). 
    * Scenario
        * List the scenarios that you want to use. They should be the same as the GCAM scenarios that you want to emulate with STITCHES
        * Ex. ssp245, ssp370, rcp245, GCAM6-SSP5-Hector, etc.
    * Ensemble
        * If using STITCHES, you do not need to provide ensemble names. 
    * target_period
        * List the year ranges for the reference data that you want to use (normally just use one, the range of the reference data)
        * The recommended range varies, and 20 years is usually seen as a minimum range and 30 years as the standard for climate modelling
        * Use the format start_year-end_year
        * Ex. 1990-2010
    * application_period
        * List the year ranges for which you want to adjust
        * You may use the "future period" of the CMIP6 archive for example, 2015-2100. 
        * You generally don't want to use the exact period as the target to avoid over-fitting. For example if the target period is 1980-2014, and you want to adjust a historical period, perhaps use 1950-2014 as the application period.
        * Use the format start_year-end_year
    * daily
        * Whether to save output data at the daily resolution
        * True or False
    * monthly
        * Whether to save output at the monthly resolution
        * True or False
    * stitched
        * True to use input ESM data created using STITCHES.

4. Open `slurm_parameters.csv` and enter details for the slurm scheduler that will be used for your jobs. The values provided by default work fine for the DIPC/HPC and should be adapted depending on the uses. 

5. The `attributes.csv` file allows you to specify the metadata in the output NetCDF files, both global, and variable-specific attributes. The file as found in the repo give examples of what might be included. However, there is great flexibility here. To add a new tag, add a column with the right name, and assign its value in any row you want it included in.

6. The file `encoding.csv` describes how the output NetCDF files will be encoded when saved. Mostly the defaults should be good for most applications. You may in particular want to change: 
    * `complevel`, which will change the level of compression applied to your data
    * `time_chunk`, `lat_chunk`, `lon_chunk`, which changes the chunk sizes of each dimension in the output NetCDF. This can effect how other programs interact and read in the data consequently. You can either enter an integer, or "max", which will use the full size of the that dimension in the given data.

7. The file `dask_parameters.csv` changes how [Dask](https://www.dask.org/), the Python package responsible for the parallelization in these processes, will split up (i.e. "chunk") the data. For machines with smaller RAM, you may want to lower from the defaults. The `dask_temp_directory` option gives you a chance to change where Dask stores intermediate files. For example, some computing clusters have a `/scratch/` directory where it is ideal to store temporary files that we don't want to be accidentally stored long term.

8. The file `variable_parameters.csv` may be edited, though the values set in the repo will be good for most cases, and more details are given in the file itself.

For IAM-FIRE runs working on the global scale, the recommended climate data to compute the BASD procedure is W5E5v2.0 which can be downloaded [here](https://data.isimip.org/search/simulation_round/ISIMIP3a/product/SecondaryInputData/climate_forcing/w5e5v2.0/). Due to their heavy weights they are not provided in the IAM-FIRE and should be downloaded directly by users. Raw files should be concatenated and split depending on the target period. For 1990-2010, an input file for tas variable should be named `tas_W5E5v2.0_1990.2010.nc`. 

## Set up the vegetation stream (P-Model) 

The CO2 data should be included in the `CO2_global_trajectory.csv` file in the `input/<experiment_name>` folder.

Dynamic inputs to the P-Model (temperature, solar radiation and VPD) are provided as outputs from the climate stream. Atmospheric pressure is estimated from elevation and is kept constant. fAPAR is assumed to be constant through time, and monthly values are derived from the monthly climatology of historic data. 

## Set up the land use stream (Demeter)  
While the GLM requires gridded outputs most IAMs do not work on spatially-explicit grids. Hence, IAM-FIRE uses the Demeter tool to downscale GCAM land use projections. There are three main options: 

**1. DEMETER to downscale aggregate land uses from GCAM runs** 

The land use projections of the historic period are derived from the ESA CCI Plant Functional Types, which requires an harmonization of GCAM land outputs.

First, upload the raw land use results from the rgcam query in the `input/land_use/raw_land_allocation` folder (`GCAM_detailed_land_alloc_SSP2-4p5.csv` is provided as an example). 

Second, run the script `code/R/Demeter_ESA_harmonization.R` which will harmonize the GCAM land use projections with the ESA basemap used by Demeter. The projected files is saved in the `demeter_setup/inputs/projected` folder and the configuration files in the `demeter_setup/config_files` folder (Examples are provided for SSP2-4.5). 

**2. DEMETER to downscale aggregate land uses from other IAMs**

If land use projections are provided to Demeter by another IAMs, then the Demeter configuration files should be updated accordingly. 

Refer to [DEMETER's main page](https://jgcri.github.io/demeter/build/html/index.html) to make the necessary adjustments. 

**3. Gridded land use projections are available**

If gridded land use projections are available, the data can be uploaded in `inputs/land_use` folder of the IAM-FIRE. Ensure the names of input files match the inputs in the `IF_land_use_projections.py`. 

Note: to ensure consistency with the GLM predictors, the spatial resolution should be 0.5-deg resolution and the basemap should be based on the ESA CCI Plant Functional Types.  

## Static input layers (HDI, TPI, VRM, Grazing pressure) 

Human development index (HDI) projections are provided in the `input/human_development` folder for the 5 SSPs.

The static layers used in the GLM are provided for the simulated period in the `input/static_layers` folder. 


# Generate the job scripts for IAM-FIRE

Run `IF_job_script_generation.py` from the root repository level, passing the name of your experiment folder as an argument, and making sure that your conda environment is activated. For example:

```
python code/python/IF_job_script_generation.py "<experiment_name>"
```

A new directory with the name of your experiment folder will be created in the `intermediate` directory. It will contain 11 files:

1. `run_manager_explicit_list.csv`
    * This will list out the details of each run that you requested explicitly. It lists one row per scenario and variable, and will be used by `basd.job` to create one array per variable. 
2. `run_manager_explicit_list_stitch.csv`
    * This will list out the details of each run that you requested explicitly. It lists on row per scenario, and will be used by `stitch_array.job`.
3. `manager.job`
    * This is a bash script responsible for calling the scripts in the correct order.
4. `glm_model.job`
   * This will list out the details of the historic GLM run.
5. `stitch_array` 
    * This is a bash script responsible for submitting a job to generate your STITCHED data. It runs one array per scenario and hence runs faster and more efficiently. 
6. `basd.job`
    * This is a bash script responsible for submitting each of your requested tasks to the slurm scheduler.
7. `climate_proj.job`
    * This is a bash script responsible for submitting a job to generate the projections of climate variables from BASD products.
8. `vegetation_proj.job`
    * This is a bash script responsible for submitting a job to generate the projections of vegetation variable (GPP) using the P-Model.
9. `fc_proj.job`
    * This is a bash script responsible for submitting a job to generate the projections of fuel consumption.
10. `land_use_proj.job`
    * This is a bash script responsible for submitting a job to generate the projections of land uses from the Demeter products. If you do not need to run Demeter, the job can be deleted from the manager or the script can be adapted to your data. 
11. `bace_pred.job`
    * This is a bash script responsible for submitting a job to predict the fire impacts (burned area and carbon emissions) from all the predictors.

It's good to check that these files were generated as you expected. For example that the `.job` files include all the slurm metadata that you input, and check the explicit list file to see the tasks you've requested, and how many there are.

Then, you're ready to run the IAM-FIRE scripts for your scenarios. 

# Running IAM-FIRE

Once the inputs and job scripts are ready, you will need to upload the entire folder on your HPC environment where the results will be generated.

1. Ensure the paths and names in the `IF_setup.py` are adapted to your HPC environment and upload all the `IAM-FIRE` folder from the local server to your HPC (for ex. `scratch/bc3lc/[project_name]`). Note: this can be done using the `pc2hpc.sh` file provided in the repository. 

2. In the folder `input/climate_variables`, download the data that will used for the BASD process (details above). Ensure the path provided in `run_manager.csv` matches this path.

3. To run all the scripts at once, submit the `manager.job` script from the root repository level on the HPC session. For example:
```
sbatch intermediate/<experiment_name>/manager.job
```
This will submit all of your jobs from the GLM model to the fire impact predictions. 

Alternatively you can run each of the `.job` scripts by hand. This may be especially useful if you have lots of tasks you want to run, but maybe not all at once, or want to test the run with just one task. In this case you can run the `stitch_array.job` and then `basd.job` individually. 

Note: Submitting the jobs could run into the following error: `Batch script constains DOS line breaks instead of expected UNIX line breaks`. This can be fixed by running `chmox +x <bash.job>` and `sed -i -e "s/\r$//" <bash.job>`. Check the `useful_commands.txt` for ready-made fixes for all the bash scripts. 

## Python code launched by IAM-FIRE three streams

The `manager` will launch successively a series of jobs which will run specific Python scripts contained in the `code/python` folder. 

1. `IF_glm_model`: The script will read the `GLM_Historic_Inputs.nc`, run the GLM and produce outputs of BA and CE predictions for 2001-2019. 
2. `IF_generate_stitched_data_array`: The script will read the temperature data provided in the inputs and run STITCHES for each of the scenario provided.
3. `IF_main`: The script will launch the BASD process for each scenario and variable emulated by STITCHES. The process is broken in two: 1. Bias-adjustment (BA) and 2. Statistical Downscaling (SD)
  * `IF_stitched_ba`: The script launches the BA process
  * `IF_stitched_sd`: The script launches the SD process
4. `IF_climate_proj`: The script will read the BASD products and compute the climate variables 
5. `IF_vegetation_proj`: The script will read the climate inputs and compute GPP
6. `IF_fc_proj`: The script will read the climate inputs and compute FC
7. `IF_land_use_proj`: The script will read the aggregate land use projections, run Demeter downscaling and process the outputs
8. `IF_bace_pred`: The script will read all the projections and compute the burned area and carbon emissions per scenarios.  

# Managing and Processing the Output

Navigate to the `output/fire_impacts` folder (or whichever you set in the run manager file). This should be populated with final products of IAM-FIRE once the `bace_pred.job` is completed. Congrats! 

The products can be downloaded to you PC with the script `hpc2pc.sh`. You can use software like NCO, with the `ncdump` command to view metadata, or you can use software like [Panopoly](https://www.giss.nasa.gov/tools/panoply/) to open and view the data plotted.
