#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=ecg_param.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mofitzgerald@ucsd.edu


# Set up environment
source ecg_env/bin/activate


#need -u for printing 
# Run the Python script and capture its output in the log file
python scripts/ecg_param_all.py