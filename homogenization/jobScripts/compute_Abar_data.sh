#!/bin/bash


#SBATCH --time=02:25:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:0
#SBATCH --mem-per-cpu=64G
#SBATCH -J "compute_Abar_data"    # job name
#SBATCH --output=outputs/compute_Abar.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu


cd ../trainModels/util_homogenization
  
python -u make_Abar_data.py
