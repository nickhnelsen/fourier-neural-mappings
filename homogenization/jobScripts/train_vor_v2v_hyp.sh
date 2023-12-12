#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH -J "train_vor_v2v_hyp_%a"    # job name
#SBATCH --output=outputs/train_vor_v2v_%a.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=1-36

cd ../trainModels/

python  -u train_FNM_v2v_model.py ./configs/hyperparam_configs/vor_vec_to_vec_configs/vor_model_v2v_hyp_size_${SLURM_ARRAY_TASK_ID}.yaml  
