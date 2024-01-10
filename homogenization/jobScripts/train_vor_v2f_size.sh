#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH -J "train_vor_size_%a"    # job name
#SBATCH --output=size_output/train_vor_size_%a.out  
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=mtrautne@caltech.edu
#SBATCH --array=25-44

cd ../trainModels

PARRAY=(10 50 250 1000 2000 4000 6000 8000 9500)

for ip1 in {0..8} # 9 options
do 
  for i in {0..4} # 5 samples
  do 
     let task_id=$ip1*5+$i
     printf $task_id"\n"
     if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
     then
        data_size=${PARRAY[$ip1]}
	    python  -u train_FNM_v2f_model.py ./configs/data_size_configs_v2f/vor_v2f_data_size_${data_size}.yaml $i
     fi
  done
done
