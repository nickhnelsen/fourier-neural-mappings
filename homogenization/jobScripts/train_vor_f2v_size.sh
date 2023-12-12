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
#SBATCH --array=0-24

cd ../trainModels

PARRAY=(2000 4000 6000 8000 9500)

for ip1 in {0..4} # 5 samples
do 
  for i in {0..4} # 5 options
  do 
     let task_id=$i*5+$ip1
     printf $task_id"\n"
     if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
     then
        data_size=${PARRAY[$ip1]}
	    python  -u train_FNM_f2v_model.py ./configs/data_size_configs_f2v/vor_f2v_data_size_${data_size}.yaml $i
     fi
  done
done
