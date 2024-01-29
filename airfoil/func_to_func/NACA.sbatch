#!/bin/bash

#SBATCH -J NACA_func_to_func
#SBATCH --qos=low
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1    

#SBATCH -t 96:00:00
          

module load anaconda/3.7.1
source activate fno


echo NACA_func_to_func_n_data_${1}_k_max_${2}_d_f_${3}

python -u NACA.py $1 $2 $3 >> log.NACA_func_to_func_n_data_${1}_k_max_${2}_d_f_${3}
