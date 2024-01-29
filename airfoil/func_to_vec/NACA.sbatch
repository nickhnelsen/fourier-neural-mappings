#!/bin/bash

#SBATCH -J NACA_func_to_vec
#SBATCH --qos=low
#SBATCH --partition=GPU
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1    

#SBATCH -t 96:00:00
          

module load anaconda/3.7.1
source activate fno


echo NACA_func_to_vec_n_data_${1}_k_max_${2}_d_f_${3}_n_layers_${4}

python -u NACA.py $1 $2 $3 $4 >> log.NACA_func_to_vec_n_data_${1}_k_max_${2}_d_f_${3}_n_layers_${4}


