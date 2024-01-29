#!/bin/bash

#SBATCH -J NACA_DNN
#SBATCH --qos=low
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1    
#SBATCH -t 96:00:00

source activate fno

echo NACA_DNN_n_data_${1}_width_${2}_n_layers_${3}

python -u NACA.py $1 $2 $3 >> log.NACA_DNN_n_data_${1}_width_${2}_n_layers_${3}

