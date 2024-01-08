#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres gpu:1
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=FAIL


echo ${1:0:1}_n${3}_d${4}_${5}_qoi${11}_L${6}_m${7}_w${8}_md${9}_wd${10}_${2::-7}

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

srun python -u train_dnn.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9} ${10} ${11}

echo done
