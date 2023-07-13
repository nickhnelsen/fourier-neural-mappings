#!/bin/bash

#SBATCH --time=00:29:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=FAIL
#SBATCH --qos=debug

echo ${1:0:1}_n${3}_d${4}_${5}_L${6}_m${7}_w${8}_md${9}_wd${10}_${2::-7}

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

srun python -u train_fno.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9} ${10}

echo done