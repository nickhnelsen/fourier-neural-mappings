#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=FAIL


echo ${1}_bd${2}_bw${3}_M${4}_logJ${5}_gamma${6}_alpha${7}_qoi${8}

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

srun python -u driver.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}

echo done
