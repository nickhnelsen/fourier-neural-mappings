#!/bin/bash

#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name="test"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=debug

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno
srun python -u run_ff_local.py
echo done
