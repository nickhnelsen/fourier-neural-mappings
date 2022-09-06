#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name="inf1"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

data_suffix=nu_1p5_ell_p25/

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno
srun --unbuffered python process_train.py $data_suffix
echo done
