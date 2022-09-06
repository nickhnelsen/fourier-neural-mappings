#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name="d15nu1p5p25"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

data_suffix=nu_1p5_ell_p25/
n_train=12000
d=15
nu=1.5
ell=0.25

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate pyapprox-base
srun --unbuffered python generate_data.py $data_suffix $n_train $d $nu $ell
echo done
