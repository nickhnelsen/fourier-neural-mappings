#!/bin/bash

#SBATCH --time=6-00:00:00
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name="eitDiri10k"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module purge
module load matlab/r2019a
seed=2022
N_loop=5
N_cond=2000
N_solves=128

for (( i=1; i<=$N_loop; i++))
do
   srun --nodes=1 --exclusive --ntasks=1 matlab -nosoftwareopengl -batch "data_generation_dirionly_script $seed$i $N_cond $N_solves" &
done
wait
echo done
