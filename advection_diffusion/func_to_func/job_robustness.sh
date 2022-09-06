#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --ntasks=9
#SBATCH --nodes=3
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name="rd1k_FNO"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

d=1000
save_prefix=robustness/
declare -a dsl=("nu_inf_ell_p25_torch/" "nu_1p5_ell_p25_torch/" "nu_inf_ell_p05_torch/")
N_train=5000
n_sigma=8

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

for data_suffix in "${dsl[@]}"
do
    for (( i=0; i<=$n_sigma; i++))
    do
        srun --exclusive --ntasks=1 python -u train_fno.py $save_prefix $data_suffix $N_train $d $i | tee ${save_prefix:0:1}_n${N_train}_d${d}_s${i}_${data_suffix::-7}.out &
    done
    wait
done
echo done
