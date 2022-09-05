#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=9
#SBATCH --nodes=3
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name="ed2_FNO"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

d=2
save_prefix=efficiency/
declare -a dsl=("nu_inf_ell_p25_torch/" "nu_1p5_ell_p25_torch/" "nu_inf_ell_p05_torch/")
declare -a Ns=("2500" "5000" "10000")
declare -a Ns_small=("10" "50" "100" "250" "500" "1000")
sigma=0

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

for N in "${Ns_small[@]}"
do
    for data_suffix in "${dsl[@]}"
    do
        srun --exclusive --ntasks=1 python -u train_fno.py $save_prefix $data_suffix $N $d $sigma | tee ${save_prefix:0:1}_n${N}_d${d}_s${sigma}_${data_suffix::-7}.out &
    done
    wait
done

for N in "${Ns[@]}"
do
    for data_suffix in "${dsl[@]}"
    do
        srun --exclusive --ntasks=1 python -u train_fno.py $save_prefix $data_suffix $N $d $sigma | tee ${save_prefix:0:1}_n${N}_d${d}_s${sigma}_${data_suffix::-7}.out &
    done
done
wait
echo done
