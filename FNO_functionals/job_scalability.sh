#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --ntasks=7
#SBATCH --nodes=2
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name="s_FNF"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

save_prefix=scalability/
declare -a dsl=("nu_inf_ell_p25_torch/" "nu_1p5_ell_p25_torch/" "nu_inf_ell_p05_torch/")
N_train=5000
declare -a ds=("1" "2" "5" "10" "15" "20" "1000")
d_big=1000
sigma=0

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

for data_suffix in "${dsl[@]}"
do
    for d in "${ds[@]}"
    do
        srun --exclusive --ntasks=1 python -u train_fnf.py $save_prefix $data_suffix $N_train $d $sigma | tee ${save_prefix:0:1}_n${N_train}_d${d}_s${sigma}_${data_suffix::-7}.out &
    done
    wait
    python -u eval_dALL_fnf.py $save_prefix $data_suffix $N_train $d_big $sigma
done
echo done
