#!/bin/bash

svpref=efficiency_DEBUG/
declare -a dsl=("nu_1p5_ell_p25_torch/")
declare -a Ns=("10")
d=1000
declare -a models=("FNO2d" "FNO1d2")
declare -a Ls=("2")
m=12
w=32
m1d=24
w1d=128

COUNT=0
for datsuf in "${dsl[@]}"; do
    dir_name="./results/${svpref}${datsuf}"
    mkdir -p ${dir_name}
    for model in "${models[@]}"; do
        for L in "${Ls[@]}"; do
            for N in "${Ns[@]}"; do
                job_name="${svpref:0:1}_n${N}_d${d}_${model}_L${L}_m${m}_w${w}_md${m1d}_wd${w1d}_${datsuf::-7}"
                std="${dir_name}R-%x.%j"
                scommand="sbatch --job-name=${job_name} --output=${std}.out --error=${std}.err train_fno.sbatch ${svpref} ${datsuf} ${N} ${d} ${model} ${L} ${m} ${w} ${m1d} ${w1d}"
                
                echo "submit command: $scommand"
                
                $scommand
                
                (( COUNT++ ))
            done
        done
    done
done

echo $COUNT