#!/bin/bash

svpref=paper/
declare -a dsl=("nu_1p5_ell_p25_torch/")
declare -a Ns=("10" "32" "100" "316" "1000" "3162" "10000")
declare -a ds=("2" "20" "1000")
declare -a models=("FNN1d")
declare -a Ls=("4")
m=12
w=96
m1d=12
w1d=96
qoi=1234

COUNT=0
for datsuf in "${dsl[@]}"; do
    dir_name="./results/${svpref}${datsuf}"
    mkdir -p ${dir_name}
    for model in "${models[@]}"; do
        for L in "${Ls[@]}"; do
            for d in "${ds[@]}"; do
                for N in "${Ns[@]}"; do
                    job_name="${svpref:0:1}_n${N}_d${d}_${model}_qoi${qoi}_L${L}_m${m}_w${w}_md${m1d}_wd${w1d}_${datsuf::-7}"
                    std="${dir_name}R-%x.%j"
                    scommand="sbatch --job-name=${job_name} --output=${std}.out --error=${std}.err train_fnn.sbatch ${svpref} ${datsuf} ${N} ${d} ${model} ${L} ${m} ${w} ${m1d} ${w1d} ${qoi}"

                    echo "submit command: $scommand"

                    $scommand

                    (( COUNT++ ))
                done
            done
        done
    done
done

echo ${COUNT} jobs
