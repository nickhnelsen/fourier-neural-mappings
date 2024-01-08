#!/bin/bash

svpref=hyper/
declare -a dsl=("nu_1p5_ell_p25_torch/")
declare -a Ns=("3162")
declare -a models=("DNN")
declare -a Ls=("3" "4" "5")
declare -a widths=("16" "32" "64" "128" "256" "512" "1024" "2048")
declare -a qois=("1234")
d=1000
m=0
m1d=0

COUNT=0
for datsuf in "${dsl[@]}"; do
    dir_name="./results/${svpref}${datsuf}"
    mkdir -p ${dir_name}
    for model in "${models[@]}"; do
        for L in "${Ls[@]}"; do
            for qoi in "${qois[@]}"; do
                for N in "${Ns[@]}"; do
                    for w in "${widths[@]}"; do
                        w1d=$w
                        job_name="${svpref:0:1}_n${N}_d${d}_${model}_qoi${qoi}_L${L}_m${m}_w${w}_md${m1d}_wd${w1d}_${datsuf::-7}"
                        std="${dir_name}R-%x.%j"
                        scommand="sbatch --job-name=${job_name} --output=${std}.out --error=${std}.err train_dnn.sbatch ${svpref} ${datsuf} ${N} ${d} ${model} ${L} ${m} ${w} ${m1d} ${w1d} ${qoi}"

                        echo "submit command: $scommand"

                        $scommand

                        (( COUNT++ ))
                    done
                done
            done
        done
    done
done

echo ${COUNT} jobs
