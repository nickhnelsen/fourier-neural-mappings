#!/bin/bash

svpref=hyper/
declare -a dsl=("nu_1p5_ell_p25_torch/")
declare -a Ns=("3162")
declare -a models=("FNN1d")
declare -a Ls=("3" "4")
declare -a modes=("3" "6" "12" "18" "24" "36")
declare -a constants=("144" "288" "576" "1152")
declare -a qois=("1234")
d=1000

COUNT=0
for datsuf in "${dsl[@]}"; do
    dir_name="./results/${svpref}${datsuf}"
    mkdir -p ${dir_name}
    for model in "${models[@]}"; do
        for L in "${Ls[@]}"; do
            for qoi in "${qois[@]}"; do
                for N in "${Ns[@]}"; do
                    for const in "${constants[@]}"; do
                        for m in "${modes[@]}"; do
                            w=$((const / m))
                            w1d=$w
                            m1d=$m
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
    done
done

echo ${COUNT} jobs
