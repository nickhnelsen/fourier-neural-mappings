#!/bin/bash

etype=ff
bd=1024
bw=1024
M=1000
logJ=12
idxg=0
declare -a alphas=("0" "1" "2")
declare -a qois=("0" "1" "2")

dir_name="./results/${etype}/"
mkdir -p ${dir_name}

COUNT=0
for idxa in "${alphas[@]}"; do
    for idxq in "${qois[@]}"; do
        job_name="${etype}_M${M}_logJ${logJ}_gamma${idxg}_alpha${idxa}_qoi${idxq}"
        std="${dir_name}R-%x.%j"
        scommand="sbatch --job-name=${job_name} --output=${std}.out --error=${std}.err driver.sbatch ${etype} ${bd} ${bw} ${M} ${logJ} ${idxg} ${idxa} ${idxq}"

        echo "submit command: $scommand"

        $scommand

        (( COUNT++ ))
    done
done

echo ${COUNT} jobs
