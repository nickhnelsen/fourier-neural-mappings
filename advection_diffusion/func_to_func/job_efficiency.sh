#!/bin/bash

svpref=efficiency_LOOP_TEST_PATHS/
datsuf=nu_1p5_ell_p25_torch/
N=1000
d=1000
model=FNO2d
declare -a Ls=("2" "4")
m=12
w=32
m1d=24
w1d=128

dir_name="./results/${svpref}${datsuf}"
mkdir -p ${dir_name}
for L in "${Ls[@]}"; do
    job_name="${svpref:0:1}_n${N}_d${d}_${model}_L${L}_m${m}_w${w}_md${m1d}_wd${w1d}_${datsuf::-7}"
    std="${dir_name}R-%x.%j"
    scommand="sbatch --job-name=${job_name} --output=${std}.out --error=${std}.err train_fno.sbatch ${svpref} ${datsuf} ${N} ${d} ${model} ${L} ${m} ${w} ${m1d} ${w1d}"
    
    echo "submit command: $scommand"
    
    $scommand
done