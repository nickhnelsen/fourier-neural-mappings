#!/bin/bash

svpref=efficiency_LOOP_TEST/
datsuf=nu_1p5_ell_p25_torch/
N=1000
d=1000
model=FNO2d
declare -a Ls=("2" "4")
m=12
w=32
m1d=24
w1d=128

for L in "${Ls[@]}"; do
    scommand="sbatch train_fno.sbatch $svpref $datsuf $N $d $model $L $m $w $m1d $w1d"
    
    echo "submit command: $scommand"
    
    $scommand
done
