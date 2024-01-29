#!/bin/sh

for n_data in 125 250 500 1000 2000; do
    for width in 128; do
            for n_layers in 4; do
                scommand="sbatch --job-name=NACA_DNN_n_data_${n_data}_width_${width}_n_layers_${n_layers} --output=NACA_DNN_n_data_${n_data}_width_${width}_n_layers_${n_layers} NACA.sbatch $n_data $width $n_layers"
                    
                echo "submit command: $scommand"
                    
                $scommand
            done
    done 
done

