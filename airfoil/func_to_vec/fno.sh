#!/bin/sh

for n_data in 125 250 500 1000 2000; do
    for k_max in 12; do
        for d_f in 128; do
            for n_layers in 4; do
               
                scommand="sbatch --job-name=NACA_func_to_vec_n_data_${n_data}_k_max_${k_max}_d_f_${d_f}_n_layers_${n_layers} --output=NACA_func_to_vec_n_data_${n_data}_k_max_${k_max}_d_f_${d_f}_n_layers_${n_layers} NACA.sbatch $n_data $k_max $d_f $n_layers"
                    
                echo "submit command: $scommand"
                    
                $scommand
            done
           
        done
    done 
done

