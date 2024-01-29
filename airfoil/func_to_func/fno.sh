#!/bin/sh

for n_data in 125 250 500 1000 2000; do
    for k_max in 12; do
        for d_f in 128; do
               
            scommand="sbatch --job-name=NACA_func_to_func_n_data_${n_data}_k_max_${k_max}_d_f_${d_f} --output=NACA_func_to_func_n_data_${n_data}_k_max_${k_max}_d_f_${d_f} NACA.sbatch $n_data $k_max $d_f"
                    
            echo "submit command: $scommand"
                    
            $scommand
           
        done
    done 
done

