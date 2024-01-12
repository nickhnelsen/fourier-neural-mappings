import numpy as np
import torch
import os, sys; sys.path.append(os.path.join('..'))

from util import plt

plt.rcParams['figure.figsize'] = [6.0, 6.0]
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 18
plt.rc('legend', fontsize=18)
plt.rcParams['lines.linewidth'] = 4
msz = 14
cm_name = 'viridis' # cividis, inferno, viridis
interp_name = 'spline36'

plt.close("all")

# USER INPUT
data_suffix = 'nu_1p5_ell_p25_torch/'

save_plots = True
plot_folder = "./figures_io/"
os.makedirs(plot_folder, exist_ok=True)

data_prefix_fno ='/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/advection_diffusion/func_to_func/results/'
eval_prefix = '/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/' + data_suffix + 'testing/'

save_pref = "io"

# %% Load data for plotting
data_type = 'paper/'

N_train = 100

end_str_fno = "m12_w32"
model = 'FNO2d'
d_list = ['2', '20', '1000']
layer = 'L4'
Q_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4']

data_folder_fno = data_prefix_fno + data_type + data_suffix

end_str = end_str_fno
data_folder = data_folder_fno

N_test = 500

for idx_d, d_str in enumerate(d_list):
    # Load eval data
    state_eval = torch.load(eval_prefix + d_str + 'd/' + 'state.pt')['state'][:-N_test,...].clone()
    vel_eval = torch.load(eval_prefix + d_str + 'd/' + 'velocity.pt')['velocity'][:-N_test,...].clone()
    
    # Load quartile indices
    obj_prefix = 'idx_min_Q1_med_Q3_max_TESTd' + d_str
    obj_suffix = '_n' + str(N_train) + '_d' + d_str + "_" + model + "_" + layer + "_" + end_str  + '.npy'
    
    # Load QoI outputs
    idxs = np.flip(np.load(data_folder + obj_prefix + obj_suffix))

    # Begin five errors plots    
    XX = torch.linspace(0, 1, state_eval.shape[-1])
    (YY, XX) = torch.meshgrid(XX, XX)
    for i, idx in enumerate(idxs):
        true_testsort = state_eval[idx,...].squeeze()
        vel = vel_eval[idx,...].squeeze()
        
        fig, ax = plt.subplots(1,1, figsize = (6,6))
        # ax.imshow(true_testsort, interpolation=interp_name, cmap=cm_name)
        ax.contourf(XX, YY, true_testsort, vmin=0, vmax=15, cmap=cm_name)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(plot_folder + 'ad_out_d' + d_str + '_N' + str(N_train) + '_' + Q_list[i] + '.pdf', format='pdf')
            
        plt.close()
        
        plt.figure()
        plt.plot(torch.linspace(0, 1, vel.shape[-1]), vel, ls='-', color='k')
        plt.xlim(0,1)
        plt.ylim(1.25,4.75)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(plot_folder + 'ad_in_d' + d_str + '_N' + str(N_train) + '_' + Q_list[i] + '.pdf', format='pdf')

    # Close open figure if running interactively
    plt.close()
