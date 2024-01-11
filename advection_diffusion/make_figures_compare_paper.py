import numpy as np
import torch
import os, sys; sys.path.append(os.path.join('..'))

from util import plt

plt.close("all")

plt.rcParams['figure.figsize'] = [6.0, 6.0]     # [6.0, 4.0]
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rc('legend', fontsize=14)

# USER INPUT
data_suffix = 'nu_1p5_ell_p25_torch/'
idx_er_type = 1     # 1 (qoi loss), 0 = Bochner error (not good for advection-diffusion)
n_std = 2

save_plots = True
handlelength = 2.75
borderpad = 0.15
plot_folder = "./figures_compare_paper/"
os.makedirs(plot_folder, exist_ok=True)

data_prefix_fno ='/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/advection_diffusion/func_to_func/results/'
data_prefix_fnf ='/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/advection_diffusion/func_to_vec/results/'
data_prefix_fnd ='/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/advection_diffusion/vec_to_func/results/'
data_prefix_fnn ='/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/advection_diffusion/vec_to_vec/results/'
data_prefix_dnn ='/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/advection_diffusion/dnn/results/'
eval_prefix = '/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/' + data_suffix + 'testing/'

type_list = ['_boch_', '_loss_']
yu = [1e0, 1e0, 1e0]
y_lim_lower = [1e-4/4, 1e-3/2, 1e-3]
const_list = [1e-2, 1.35e-0, 1.25e0]

save_pref = "compare_qoi"

# %% Load data for plotting
data_type = 'paper/'

qoi = "1234"
N_test = 500

end_str_fno = "m12_w32"
end_str_fnf = "m12_w32_md12_wd32"
end_str_fnd = "m12_w32"
end_str_fnn = "m12_w96_md12_wd96"
end_str_dnn = "m0_w2048_md0_wd2048"

N_train_list = np.array([10, 32, 100, 316, 1000, 3162, 10000])
model_list = ['FNO2d', 'FNF2d', 'FND2d', 'FNN1d', 'DNN']
model_str = ['F2F', 'F2V', 'V2F', 'V2V', 'DNN']
d_list = ['2', '20', '1000']
layer = 'L4'

data_folder_fno = data_prefix_fno + data_type + data_suffix
data_folder_fnf = data_prefix_fnf + data_type + data_suffix
data_folder_fnd = data_prefix_fnd + data_type + data_suffix
data_folder_fnn = data_prefix_fnn + data_type + data_suffix
data_folder_dnn = data_prefix_dnn + data_type + data_suffix

end_str_list = [end_str_fno, end_str_fnf, end_str_fnd, end_str_fnn, end_str_dnn]
data_folder_list = [data_folder_fno, data_folder_fnf, data_folder_fnd, data_folder_fnn, data_folder_dnn]

idx_qoi = [int(x) for x in qoi]    # QoI indices (0 through 5) to learn, e.g.: "012345"
qoi_str = "_qoi" + qoi

data_to_plot = np.zeros([len(d_list), len(model_list), len(N_train_list), 2]) # last axis is mean and nstd*stdev

for idx_d, d_str in enumerate(d_list):
    # Load eval data
    qoi_eval = torch.load(eval_prefix + d_str + 'd/' + 'qoi.pt')['qoi'][:-N_test,...,idx_qoi].clone()
    N_qoi = qoi_eval.shape[-1]
    
    for idx_loop, (model, mstr, end_str, data_folder) in enumerate(zip(model_list,
                                          model_str,
                                          end_str_list,
                                          data_folder_list)):
            
        er_mean = []
        er_std = []
        if mstr=="F2F" or mstr=="V2F": # Get specific QoI for function output methods
            for i, N_train in enumerate(N_train_list):
                obj_prefix = 'qoi_out_all_TESTd' + d_str
                obj_suffix = '_n' + str(N_train) + '_d' + d_str + "_" + model + "_" + layer + "_" + end_str  + '.npy'
                
                # Load QoI outputs
                qoi_out = torch.from_numpy(np.load(data_folder + obj_prefix + obj_suffix)[...,idx_qoi])
                er = torch.mean(torch.linalg.norm(qoi_eval - qoi_out, dim=-1)
                                  / torch.linalg.norm(qoi_eval, dim=-1), dim=-1).numpy() # len(er) = len(model_list)
                er_mean.append(np.mean(er,axis=0))
                er_std.append(np.std(er,axis=0))
        else:
            if mstr=="DNN":
                layer_tmp = 'L3'
            else:
                layer_tmp = layer
            for i, N_train in enumerate(N_train_list):
                obj_prefix = 'test_errors_all_TESTd' + d_str
                obj_suffix = '_n' + str(N_train) + '_d' + d_str +"_" + model + qoi_str + "_" + layer_tmp + "_" + end_str  + '.npz'
                er = np.load(data_folder + obj_prefix + obj_suffix)
                er = er[er.files[0]][..., idx_er_type] # QoIB, QoIL
                er_mean.append(np.mean(er,axis=0))
                er_std.append(np.std(er,axis=0))
        
        # Store
        data_to_plot[idx_d, idx_loop, :, 0] = np.asarray(er_mean)
        data_to_plot[idx_d, idx_loop, :, 1] = np.asarray(er_std)

# %% Plotting

# TODO: add errorbar ticks

style_list = ['ko:', 'C0d-.', 'C3s--', 'C1v:', 'C2X-']
color_list = ['k', 'C0', 'C3', 'C1', 'C2']

for idx_d, d_str in enumerate(d_list):
    plt.figure(idx_d)
    
    plt.loglog(N_train_list, const_list[idx_d]*N_train_list**(-0.5), ls=(0, (3, 1, 1, 1, 1, 1)), color='darkgray', label=r'$N^{-1/2}$')
        
    for i in range(len(model_list)):
        errors = data_to_plot[idx_d, i, :, 0]
        stds = data_to_plot[idx_d, i, :, 1]
        lb = errors - n_std*stds
        ub = errors + n_std*stds
        plt.loglog(N_train_list, errors, style_list[i], label=model_str[i])
        plt.fill_between(N_train_list, lb, ub, facecolor=color_list[i], alpha=0.2)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,3,4,5,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=1,borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    plt.xlabel(r'$N$')
    plt.ylabel(r'Average Relative Error')
    plt.grid()
    plt.ylim([y_lim_lower[idx_d], yu[idx_d]])
    
    if save_plots:
        plt.savefig(plot_folder + save_pref + qoi + "_d" + d_str + type_list[idx_er_type] + data_suffix[:-7] + '.pdf', format='pdf')
