import numpy as np
import os, sys; sys.path.append(os.path.join('../..'))

from util import plt

plt.close("all")

plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rc('legend', fontsize=12)

# USER INPUT
data_suffix = 'nu_inf_ell_p05_torch/'   # 'nu_inf_ell_p25_torch/', 'nu_1p5_ell_p25_torch/', 'nu_inf_ell_p05_torch/'
idx_er_type = 1     # 0 (qoi bochner), 1 (qoi loss)
n_std = 2

save_plots = True
handlelength = 2.75
borderpad = 0.15
plot_folder = "./results/figures/"
os.makedirs(plot_folder, exist_ok=True)

data_prefix ='/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/advection_diffusion/func_to_vec/results/'
type_list = ['_boch_', '_loss_']
# y_lim_lower = 1e-4
# y_lim_lower = 1e-3
y_lim_lower = 1e-2
yu = [1e0, 1e0]
# const_list = [1e-2, 1.75e-2]
# const_list = [1e-2, 1.3e-1]
const_list = [1e-2, 1.35e-0]

save_pref = "fnf_qoi"

# %% Efficiency (state and total QoI)
data_type = 'efficiency/'
d_str = "1000"
end_str = "m12_w32_md24_wd128"
obj_prefix = 'test_errors_all_TESTd' + d_str

qoi_list = ["012345", "12345", "12", "34", "0", "1", "2", "3", "4", "5"]
N_train_list = np.array([10, 32, 100, 316, 1000, 3162, 10000])
model_list = ['FNF2d', 'FNF1d']
model_str = ['$2d$', '$1d$']
layer_list = ['L2', 'L4']

style_list = ['ko:', 'C0d-.', 'C3s--', 'C1v--']
color_list = ['k', 'C0', 'C3', 'C1']

data_folder = data_prefix + data_type + data_suffix

for idx_qoi, qoi in enumerate(qoi_list):
    plt.figure(idx_qoi)
    
    plt.loglog(N_train_list, const_list[idx_er_type]*N_train_list**(-0.5), ls=(0, (3, 1, 1, 1, 1, 1)), color='darkgray', label=r'$N^{-1/2}$')
    # plt.loglog(N_train_list, const_list[idx_er_type]*N_train_list**(-0.5), ls=(0, (3, 1, 1, 1, 1, 1)), color='darkgray')
    
    cc = 0
    for layer in layer_list:
        for idx_mod, model in enumerate(model_list):
            er_mean = []
            lb = []
            ub = []
            for i, N_train in enumerate(N_train_list):
                obj_suffix = '_n' + str(N_train) + '_d' + d_str +"_" + model + "_qoi" + \
                    qoi + "_" + layer + "_" + end_str  + '.npz'
                er = np.load(data_folder + obj_prefix + obj_suffix)
                er = er[er.files[0]] 
                er_mean.append(np.mean(er,axis=0)) # QoIB, QoIL
                std = np.std(er,axis=0)
                lb.append(er_mean[i] - n_std*std)
                ub.append(er_mean[i] + n_std*std)
            
            er_mean = np.asarray(er_mean)
            lb = np.asarray(lb)
            ub = np.asarray(ub)
            
            plt.loglog(N_train_list, er_mean[:,idx_er_type], style_list[cc], label=model_str[idx_mod] + r' ($L=%d$)' % (int(layer[-1])))
            plt.fill_between(N_train_list, lb[:,idx_er_type], ub[:,idx_er_type], facecolor=color_list[cc], alpha=0.2)
            cc += 1
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,3,4,0]
    # order = [0,1,2,3]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=1,borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    plt.xlabel(r'$N$')
    plt.ylabel(r'Relative error')
    # plt.title(r'Sample Complexity')
    plt.grid()
    plt.ylim([y_lim_lower, yu[idx_er_type]])
    
    if save_plots:
        plt.savefig(plot_folder + save_pref + qoi + "_" + data_type[0] + type_list[idx_er_type] + data_suffix[:-7] + '.pdf', format='pdf')
