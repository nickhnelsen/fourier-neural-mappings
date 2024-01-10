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
data_suffix = 'nu_1p5_ell_p25_torch/'
idx_er_type = 1     # 0 (qoi bochner), 1 (qoi loss)
n_std = 2

save_plots = True
handlelength = 2.75
borderpad = 0.15
plot_folder = "./results/figures/hyper/"
os.makedirs(plot_folder, exist_ok=True)

data_prefix ='/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/advection_diffusion/dnn/results/'
type_list = ['_boch_', '_loss_']
# y_lim_lower = 1e-4
# y_lim_lower = 1e-3
y_lim_lower = 1e-2
yu = [1e0, 1e0]
# const_list = [1e-2, 1.75e-2]
# const_list = [1e-2, 1.3e-1]
const_list = [1e-2, 1.35e-0]

save_pref = "dnn_hyper_qoi"

mode=0

# %% Hyperparameter complexity sweep
data_type = 'hyper/'
d_str = "1000"
obj_prefix = 'test_errors_all_TESTd' + d_str

width_list = [16, 32, 64, 128, 256, 512, 1024, 2048]

qoi = "1234"
N_train = 3162
model = 'DNN'
layer_list = ['L3', 'L4', 'L5']

style_list = ['ko:', 'C0d-.', 'C3s--', 'C1v-']
color_list = ['k', 'C0', 'C3', 'C1']

data_folder = data_prefix + data_type + data_suffix

plt.figure(0)
for idx_layer, layer in enumerate(layer_list):
    
    er_mean = []
    lb = []
    ub = []
    nstd = []
    for i, w in enumerate(width_list):
        end_str = "m" + str(mode) + "_w" + str(w) + "_md" + str(mode) + "_wd" + str(w)
        obj_suffix = '_n' + str(N_train) + '_d' + d_str +"_" + model + "_qoi" + \
            qoi + "_" + layer + "_" + end_str  + '.npz'
        er = np.load(data_folder + obj_prefix + obj_suffix)
        er = er[er.files[0]]
        m_tmp = np.mean(er,axis=0)
        er_mean.append(m_tmp) # QoIB, QoIL
        std = np.std(er,axis=0)
        nstd.append(n_std*std)
        lb.append(m_tmp - n_std*std)
        ub.append(m_tmp + n_std*std)
    
    er_mean = np.asarray(er_mean)
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    nstd = np.asarray(nstd)
    
    plt.loglog(width_list, er_mean[:,idx_er_type], style_list[idx_layer], label=layer)
    
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=1,borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.xlabel(r'Width')
plt.ylabel(r'Relative Error')
# plt.title(r'Parameter Complexity')
# plt.grid()
# plt.ylim([y_lim_lower, yu[idx_er_type]])
    
if save_plots:
    plt.savefig(plot_folder + save_pref + qoi + "_" + data_type[0] + type_list[idx_er_type] + data_suffix[:-7] + '.pdf', format='pdf')

# %% Validation error

obj_prefix = 'train_errors_all'

data_folder = data_prefix + data_type + data_suffix

plt.figure(1)
for idx_layer, layer in enumerate(layer_list):
    
    er_mean = []
    lb = []
    ub = []
    nstd = []
    for i, w in enumerate(width_list):
        end_str = "m" + str(mode) + "_w" + str(w) + "_md" + str(mode) + "_wd" + str(w)
        obj_suffix = '_n' + str(N_train) + '_d' + d_str +"_" + model + "_qoi" + \
            qoi + "_" + layer + "_" + end_str  + '.npy'
        er = np.load(data_folder + obj_prefix + obj_suffix)
        er = er[..., -1, 1]
        m_tmp = np.mean(er,axis=0)
        er_mean.append(m_tmp) # QoIB, QoIL
        std = np.std(er,axis=0)
        nstd.append(n_std*std)
        lb.append(m_tmp - n_std*std)
        ub.append(m_tmp + n_std*std)
    
    er_mean = np.asarray(er_mean)
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    nstd = np.asarray(nstd)
    
    plt.loglog(width_list, er_mean, style_list[idx_layer], label=layer)
    
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=1,borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.xlabel(r'Width')
plt.ylabel(r'Relative Error')
# plt.title(r'Parameter Complexity')
# plt.grid()
# plt.ylim([y_lim_lower, yu[idx_er_type]])

if save_plots:
    plt.savefig(plot_folder + save_pref + qoi + "_" + layer + "_" + data_type[0] + type_list[idx_er_type] + data_suffix[:-7] + '_val' + '.pdf', format='pdf')
