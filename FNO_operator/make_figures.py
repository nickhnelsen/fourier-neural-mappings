import numpy as np
import os, sys; sys.path.append(os.path.join('..', 'util'))

from configure_plots import plt

plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

data_prefix = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/results/'
idx_er_type = 2     # 0 (qoi), 1 (bochner), 2 (loss)
type_list = ['_qoi', '_bochner', '_loss']
yu = [1e-1, 1e0, 1e0]
const_list = [1e-2, 1.75e-2, 1.5e-2]
n_std = 1

save_plots = not True
plot_folder = "./results/figures_raise/"
os.makedirs(plot_folder, exist_ok=True)

# %% Robustness
data_type = 'robustness/'
N_train = 5000
obj_prefix = 'test_errors_all_TESTd2'

sigma_list = np.arange(0.0, 2.01, 0.25)
d_str_list = ['2', '1000']
data_suffix_list = ['nu_inf_ell_p25_torch/', 'nu_1p5_ell_p25_torch/']

style_list = ['ko:', 'C0d-.', 'C3s--']
color_list = ['k', 'C0', 'C3']
cc = 0

plt.figure(1)
for k, data_suffix in enumerate(data_suffix_list):
    data_folder = data_prefix + data_type + data_suffix
    if k > 0:
        del d_str_list[0]
        nu_str = r'$\nu_{\mathrm{tr}}=1.5$, '
    else:
        nu_str = r'$\nu_{\mathrm{tr}}=\infty\,$, '
    
    for j, d_str in enumerate(d_str_list):
        er_mean = []
        lb = []
        ub = []

        for i, sigma in enumerate(sigma_list):
            obj_suffix = '_n' + str(N_train) + '_d' + d_str + '_s' + str(i) + '.npz'
            er = np.load(data_folder + obj_prefix + obj_suffix)
            er = er[er.files[0]] 
            er_mean.append(np.mean(er,axis=0)) # QoI, Bochner, Loss
            std = np.std(er,axis=0)
            lb.append(er_mean[i] - n_std*std)
            ub.append(er_mean[i] + n_std*std)
        
        er_mean = np.asarray(er_mean)
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        
        plt.semilogy(sigma_list, er_mean[:,idx_er_type], style_list[cc], label=nu_str + r'$d_{\mathrm{tr}}=%d$' % (int(d_str)))
        plt.fill_between(sigma_list, lb[:,idx_er_type], ub[:,idx_er_type], facecolor=color_list[cc], alpha=0.2)
        cc += 1

plt.legend(ncol=1, loc='best').set_draggable(True)
plt.xlabel(r'$\sigma$')
plt.ylabel(r'Relative error')
plt.title(r'Noise-Robustness ($d=2$, $N=%d$)' % (N_train))
plt.grid()
plt.ylim([1e-4, yu[idx_er_type]])

if save_plots:
    plt.savefig(plot_folder + 'fno_' + data_type[0] + type_list[idx_er_type] + '.pdf', format='pdf')

# %% Scalability
data_type = 'scalability/'
N_train = 5000
sigma = 0

d_str_list = ['1', '2', '5', '10', '15', '20']
data_suffix_list = ['nu_inf_ell_p25_torch/', 'nu_1p5_ell_p25_torch/']

style_list = ['C0d-.', 'C3s--']
color_list = ['C0', 'C3']
cc = 0

plt.figure(2)
d_str_np = np.asarray([int(d) for d in d_str_list])
for k, data_suffix in enumerate(data_suffix_list):
    data_folder = data_prefix + data_type + data_suffix
    
    if k > 0:
        del d_str_list[0]
        nu_str = r'$\nu_{\mathrm{tr}}=1.5$, '
    else:
        nu_str = r'$\nu_{\mathrm{tr}}=\infty\,$, '
    
    if k == 0:
        er_mean = []
        lb = []
        ub = []
        for i, d_str in enumerate(d_str_list):
            obj_prefix = 'test_errors_all_TESTd' + d_str
            obj_suffix = '_n' + str(N_train) + '_d' + d_str + '_s' + str(sigma) + '.npz'
            er = np.load(data_folder + obj_prefix + obj_suffix)
            er = er[er.files[0]] 
            er_mean.append(np.mean(er,axis=0)) # QoI, Bochner, Loss
            std = np.std(er,axis=0)
            lb.append(er_mean[i] - n_std*std)
            ub.append(er_mean[i] + n_std*std)
        
        er_mean = np.asarray(er_mean)
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        
        plt.semilogy(d_str_np, er_mean[:,idx_er_type], 'ko:', label=nu_str + r'$d_{\mathrm{tr}}=d$')
        plt.fill_between(d_str_np, lb[:,idx_er_type], ub[:,idx_er_type], facecolor='k', alpha=0.2)
    
    d_str = '1000'
    er = np.load(data_folder + 'test_errors_all_dALL' + '_n' + str(N_train) + '_d' + d_str + '_s' + str(sigma) + '.npz')
    er = er[er.files[0]] 
    er_mean = np.mean(er,axis=0) # QoI, Bochner, Loss
    std = np.std(er,axis=0)
    lb = er_mean - n_std*std
    ub = er_mean + n_std*std
    
    plt.semilogy(d_str_np, er_mean[:,idx_er_type], style_list[cc], label=nu_str + r'$d_{\mathrm{tr}}=%d$' % (int(d_str)))
    plt.fill_between(d_str_np, lb[:,idx_er_type], ub[:,idx_er_type], facecolor=color_list[cc], alpha=0.2)
    cc += 1

plt.legend(ncol=1, loc='best').set_draggable(True)
plt.xlabel(r'$d$')
plt.ylabel(r'Relative error')
plt.title(r'Scalability ($N=%d$, $\sigma=0$)' % (N_train))
plt.grid()
plt.xlim([0.0, 20.5])
plt.ylim([1e-4, yu[idx_er_type]])

if save_plots:
    plt.savefig(plot_folder + 'fno_' + data_type[0] + type_list[idx_er_type] + '.pdf', format='pdf')

# %% Efficiency
data_type = 'efficiency/'
sigma = 0
obj_prefix = 'test_errors_all_TESTd2'

N_train_list = np.array([10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
d_str_list = ['2', '1000']
data_suffix_list = ['nu_inf_ell_p25_torch/', 'nu_1p5_ell_p25_torch/']

style_list = ['ko:', 'C0d-.', 'C3s--']
color_list = ['k', 'C0', 'C3']
cc = 0

plt.figure(3)

plt.loglog(N_train_list, const_list[idx_er_type]*N_train_list**(-0.5), ls=(0, (3, 1, 1, 1, 1, 1)), color='darkgray', label=r'$N^{-1/2}$')

for k, data_suffix in enumerate(data_suffix_list):
    data_folder = data_prefix + data_type + data_suffix
    if k > 0:
        del d_str_list[0]
        nu_str = r'$\nu_{\mathrm{tr}}=1.5$, '
    else:
        nu_str = r'$\nu_{\mathrm{tr}}=\infty\,$, '

    for j, d_str in enumerate(d_str_list):
        er_mean = []
        lb = []
        ub = []
        for i, N_train in enumerate(N_train_list):
            obj_suffix = '_n' + str(N_train) + '_d' + d_str + '_s' + str(sigma) + '.npz'
            er = np.load(data_folder + obj_prefix + obj_suffix)
            er = er[er.files[0]] 
            er_mean.append(np.mean(er,axis=0)) # QoI, Bochner, Loss
            std = np.std(er,axis=0)
            lb.append(er_mean[i] - n_std*std)
            ub.append(er_mean[i] + n_std*std)
        
        er_mean = np.asarray(er_mean)
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        
        plt.loglog(N_train_list, er_mean[:,idx_er_type], style_list[cc], label=nu_str + r'$d_{\mathrm{tr}}=%d$' % (int(d_str)))
        plt.fill_between(N_train_list, lb[:,idx_er_type], ub[:,idx_er_type], facecolor=color_list[cc], alpha=0.2)
        cc += 1

handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,3,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=1).set_draggable(True)
plt.xlabel(r'$N$')
plt.ylabel(r'Relative error')
plt.title(r'Efficiency ($d=2$, $\sigma=0$)')
plt.grid()
plt.ylim([1e-4, yu[idx_er_type]])

if save_plots:
    plt.savefig(plot_folder + 'fno_' + data_type[0] + type_list[idx_er_type] + '.pdf', format='pdf')
