import numpy as np
import os, sys; sys.path.append(os.path.join('..'))

from util import plt

plt.close("all")

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
# plt.rcParams['font.size'] = 16
# plt.rc('legend', fontsize=14)
# plt.rcParams['lines.linewidth'] = 3
# msz = 11
# handlelength = 4.25     # 2.75
# borderpad = 0.4     # 0.15
plt.rcParams['font.size'] = 17
plt.rc('legend', fontsize=15)
plt.rcParams['lines.linewidth'] = 3.5
msz = 13
handlelength = 4.0     # 2.75
borderpad = 0.25     # 0.15

linestyle_tuples = {
     'solid':                 '-',
     'dashdot':               '-.',
     
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     
     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}

# USER INPUT
n_std = 2
save_plots = True
FLAG_WIDE = not False
plot_tol = 1e-6
plot_folder = "./figures/"
os.makedirs(plot_folder, exist_ok=True)

yu = [1e0, 1e0]
y_lim_lower = [1e-2, 1e-4]
const_list = [8e-1, 1.66e-1]

save_pref = "compare_"

if FLAG_WIDE:
    plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
else:
    plt.rcParams['figure.figsize'] = [6.0, 6.0]     # [6.0, 4.0]

# %% Load data for plotting
data_type = 'paper/'

def get_stats(ar):
    out = np.zeros((*ar.shape[-2:], 2))
    out[..., 0] = np.mean(ar, axis=0)
    out[..., 1] = np.std(ar, axis=0)
    return out

data_airfoil = get_stats(np.load('airfoil.npy'))
data_elliptic = get_stats(np.load('elliptic.npy'))

N_train_airfoil = np.array([125, 250, 500, 1000, 2000])
N_train_elliptic = np.array([10, 50, 250, 1000, 2000, 4000, 6000, 8000, 9500])

model_str = ['F2F', 'F2V', 'V2F', 'V2V', 'NN']

data_name_list = ['airfoil', 'elliptic']
data_list = [data_airfoil, data_elliptic]
N_train_all = [N_train_airfoil, N_train_elliptic]

y_label_list = [r'Average Relative Error', r'Average Absolute Error']

# %% Plotting

marker_list = ['o', 'd', 's', 'v', 'X', "*", "P", "^"]
style_list = ['-.', linestyle_tuples['dotted'], linestyle_tuples['densely dashdotted'],
              linestyle_tuples['densely dashed'], linestyle_tuples['densely dashdotdotted']]
color_list = ['k', 'C0', 'C3', 'C1', 'C2', 'C5', 'C4', 'C6', 'C7', 'C8', 'C9']



for idx_d, (data_name, data_to_plot) in enumerate(zip(data_name_list, data_list)):
    plt.figure(idx_d)
    
    N_train_list = N_train_all[idx_d]
    
    plt.loglog(N_train_list, const_list[idx_d]*N_train_list**(-0.5), ls='--', color='darkgray', label=r'$N^{-1/2}$')
    
    for i in range(len(model_str)):
        errors = data_to_plot[i, :, 0]
        stds = data_to_plot[i, :, 1]
        twosigma = n_std*stds
        lb = np.maximum(errors - twosigma, plot_tol)
        ub = errors + twosigma
        plt.loglog(N_train_list, errors, ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=model_str[i])
        plt.fill_between(N_train_list, lb, ub, facecolor=color_list[i], alpha=0.125)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,3,4,5,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
    if idx_d==0:
        plt.xticks(N_train_list, [str(x) for x in N_train_list])
    plt.xlabel(r'$N$')
    plt.ylabel(y_label_list[idx_d])
    plt.grid()
    plt.ylim([y_lim_lower[idx_d], yu[idx_d]])
    if save_plots:
        if FLAG_WIDE:
            data_name = data_name + '_wide'
        plt.savefig(plot_folder + save_pref + data_name + '.pdf', format='pdf')
 