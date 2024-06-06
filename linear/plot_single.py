import numpy as np
from numpy.polynomial.polynomial import polyfit
import os, sys; sys.path.append(os.path.join('..'))
from util import plt

plt.close("all")

fsz = 16
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = fsz
plt.rc('legend', fontsize=fsz)
plt.rcParams['axes.labelsize'] = fsz
plt.rcParams['axes.titlesize'] = fsz
plt.rcParams['xtick.labelsize'] = fsz
plt.rcParams['ytick.labelsize'] = fsz
plt.rcParams['lines.linewidth'] = 3
msz = 8
handlelength = 2.75     # 2.75
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

marker_list = ['o', 'd', 's', 'v', 'X', "*", "P", "^"]
style_list = ['-.', linestyle_tuples['dotted'], linestyle_tuples['densely dashdotted'],
              linestyle_tuples['densely dashed'], linestyle_tuples['densely dashdotdotted']]
color_list = ['k', 'C0', 'C3', 'C1', 'C2', 'C5', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, blue, red, orange, green, brown, magenta, pink, gray, olive, cyan

# USER INPUT
n_std = 1
plot_tol = 1e-6
save_plots = True
FLAG_WIDE = False
beta = 1.5

# Experiment choice
est_type = "ee"
M = 1000
logJ = 13
idxg = 0

# Choose individual experiment
idxa = 0 # alpha idx
idxq = 2 # QoI idx

# Least squares shift
shift = -5

# Derived
alpha_id = idxa
if alpha_id == 0:
    alpha_data = 0.75
elif alpha_id == 1:
    alpha_data = 2.
elif alpha_id == 2:
    alpha_data = 4.5
else:
    raise ValueError('gamma_id must be 0 or 1')

qoi_id = idxq
if qoi_id == 0: # point evaluation of first derivative
    r = -1.5
elif qoi_id == 1: # point evaluation
    r = -0.5
elif qoi_id == 2: # mean
    r = 0.5
else:
    raise ValueError('qoi_id must be 0, 1, or 2')

if est_type == "ee":
    rate = 1. - (1. / (2 + 2*alpha_data + 2*beta + 2*r))
elif est_type == "ff":
    rate = 1. if r>=0 else (1. - (-2*r / (1 + 2*alpha_data + 2*beta)))
else:
    raise ValueError('est_type must be "ee" or "ff"')

# File I/O
path_suffix = 'M' + str(M) + '_logJ' + str(logJ) + '_gam' + str(idxg) + '_al' + str(idxa) + '/'
save_path = './results/' + est_type + '/' + path_suffix
obj_suffix = '_qoi' + str(idxq) + '.npy'

# Load data
all_N = np.load(save_path + "n_list.npy")
nn = len(all_N)
errors = np.load(save_path + "errors" + obj_suffix)

# Compute statistics
mean_errors = np.mean(errors, axis=0)
stds = np.std(errors, axis=0)
twosigma = n_std*stds
lb = np.maximum(mean_errors - twosigma, plot_tol)
ub = mean_errors + twosigma

# Least square fit to error data
nplot = all_N[shift:]
nplota = all_N
linefit = polyfit(np.log2(nplot), np.log2(mean_errors[shift:]), 1)
lineplot = linefit[0,...] + linefit[1,...]*np.log2(nplot)[:,None]
lineplota = linefit[0,...] + linefit[1,...]*np.log2(nplota)[:,None]
print("Least square slope fit is: ")
print(-linefit[-1])
np.save(save_path + 'rate_ls' + obj_suffix, -linefit[-1])

# Experimental rates of convergence table
eocBoch = np.zeros([nn-1, 1])
for i in range(nn-1):
    eocBoch[i,...] = np.log2(mean_errors[i,...]/mean_errors[i + 1,...])/np.log2(all_N[i + 1]/all_N[i])
print("\nEOC for Bochner norm is: ")
print(eocBoch)
np.save(save_path + "rate_table" + obj_suffix, eocBoch)

# True rate
print('\nTheoretical convergence rate is:', rate)

# Make single plot
if FLAG_WIDE:
    plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
else:
    plt.rcParams['figure.figsize'] = [6.0, 6.0]     # [6.0, 4.0]
plt.figure(0)
plt.loglog(all_N, mean_errors[shift] * all_N[shift]**rate * all_N**(-rate), 'm-', label=r'Theory')
plt.loglog(all_N, mean_errors, 'ko:', label=r'Simulated', markersize=msz)
# plt.loglog(all_N, mean_errors[shift] * all_N[shift]**1. * all_N**(-1.), 'k-', label=r'$N^{-1}$')
# plt.loglog(all_N, mean_errors[shift] * all_N[shift]**0.5 * all_N**(-0.5), 'k-.', label=r'$N^{-1/2}$')
plt.fill_between(all_N, lb, ub, facecolor=color_list[0], alpha=0.125)
plt.xlabel(r'$N$')
plt.xlim((all_N[0]*0.62, all_N[-1]*1.5))
plt.ylabel(r'Relative Squared Error')
plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid()
plt.tight_layout()
if save_plots:
    plt.savefig(save_path + 'rate' + obj_suffix[:-4] + '.pdf', format='pdf')
plt.show()
