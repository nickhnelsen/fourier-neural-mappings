import numpy as np
from numpy.polynomial.polynomial import polyfit
import os, sys; sys.path.append(os.path.join('..'))
from util import plt

plt.close("all")

fsz = 16
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 17
plt.rc('legend', fontsize=16)
plt.rcParams['lines.linewidth'] = 3.5
msz = 13
handlelength = 5.0     # 2.75
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
color_list = ['k', 'C5', 'C3', 'C1', 'C2', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, brown, red, orange, green, blue, purple, pink, gray, olive, cyan

# USER INPUT
n_std = 2
plot_tol = 1e-8
save_plots = True
FLAG_WIDE = False
beta = 1.5

# Experiment choice
est_type = "ff"
M = 1000
logJ = 12
idxg = 0

# Choose individual experiment
idxa = 2 # alpha idx

# Least squares shifts
# ----idxa = 0
# shift_list = [-3, -7, 2]
# shift_list = [2, -7, 2]
# ----idxa = 1
# shift_list = [-4, 3, 1]
# shift_list = [-4, 3, 1]
# ----idxa = 2
# shift_list = [4, -8, 1]
shift_list = [4, -8, 1]

# Legend
legs = [r"$r=-3/2$", r"$r=-1/2$", r"$r=1/2$"]
yl = r""

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
    
if FLAG_WIDE:
    plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
else:
    plt.rcParams['figure.figsize'] = [6.0, 6.0]     # [6.0, 4.0]

# Loop over QoIs and plot
plt.figure(0)
for idxq in range(2 + 1):
    shift = shift_list[idxq]
    
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

    # Experimental rates of convergence table
    eocBoch = np.zeros([nn-1, 1])
    for i in range(nn-1):
        eocBoch[i,...] = np.log2(mean_errors[i,...]/mean_errors[i + 1,...])/np.log2(all_N[i + 1]/all_N[i])
    print("\nEOC for Bochner norm is: ")
    print(eocBoch)

    # True rate
    print('\nTheoretical convergence rate is:', rate)

    # Plot
    plt.loglog(all_N, 2**lineplota[...,0], ls='-', color=color_list[6]) # purple; also can use "m-"
    plt.loglog(all_N, mean_errors, ls=style_list[idxq], color=color_list[idxq], marker=marker_list[idxq], markersize=msz, label=legs[idxq])
    # plt.fill_between(all_N, lb, ub, facecolor=color_list[idxq], alpha=0.125)
    
# Outside of loop
plt.xlabel(r'$N$')
ax1 = plt.gca()
ax1.yaxis.set_label_text(r'Relative Squared Error')
if est_type == "ee":
    plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
else:
    ax1.yaxis.label.set_visible(False)
plt.grid()
plt.grid(alpha=0.67) # axis='y')
plt.xlim((all_N[0]*0.62, all_N[-1]*1.5))

if idxa == 0:
    plt.ylim([0.75*1e-7, 7.5*1e-2])
elif idxa == 1:
    plt.ylim([0.75*1e-7, 3*1e-2])
elif idxa == 2:
    plt.ylim([0.9*1e-7, 1.1*1e-2])

plt.tight_layout()
if save_plots:
    plt.savefig(save_path + 'rate_all' + '.pdf', format='pdf')
plt.show()
