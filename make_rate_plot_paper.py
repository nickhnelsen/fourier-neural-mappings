import numpy as np
import os, sys; sys.path.append(os.path.join('..'))

from util import plt

plt.close("all")

plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 16
plt.rc('legend', fontsize=14)
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

# USER INPUT
save_plots = True
plot_folder = "./figures/"
os.makedirs(plot_folder, exist_ok=True)
save_pref = "rates_theory"

# %% Plotting

marker_list = ['o', 'd', 's', 'v', 'X', "*", "P", "^"]
style_list = ['-.', linestyle_tuples['dotted'], linestyle_tuples['densely dashdotted'],
              linestyle_tuples['densely dashed'], linestyle_tuples['densely dashdotdotted']]
color_list = ['k', 'C0', 'C3', 'C1', 'C2', 'C5', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, blue, red, orange, green, brown, magenta, pink, gray, olive, cyan

def trunc(x):
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    return x

x = np.linspace(-5, 4.5, 4096)

# c_list = [3.5, 5.5, 7.5]
# alpha_list = [1, 0.5, 0.25]
# alpha_list = [0.25, 0.5, 1]

c_list = [3.5, 7.5]
alpha_list = [1, 0.333]

plt.figure()
plt.axvline(x=-1/2, ls=linestyle_tuples['densely dashed'], color='darkgray')
for idx in range(len(c_list)):
    ee = trunc((x >= -(1 + c_list[idx])/2)*(1. - (1./(2 + c_list[idx] + 2*x))))
    ff = trunc((1 + c_list[idx] + 2*np.minimum(x, 0))/(1 + c_list[idx]))
    
    r_dot = [-(1+c_list[idx])/2, -1/2]
    rho_dot = [0, c_list[idx]/(1 + c_list[idx])]
    
    if idx==0:
        plt.plot(x, ee, ls=linestyle_tuples['densely dashdotdotted'], color='C0', label=r'EE', alpha=alpha_list[idx])
        plt.plot(x, ff, ls='-', color='C1', label=r'FF', alpha=alpha_list[idx])
    else:
        plt.plot(x, ee, ls=linestyle_tuples['densely dashdotdotted'], color='C0', alpha=alpha_list[idx])
        plt.plot(x, ff, ls='-', color='C1', alpha=alpha_list[idx])
    
    for i in range(len(r_dot)):
        plt.plot(r_dot[i], rho_dot[i], 'C5o', markersize=msz, alpha=0.75, markeredgecolor='C5') 

plt.legend(loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.xlabel(r'$r$, QoI Decay Exponent')
plt.ylabel(r'Convergence Rate Exponent')
# plt.xticks(np.arange(-5, 5, 1.0))
plt.grid(alpha=0.67) # axis='y')
plt.xlim([x[0], x[-1]])
plt.ylim([-0.05, 1.05])
plt.tight_layout()

if save_plots:
    plt.savefig(plot_folder + save_pref + '.pdf', format='pdf')
