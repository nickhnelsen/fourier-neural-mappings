import torch
import numpy as np
from timeit import default_timer
import os, sys; sys.path.append(os.path.join('..'))
from util import plt

plt.close("all")

fsz = 16
plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
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

def normBsq(seq, eig):
    """
    L_{\nu'}^2(H;H) SQUARED Bochner norm weighted by the test data measure \nu'
       
       seq: (K,) tensor
       eig: (K,) tensor
    """
    return torch.sum(eig*(seq**2))

# Defaults
PI_SQRD = np.pi**2
device = torch.device('cuda')
torch.set_printoptions(precision=12)

# User input
est_type = "ee"
FLOAT64_FLAG = True
FLAG_WIDE = False
batch_data = 1024
batch_wave = 1024
M = 50 # number of random repetitions of the experiment, M=100 or M=250
J = 2**12 # number of modes, 2**12 (0) or 2**15 (1)
all_N = 2**np.arange(4 + 4, 14 + 1 - 2)
nhn_comment = "debug EE"

# Noise standard deviation \gamma (not squared!)
gamma = 1e-3 # 1e-3 (0) or try 1e-5 (1)

# Input data params
tau_data = 15.0 
alpha_data = 2. # 4.5 (2), 2. (1), 0.75 (0)

# True coordinates
wavenumbers = torch.arange(1, J + 1, device=device)
if FLOAT64_FLAG:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)
    
# QoIs
qoi_id = 0
x0 = np.sqrt(2)/2
if qoi_id == 0: # point evaluation of first derivative
    r = -1.5 # -1.5, -0.5, 0.5
    qoi = np.sqrt(2)*np.pi*wavenumbers*torch.cos(np.pi*wavenumbers*x0)
elif qoi_id == 1: # point evaluation
    r = -0.5 # -1.5, -0.5, 0.5
    qoi = np.sqrt(2)*torch.sin(np.pi*wavenumbers*x0)
elif qoi_id == 2: # mean
    r = 0.5 # -1.5, -0.5, 0.5
    qoi = np.sqrt(2)*(1. - (-1.)**wavenumbers) / (np.pi*wavenumbers)
else:
    raise ValueError('qoi_id must be 0, 1, or 2')

# Operator (inverse negative Laplacian in 1D)
beta = 1.5
ell_true = (wavenumbers**(-2.)) / PI_SQRD

# Functional
f_true = ell_true * qoi

# Prior for EE
alpha_prior = beta + r + 1
tau_prior = 2.
sig_prior = tau_prior**(alpha_prior - 1/2)
eig_prior = sig_prior**2*(PI_SQRD*(wavenumbers**2) + tau_prior**2)**(-alpha_prior) 

# Training measure
sig_data = tau_data**(alpha_data - 1/2)
eig_data = sig_data**2*(PI_SQRD*(wavenumbers**2) + tau_data**2)**(-alpha_data)

# File IO
if alpha_data == 0.75:
    al_id = 0
elif alpha_data == 2.:
    al_id = 1
elif alpha_data == 4.5:
    al_id = 2
obj_suffix = '_qoi' + str(qoi_id) + '.npy'
path_suffix = 'M' + str(M) + '_J' + str(int(J==2**15)) + '_gam' + str(int(gamma==1e-5)) + '_al' + str(al_id) + '/'
save_path = './results/' + est_type + '/' + path_suffix
os.makedirs(save_path, exist_ok=True)

# Write n_list to file
np.save(save_path + 'n_list.npy', all_N)

# Setup
nN = all_N.shape[0]
gt_Bsq = normBsq(f_true, eig_data)
errors = torch.zeros((M, nN))
if est_type == "ff":
    f_approx = torch.zeros(J, device=device)

# Loop
idx_list = torch.arange(J)
ts = default_timer()
for k in range(M): # outer Monte Carlo loop
    tM1 = default_timer()
    for j in range(nN): # inner loop over the sample size N
        N = all_N[j]
        idx_listd = torch.arange(N)
        t1 = default_timer()
        
        # generate data 
        X = torch.sqrt(eig_data)[:, None].cpu() * torch.randn(J, N) # cpu
        Y = torch.zeros(N, device=device) # gpu
        for idx_N, x in zip(torch.split(idx_listd, batch_data),
                            torch.split(X, batch_data, dim=-1)):
            for idx_J, xb in zip(torch.split(idx_list, batch_wave),
                                 torch.split(x, batch_wave)):
                xb = xb.to(device)
                Y[idx_N] += torch.einsum("jn,j->n", xb, f_true[idx_J])
        Y += gamma * torch.randn(N, device=device)

        # form kernel matrix
        G = torch.zeros(N, N, device=device) # gpu
        for idx_N, x in zip(torch.split(idx_listd, batch_data),
                            torch.split(X, batch_data, dim=-1)):
            for idx_NN, xx in zip(torch.split(idx_listd, batch_data),
                                torch.split(X, batch_data, dim=-1)):
                for idx_J, xb, xxb in zip(torch.split(idx_list, batch_wave),
                                     torch.split(x, batch_wave),
                                     torch.split(xx, batch_wave)):
                    xb = xb.to(device)
                    xxb = xxb.to(device)
                    G[idx_N[:, None], idx_NN[None, :]] += torch.einsum("jn,jm->nm", eig_prior[idx_J, None]*xb, xxb)
        
        # linear solve
        G.diagonal().copy_(G.diagonal().add_(gamma**2)) # in-place operation to save GPU memory
        G = torch.linalg.solve(G, Y)

        # estimator
        Y = torch.zeros(J, device=device)
        for idx_J, x in zip(torch.split(idx_list, batch_wave),
                            torch.split(X, batch_wave)):
            for idx_N, xb in zip(torch.split(idx_listd, batch_data),
                                 torch.split(x, batch_data, dim=-1)):
                xb = xb.to(device)
                Y[idx_J] += eig_prior[idx_J] * torch.einsum("jn,n->j", xb, G[idx_N])
        
        # test error
        Y -= f_true
        errors[k,j] = (normBsq(Y, eig_data)/gt_Bsq).item()
        
        t2 = default_timer()
        print(k + 1, j + 1, t2 - t1)

    # Save to file after every Monte Carlo iteration
    np.save(save_path + "errors" + obj_suffix, errors.cpu().numpy())
    tM2 = default_timer()
    print("Time elapsed for this Monte Carlo run (sec):", tM2 - tM1)
    print("Time elapsed since start (min):", (tM2 - ts)/60)

tf = default_timer()
print("Total time elapsed for experiment:", tf - ts)


# Compute theoretical EE convergence rate here
rate = 1. - (1. / (2 + 2*alpha_data + 2*beta + 2*r))

# Visualize error and results
errors = errors.cpu().numpy()
mean_errors = np.mean(errors, axis=0)
print('Convergence rate is', rate)
print(mean_errors)
if FLAG_WIDE:
    plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
else:
    plt.rcParams['figure.figsize'] = [6.0, 6.0]     # [6.0, 4.0]
plt.figure(0)
plt.loglog(all_N, mean_errors, 'o:', label=r'Simulated', )
plt.loglog(all_N, mean_errors[0] * all_N[0]**rate * all_N**(-rate), '--', label=r'Theory')
plt.loglog(all_N, mean_errors[0] * all_N[0]**1. * all_N**(-1.), 'k-', label=r'$N^{-1}$')
plt.loglog(all_N, mean_errors[0] * all_N[0]**0.5 * all_N**(-0.5), 'k-.', label=r'$N^{-1/2}$')
plt.xlabel(r'$N$')
plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.tight_layout()
plt.savefig(save_path + 'rates_temp' + obj_suffix[:-4] + '.pdf', format='pdf')
plt.show()
