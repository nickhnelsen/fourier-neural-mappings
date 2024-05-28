import torch
import numpy as np
from datetime import datetime
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


# TODO: edit path to /results/r=1/2, etc
def make_save_path(test_num, pth = "/L2_"):
    save_path = "experiments/nondiag/" + datetime.today().strftime('%Y-%m-%d') + pth + str(test_num) +"/"
    return save_path

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
torch.manual_seed(0) # TODO: temp to debug

# User input
FLOAT64_FLAG = True
batch_data = 1024
batch_wave = 1024
M = 1 # number of random repetitions of the experiment
J = 2**15 # number of modes
all_N = 2**np.arange(4, 14 + 1 - 0)
nhn_comment = "debug"

#Noise variance \sigma_2 = \gamma (not squared!)
gamma = 1e-3 # TODO: or try 1e-5

# Input data params
tau_data = 15.0 
alpha_data = 2.00 # TODO: also try 2.25 to align with Figure 2 rate plot in paper; 4.5

# True coordinates
wavenumbers = torch.arange(1, J + 1, device=device)
if FLOAT64_FLAG:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)
    
# TODO: QoI as command line arg
qoi_id = 2
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
    raise ValueError('qoi_id mus be 0, 1, or 2')


# Operator (inverse negative Laplacian in 1D)
beta = 1.5
ell_true = (wavenumbers**(-2.)) / PI_SQRD

# Functional
f_true = ell_true * qoi

# Prior
alpha_prior = beta + r + 1
tau_prior = 2.
sig_prior = tau_prior**(alpha_prior - 1/2)
eig_prior = sig_prior**2*(PI_SQRD*(wavenumbers**2) + tau_prior**2)**(-alpha_prior) 

# Training measure
sig_data = tau_data**(alpha_data - 1/2)
eig_data = sig_data**2*(PI_SQRD*(wavenumbers**2) + tau_data**2)**(-alpha_data)       # train error covariance eigenvalues

# TODO: File I/O
# save_path = make_save_path(TEST_NUM)
# os.makedirs(save_path, exist_ok=True)
hyp_array = np.array((FLOAT64_FLAG, batch_wave, M, J, all_N[-1].item(), nhn_comment, gamma, tau_data, alpha_data)) # array of hyperparameters

# TODO: Write n_list and hyperparameters to file
# np.save(save_path + 'n_list.npy', all_N)
# np.save(save_path + "hyperparameters.npy", hyp_array)

# TODO: compute theoretical convergence rate here
rate = 1. - (1. / (2 + 2*alpha_data + 2*beta + 2*r))

# Setup
nN = all_N.shape[0]
gt_Bsq = normBsq(f_true, eig_data)
errors = torch.zeros((M, nN))

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
    # np.save(save_path + "boch_store.npy", errors.cpu().numpy()) # TODO: save
    tM2 = default_timer()
    print("Time elapsed for this Monte Carlo run (sec):", tM2 - tM1)
    print("Time elapsed since start (min):", (tM2 - ts)/60)

tf = default_timer()
print("Total time elapsed for experiment:", tf - ts)


# TODO: temp visualize error and results
errors = errors.cpu().numpy()
mean_errors = np.mean(errors, axis=0)
print('Convergence rate is', rate)
print(mean_errors)
plt.figure(0)
plt.loglog(all_N, mean_errors, 'ko:')
plt.loglog(all_N, 3e-3*all_N**(-rate), 'r--')
plt.loglog(all_N, 3e-3*all_N**(-1.), 'b-')

plt.show()