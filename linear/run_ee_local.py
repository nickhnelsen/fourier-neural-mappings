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


# User input
FLOAT64_FLAG = True
batch_data = 256 # TODO: batch
batch_wave = 256
M = 2 # number of random repetitions of the experiment
J = 2**11 # number of modes
all_N = 2**np.arange(4, 14 + 1 - 3)
nhn_comment = "debug"

#Noise variance \sigma_2 = \gamma (not squared!)
gamma = 1e-3 # TODO: or try 1e-5

# Input data params
tau_data = 15.0 
alpha_data = 4.5 # TODO: also try 2.25 to align with Figure 2 rate plot in paper

# True coordinates
wcpu = torch.arange(1, J + 1)
wavenumbers = wcpu.to(device)
if FLOAT64_FLAG:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)
    
# TODO: QoI as command line arg
qoi_id = 2
x0 = np.sqrt(2)/2
if qoi_id == 0:
    r = -1.5 # -1.5, -0.5, 0.5
    qoi = np.sqrt(2)*np.pi*wavenumbers*torch.cos(np.pi*wavenumbers*x0)
elif qoi_id == 1:
    r = -0.5 # -1.5, -0.5, 0.5
    qoi = np.sqrt(2)*torch.sin(np.pi*wavenumbers*x0)
elif qoi_id == 2:
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
        t1 = default_timer()
        
        # generate data and form kernel matrix
        X = torch.sqrt(eig_data)[:, None].cpu() * torch.randn(J,N) # cpu
        Y = torch.zeros(N, device=device) # gpu
        G = torch.zeros(N, N, device=device) # gpu
        for idx, x in zip(torch.split(idx_list, batch_wave),
                        torch.split(X, batch_wave)):
            x = x.to(device)
            lam_x = eig_prior[idx,None]*x
            G += torch.einsum("jn,jm->nm", lam_x, x)
            Y += torch.einsum("jn,j->n", x, f_true[idx])
        Y += gamma * torch.randn(N, device=device)
        
        # linear solve
        G += (gamma**2)*torch.eye(N, device=device) # TODO: this uses too much memory
        G = torch.linalg.solve(G, Y)

        # estimator
        f_hat = torch.zeros(J, device=device)
        for idx, x in zip(torch.split(idx_list, batch_wave),
                        torch.split(X, batch_wave)):
            x = x.to(device)
            f_hat[idx] = eig_prior[idx] * torch.einsum("jn,n->j", x, G)
        
        f_hat -= f_true
        errors[k,j] = (normBsq(f_hat, eig_data)/gt_Bsq).item()
        
        t2 = default_timer()

        print(k + 1, j + 1, t2 - t1)

    # Save to file every iteration in Monte Carlo run M
    # np.save(save_path + "boch_store.npy", errors.cpu().numpy())
    tM2 = default_timer()
    print("Time elapsed for this Monte Carlo run (sec):", tM2 - tM1)
    print("Time elapsed since start (min):", (tM2 - ts)/60)

tf = default_timer()
print("Total time elapsed for experiment:", tf - ts)

errors = errors.cpu().numpy()

mean_errors = np.mean(errors, axis=0)
print(mean_errors)

plt.loglog(all_N, mean_errors, 'ko:')

plt.show()