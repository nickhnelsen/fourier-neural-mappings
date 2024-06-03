import torch
import numpy as np
from timeit import default_timer
import os, sys; sys.path.append(os.path.join('..'))


def normBsq(seq, eig):
    """
    L_{\nu'}^2(H;H) SQUARED Bochner norm weighted by the test data measure \nu'
       
       seq: (K,) tensor
       eig: (K,) tensor
    """
    return torch.sum(eig*(seq**2))


# Process command line arguments
print(sys.argv)
est_type = sys.argv[1]  # "ee" or "ff"
batch_data = int(sys.argv[2])
batch_wave = int(sys.argv[3])
M = int(sys.argv[4])    # number of random repetitions of the experiment, M=100 or M=250
logJ = int(sys.argv[5]) # log_2(J), e.g.: 12, 15
gamma_id = int(sys.argv[6])     # 0 for 1e-3 or 1 for 1e-5
alpha_id = int(sys.argv[7])     # 0 for 0.75, 1 for 2., 2 for 4.5 
qoi_id = int(sys.argv[8])       # 0, 1, 2 for r=-1.5, -0.5, 0.5

# Defaults
FLOAT64_FLAG = True
PI_SQRD = np.pi**2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)
torch.set_printoptions(precision=12)

# User input
J = 2**logJ             # number of modes, e.g.: 2**12 or 2**15
all_N = 2**np.arange(4, 14 + 1)

# Noise standard deviation \gamma (not squared!)
if gamma_id == 0:
    gamma = 1e-3
elif gamma_id == 1:
    gamma = 1e-5
else:
    raise ValueError('gamma_id must be 0 or 1')

# Input data params
tau_data = 15.0 
if alpha_id == 0:
    alpha_data = 0.75
elif alpha_id == 1:
    alpha_data = 2.
elif alpha_id == 2:
    alpha_data = 4.5
else:
    raise ValueError('gamma_id must be 0 or 1')

# True coordinates
wavenumbers = torch.arange(1, J + 1, device=device)
if FLOAT64_FLAG:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)
    
# QoIs
x0 = np.sqrt(2)/2
if qoi_id == 0: # point evaluation of first derivative
    r = -1.5
    qoi = np.sqrt(2)*np.pi*wavenumbers*torch.cos(np.pi*wavenumbers*x0)
elif qoi_id == 1: # point evaluation
    r = -0.5
    qoi = np.sqrt(2)*torch.sin(np.pi*wavenumbers*x0)
elif qoi_id == 2: # mean
    r = 0.5
    qoi = np.sqrt(2)*(1. - (-1.)**wavenumbers) / (np.pi*wavenumbers)
else:
    raise ValueError('qoi_id must be 0, 1, or 2')

# Operator (inverse negative Laplacian in 1D)
beta = 1.5
ell_true = (wavenumbers**(-2.)) / PI_SQRD

# Functional
f_true = ell_true * qoi

# Priors
if est_type == "ee":
    alpha_prior = beta + r + 1
elif est_type == "ff":
    alpha_prior = beta + 1/2
else:
    raise ValueError('est_type must be "ee" or "ff"')
tau_prior = 2.
sig_prior = tau_prior**(alpha_prior - 1/2)
eig_prior = sig_prior**2*(PI_SQRD*(wavenumbers**2) + tau_prior**2)**(-alpha_prior) 

# Training measure
sig_data = tau_data**(alpha_data - 1/2)
eig_data = sig_data**2*(PI_SQRD*(wavenumbers**2) + tau_data**2)**(-alpha_data)

# File IO
obj_suffix = '_qoi' + str(qoi_id) + '.npy'
path_suffix = 'M' + str(M) + '_logJ' + str(logJ) + '_gam' + str(int(gamma==1e-5)) + '_al' + str(alpha_id) + '/'
save_path = './results/' + est_type + '/' + path_suffix
os.makedirs(save_path, exist_ok=True)

# Write n_list to file
np.save(save_path + 'n_list.npy', all_N)

# Setup
nN = all_N.shape[0]
gt_Bsq = normBsq(f_true, eig_data)
errors = torch.zeros((M, nN))

# Loop
idx_list = torch.arange(J)
ts = default_timer()
if est_type == "ff":
    f_approx = torch.zeros(J, device=device)
    for k in range(M): # outer Monte Carlo loop
        tM1 = default_timer()
        for j in range(nN): # inner loop over the sample size N
            N = all_N[j]
            idx_listd = torch.arange(N)
            t1 = default_timer()
            
            # generate data and form diagonal estimator
            f_approx.zero_()
            for idx_J in torch.split(idx_list, batch_wave):
                yx = torch.zeros(len(idx_J), device=device)
                xx = torch.zeros(len(idx_J), device=device)
                for idx_N in torch.split(idx_listd, batch_data):
                    x = torch.sqrt(eig_data[idx_J, None]) * torch.randn(len(idx_J), len(idx_N), device=device)
                    y = ell_true[idx_J, None]*x
                    y += gamma * torch.randn(len(idx_J), len(idx_N), device=device)
                    yx += torch.sum(y*x, dim=-1)
                    xx += torch.sum(x**2, dim=-1)
                f_approx[idx_J] = qoi[idx_J] * yx / (xx + (gamma**2 / eig_prior[idx_J])) 
            
            # test error
            f_approx -= f_true
            errors[k,j] = (normBsq(f_approx, eig_data)/gt_Bsq).item()
            
            t2 = default_timer()
            print(k + 1, j + 1, t2 - t1)
    
        # Save to file after every Monte Carlo iteration
        np.save(save_path + "errors" + obj_suffix, errors.cpu().numpy())
        tM2 = default_timer()
        print("Time elapsed for this Monte Carlo run (sec):", tM2 - tM1)
        print("Time elapsed since start (min):", (tM2 - ts)/60)
elif est_type == "ee":
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
print("Total time elapsed for experiment (hr):", (tf - ts)/3600)


# Compute theoretical convergence rates
rate_ee = 1. - (1. / (2 + 2*alpha_data + 2*beta + 2*r))
rate_ff = 1. if r>=0 else (1. - (-2*r / (1 + 2*alpha_data + 2*beta)))

# Print
errors = errors.cpu().numpy()
mean_errors = np.mean(errors, axis=0)
print('True EE convergence rate is', rate_ee)
print('True FF convergence rate is', rate_ff)
print(mean_errors)
