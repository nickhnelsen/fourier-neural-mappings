import torch
import numpy as np
import os, sys; sys.path.append(os.path.join('..'))
from util.utilities_module import to_torch


# Process command line arguments
print(sys.argv)
load_suffix = sys.argv[1]   # e.g., 'nu_inf_ell_p05/' or 'nu_1p5_ell_p25/'  

# Defaults
data_prefix = '/groups/astuart/nnelsen/data/raise/training/'
save_suffix = load_suffix[:-1] + "_torch/"
K = 33
T = 76

# Save inputs and qoi
dim_list = [1, 2, 5, 10, 15, 20, 1000]
name_list = ["params", "qoi", "velocity"]
for dim in dim_list:
    loadpath = data_prefix + load_suffix + str(dim) + "d/"
    savepath = data_prefix + save_suffix + str(dim) + "d/"
    os.makedirs(savepath, exist_ok=True)
    for name in name_list:
        x = np.load(loadpath + name + ".npy")
        x = to_torch(x)
        torch.save({name: x}, savepath + name + ".pt")

# Save time series
for dim in dim_list:
    N_train = 12000 if dim in [2, 20, 1000] else 6000
    x_all = torch.zeros(N_train, K, K, T)
    num_list = list(np.arange(N_train))
    loadpath = data_prefix + load_suffix + str(dim) + "d/"
    savepath = data_prefix + save_suffix + str(dim) + "d/"
    os.makedirs(savepath, exist_ok=True)
    for num in num_list:
        x = np.load(loadpath + "state" + str(num) + ".npy")
        x = x.reshape(T, K, K)
        x = to_torch(x)
        x = x.permute(1, 2, 0)
        x_all[num, ...] = x
    torch.save({"state": x_all}, savepath + "state" + ".pt")
    print("state saved for d = ", dim)
