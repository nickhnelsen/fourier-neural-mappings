import torch
import numpy as np
import os

def to_torch(x, to_float=True):
    if to_float:
        if np.iscomplexobj(x):
            x = x.astype(np.complex64)
        else:
            x = x.astype(np.float32)
    return torch.from_numpy(x)

# data_folder = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/validation/'

K = 33
N_test = 2000
T = 76

dim_list = [1, 2, 5, 10, 15, 20]
name_list = ["params", "qoi"]
for dim in dim_list:
    datapath = data_folder + str(dim) + "d/"
    savepath = data_folder + str(dim) + "d_torch/"
    os.makedirs(savepath, exist_ok=True)
    for name in name_list:
        x = np.loadtxt(datapath + name + ".txt")
        x = to_torch(x)
        torch.save({name: x}, savepath + name + ".pt")

# Save time series
num_list = list(np.arange(N_test))
for dim in dim_list:
    x_all = torch.zeros(N_test, K, K, T)
    datapath = data_folder + str(dim) + "d/"
    savepath = data_folder + str(dim) + "d_torch/"
    os.makedirs(savepath, exist_ok=True)
    for num in num_list:
        x = np.loadtxt(datapath + str(num) + ".txt")
        x = x.reshape(T, K, K)
        x = to_torch(x)
        x = x.permute(1, 2, 0)
        x_all[num, ...] = x
    torch.save({"state": x_all}, savepath + "state" + ".pt")

# Process velocity
name = "velocity"
for dim in dim_list:
    savepath = data_folder + str(dim) + "d_torch/"
    datapath = savepath
    os.makedirs(savepath, exist_ok=True)
    x = np.load(datapath + name + ".npy")
    x = to_torch(x)
    torch.save({name: x}, savepath + name + ".pt")