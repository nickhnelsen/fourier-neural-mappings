import numpy as np
import os

import pyapprox.pde.karhunen_loeve_expansion as kle

data_folder = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/validation/'

K = 1 + 4096
nu = np.inf     # other values include 0.5, 1.5, 2.5
ell = 0.25      # for operator learning training, use 0.05 KLE lengthscale

kl = kle.MeshKLE(mesh_coords=np.linspace(0,1,K)[None, :], mean_field=3, matern_nu=nu)
kl.matern = kl.matern_nu

dim_list = [1, 2, 5, 10, 15, 20]
name = "params"
savename = "velocity"
for dim in dim_list:
    kl.compute_basis(ell, sigma=1, nterms=dim)
    datapath = data_folder + str(dim) + "d/"
    savepath = data_folder + str(dim) + "d_torch/"
    # os.makedirs(savepath, exist_ok=True)
    xi_all = np.loadtxt(datapath + name + ".txt")
    if dim == 1:
        xi_all = xi_all[:,None]
    xi_all = xi_all.swapaxes(0, 1)
    vel = kl(xi_all)
    vel = vel.swapaxes(0, 1)
    np.save(savepath + savename + ".npy", vel)
  
# d=2 QoI input data for plot
nx = 33
dd = 2
savepath = data_folder + str(dd) + "d_qoi_plot/"
os.makedirs(savepath, exist_ok=True)
kl = kle.MeshKLE(mesh_coords=np.linspace(0,1,K)[None, :], mean_field=3, matern_nu=nu)
kl.matern = kl.matern_nu
kl.compute_basis(ell, sigma=1, nterms=dd)
X = np.linspace(-1., 1., nx)
X, Y = np.meshgrid(X, X)
xi_scat = np.array((X.flatten(), Y.flatten()))
np.save(savepath + "params" + ".npy", xi_scat.swapaxes(0, 1)) # invert with np.reshape(nx, nx, -1)
vel_plot = kl(xi_scat)
vel_plot = vel_plot.swapaxes(0, 1)
np.save(savepath + savename + ".npy", vel_plot)