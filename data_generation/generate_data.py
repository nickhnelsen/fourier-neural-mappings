#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:13:16 2022

@author: nnelsen
"""

import os
from timeit import default_timer
import numpy as np
import fenics as fe
import pyapprox.pde.karhunen_loeve_expansion as kle
import warnings 
warnings.filterwarnings('ignore')
fe.set_log_level(50)


# Convenience class, computes solution and stores QoI; contains evaluation wrapper
class AdvecDiff2D():
    '''
    Wrapper for advection-diffusion in 2D using FEniCS 
    '''
    def __init__(self,
                 vel_kl,
                 num_partitions=32,
                 T=0.75,
                 dt=0.01,
                 store_snapshots=True):
        '''
        Computes solution on num_partitions x num_partitions grid.
        Stores qoi and, if desired, snapshots.
        
        INPUTS:
            vel_kl             : values of 1D VELOCITY vector in x1 coord, arraylike
            num_partitions  : size of PDE mesh, int
            T               : final time, float
            dt              : time step, float
            store_snapshots : flag to store solution on mesh, bool
        '''
        # Mesh
        mesh = fe.UnitSquareMesh(num_partitions, num_partitions)
        V = fe.FunctionSpace(mesh,'Lagrange',2)
        
        # Diffusion coefficient
        D = 0.05

        # Velocity field
        W = fe.VectorFunctionSpace(mesh, 'Lagrange', 2)
        vel = fe.Function(W)
        xvec = fe.Function(W)
        xvec.assign(fe.project(fe.Expression(('x[0]', 'x[1]'), degree=1),W))
        v1 = np.asarray(vel_kl).flatten()
        v1 = np.interp(xvec.vector()[::2], np.linspace(0,1,v1.shape[0]), v1)
        v2 = np.zeros((v1.size,))
        vel.vector()[:] = np.vstack((v1,v2)).T.flatten()
        
        # Gaussian source
        g = fe.Expression('5.0 / (sigma*sigma*2.0*M_PI) * exp(-0.5*(pow((x[0]-0.2)/sigma, 2) + pow((x[1]-0.5)/sigma, 2)))', 
                      degree=10, sigma=0.02)
        
        # Set time-stepping parameters
        Nsteps = int(np.ceil(T/dt))

        # Previous and current solution
        u0 = fe.interpolate(fe.Expression('0', degree=0), V)   # Initialize at initial profile
        u1 = fe.Function(V)

        # Initialize variables for weak formulation
        w = fe.TrialFunction(V)
        v = fe.TestFunction(V)

        # Define weak formulation with linear and bilinear forms
        a = ((w*v) + dt*fe.inner(D*fe.grad(w), fe.grad(v)) + dt*fe.div(vel*w)*v)*fe.dx 
        L = u0*v*fe.dx + dt*g*v*fe.dx

        if store_snapshots:
            self.snapshots = []
            (X,Y) = np.meshgrid(np.linspace(0.0,1.0,num_partitions+1), np.linspace(0.0,1.0,num_partitions+1))
            Xflat, Yflat = X.flatten(), Y.flatten()
            Z = []
            for i in range(X.size):
                Z.append(u0([Xflat[i], Yflat[i]]))
            self.snapshots.append(Z)

        # time stepper
        for i in range(1,Nsteps+1):
            # Solve
            fe.solve(a == L, u1)

            # Update
            u0.assign(u1)

            if store_snapshots:
                Z = []
                for j in range(X.size):
                    Z.append(u0([Xflat[j], Yflat[j]]))
                self.snapshots.append(Z)
        self.u = u0
        self.qoi = u0([0.5, 0.5])
        if store_snapshots:
            self.snapshots = np.asarray(self.snapshots)
        
    def evaluate(self, points):
        '''
        Wrapper to evaluate PDE solution
        
        INPUTS:
            points : evaluation points, arraylike with shape (num_points, 2)
            
        OUTPUTS:
            values of solution, size (num_points)
        '''
        points = np.atleast_2d(np.asarray(points))
        if points.shape[1] != 2:
            raise ValueError('points must have shape (num_points, 2)')
        return np.asarray([self.u(x.tolist()) for x in points]) if points.shape[0] > 1 else self.u(points[0,:].tolist())
    

if __name__ == '__main__':
    
    # USER INPUT
    data_suffix = 'nu_1p5_ell_p25/'     # 'nu_inf_ell_p05/'
    save_suffix = '_TESTvel_bigd/'      # '_scratch6k/'
    d = 1000                     # number of KLE coefficients; d<=1000 should work, but avoid d near K
    nu = 1.5                    # values are 0.5, 1.5, 2.5, np.inf
    ell = 0.25                  # other recommended values: 0.05. 0.25     
    n_train = 10                # number of solves (training sample size)
    SAVE_AFTER = 20             # save qoi every this many solves
    K = 1 + 4096                # velocity 1D high resolution (one plus power of two)
    K_sub = 16                  # subsample factor with respect to K (power of two)
    
    # TODO: sys.argv
    # Process args
    if nu > 2.5: # allow easy way to set nu = np.inf from command line
        nu = np.inf
    if not (nu in [0.5, 1.5, 2.5, np.inf]):
        raise ValueError("nu must be in [0.5, 1.5, 2.5, np.inf]")

    # Check dimensions
    if not ((K-1 & (K-1-1) == 0) and K-1 != 0):
        raise ValueError("K-1 must be power of two")
    if not ((K_sub & (K_sub-1) == 0) and K_sub != 0):
        raise ValueError("K_sub must be power of two")
    if d > K:
        raise ValueError("d cannot be larger than K, the velocity 1D high resolution. We recommend d << K.")

    # File IO
    # data_prefix = '/groups/astuart/nnelsen/data/raise/training/'
    data_prefix = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/training/'
    data_folder = data_prefix + data_suffix
    savepath = data_folder + str(d) + "d" + save_suffix
    os.makedirs(savepath, exist_ok=True)
    
    # Allocate output data arrays
    qoi = np.zeros(n_train)
    
    # Setup KLE
    kl = kle.MeshKLE(mesh_coords=np.linspace(0,1,K)[None, :], mean_field=3, matern_nu=nu)
    kl.matern = kl.matern_nu
    kl.compute_basis(ell, sigma=1, nterms=d)
    
    # Sample the input data measure
    params = np.random.uniform(-1, 1, (n_train, d))
    velocity = kl(params.swapaxes(0, 1))
    velocity = velocity.swapaxes(0, 1)
    
    # Save problem inputs
    np.save(savepath + "params" + ".npy", params)
    np.save(savepath + "velocity" + ".npy", velocity)
    
    # Adjust resolution of velocity input to PDE solver to avoid interp instability
    velocity = velocity[..., ::K_sub]
    
    # Time-step n_train PDEs
    start = default_timer()
    for i, vel in enumerate(velocity):
        t1 = default_timer()
        solution = AdvecDiff2D(vel)
        t2 = default_timer()
        print("Loop", i + 1, "Time", t2-t1)
        qoi[i] = solution.qoi
        np.save(savepath + "state" + str(i) + ".npy", solution.snapshots)
        
        if i % SAVE_AFTER == 0:
            np.save(savepath + "qoi" + ".npy", qoi)
    
    # Last save
    np.save(savepath + "qoi" + ".npy", qoi)
    end = default_timer()
    print("Total time for N =", n_train, "solves is", (end-start)/3600, "hours.")
