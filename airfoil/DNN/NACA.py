#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import os, sys; sys.path.append(os.path.join('../..'))
from timeit import default_timer

from models import  DNN
from util import Adam
from util.utilities_module import LpLoss, LppLoss, count_params, validate, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
import torch



device = 'cuda' if torch.cuda.is_available() else 'cpu'


cnx1 = 50
cnx2 = 120
cy = 50

    
prefix = "../../../../data/NACA/"
XC = np.load(prefix+"NACA_Cylinder_X.npy")
YC = np.load(prefix+"NACA_Cylinder_Y.npy")

Pressure = np.load(prefix+"NACA_Cylinder_Q.npy")[:, 3, :, :]
theta = np.load(prefix+"NACA_theta.npy")                    


# In[6]:


# compute drag and lift
def compute_F_coeff(XC, YC, p, cnx1 = 50, cnx2 = 120, cny = 50):
    xx, yy, p = XC[cnx1:-cnx1,0], YC[cnx1:-cnx1,0], p[cnx1:-cnx1,0]
     
    drag  = np.dot(yy[0:cnx2]-yy[1:cnx2+1], (p[0:cnx2] + p[1:cnx2+1])/2.0)
    lift  = np.dot(xx[1:cnx2+1]-xx[0:cnx2], (p[0:cnx2] + p[1:cnx2+1])/2.0)
    F = np.array([drag, lift])
    
    # F_ref = 0.5 rho_oo * u_oo^2 A
    rho_oo = 1.0
    A = 1.0
    u_oo   = 0.8*np.sqrt(1.4*1.0/1.0)
    F_ref  = 0.5*rho_oo*u_oo**2 * A
    
    return F/F_ref


    
n_data = theta.shape[0]
F_coeff = np.zeros((n_data, 2))

for i in range(n_data):
    F_coeff[i, :] = compute_F_coeff(XC[i, :, :], YC[i, :, :], Pressure[i, :, :])
   

# learning_rate = 0.001
# epochs = 1001
# step_size = 100
# gamma = 0.5

learning_rate = 0.0005
epochs = 2001
step_size = 100
gamma = 0.8


n_train = int(sys.argv[1]) #5000
width = int(sys.argv[2]) 
n_layers = int(sys.argv[3]) 

n_test = 400





##############################################################################
# # Fully connected neural network: $\theta \rightarrow F_{coeff}$
batch_size = 64
# In[9]:

for _  in range(5):
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(theta[0:n_train, 1:], dtype=torch.float), torch.tensor(F_coeff[0:n_train, :], dtype=torch.float)), batch_size=batch_size,
                                            shuffle=True)                                 
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(theta[-n_test:, 1:], dtype=torch.float), torch.tensor(F_coeff[-n_test:, :], dtype=torch.float)), batch_size=batch_size,
                                            shuffle=False)

    width_list = [width for i in range(n_layers)]
    width_list.insert(0, 7)
    width_list.append(2)

    model = DNN(width_list).to(device)
    print("FNN #params : ", count_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_data = np.zeros((3, epochs))

    myloss = LpLoss(size_average=False)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            _batch_size = x.shape[0]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            loss = myloss(out.view(_batch_size, -1), y.view(_batch_size, -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                _batch_size = x.shape[0]
                x, y = x.to(device), y.to(device)
                out =  model(x)
                test_l2 += myloss(out.view(_batch_size, -1), y.view(_batch_size, -1)).item()
                
        train_l2 /= n_train
        test_l2  /= n_test

        t2 = default_timer()
        print(ep, t2 - t1, train_l2, test_l2)

        loss_data[:, ep] = ep, train_l2, test_l2
    

