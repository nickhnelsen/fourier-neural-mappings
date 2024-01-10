import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import pickle as pkl
import pdb
import numpy as np
import scipy.io
import sys
import torch.utils.data
from tqdm import tqdm
import yaml
import json
import os
import sys
# add parent directories to path
sys.path.append('../')
sys.path.append('../../')

from models.func_to_vec2d import FNF2d
from util_homogenization.utilities_module import *

def train_model(config):
    # Take in user arguments
    data_path = config['data_path']
    model_name = config['model_name']
    Ntotal = config['Ntotal']
    N_train = config['N_train']
    N_modes = config['N_modes']
    width = config['width']
    epochs = config['epochs']
    b_size = config['batch_size']
    lr = config['lr']
    USE_CUDA = config['USE_CUDA']

    if USE_CUDA:
        gc.collect()
        torch.cuda.empty_cache()

    model_info_path = 'Trained_Models/' + model_name + '_info.pkl'
    model_path = 'Trained_Models/' + model_name
    
    with open(data_path, 'rb') as handle:
        A_input, Abar = pkl.load(handle)

    # Abar shape (num_data, 2,2)


    sgc = 128 


    (N_data, N_nodes,dummy1, dummy2) = np.shape(A_input)
  
    train_size = N_train
    test_start = Ntotal - 500
    test_end = Ntotal

    N_test = test_end - test_start

    data_output = np.reshape(Abar, (N_data,4))

    print(data_output[0,:])
    data_input = np.reshape(A_input, (N_data,sgc, sgc,4))
    data_input = np.delete(data_input,2,axis = 3) # Symmetry of A: don't need both components

    # Input shape (of x): (batch, channels_in, nx_in, ny_in)
    data_input = np.transpose(data_input, (0,3,1,2))

    # Output shape:       (batch, channels_out)


    #================== TRAINING ==================#

    # Split data into training and testing
    y_train = data_output[:train_size,:]
    y_test = data_output[test_start:test_end,:]

    x_train = data_input[:train_size,:,:,:]
    x_test = data_input[test_start:test_end,:,:,:]

    # Set loss function to be H1 loss
    loss_func = frob_loss

    # Convert to torch arrays
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Specify pointwise degrees of freedom
    d_in = 3 # A \in \R^{2 \times 2}_sym
    d_out = 4 # \chi \in \R^2

    # Initialize model
    net = FNF2d(modes1 = N_modes, modes2= N_modes, width = width, d_in = d_in, d_out = d_out)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)

    if USE_CUDA:
        net.cuda()
    
    # Wrap training data in loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=b_size,
                                           shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,y_test), batch_size = b_size, shuffle = False)
    
    # paths
    model_path = '/groups/astuart/mtrautne/FNM/trainedModels/' + model_name
    model_info_path = 'trainedModels/' + model_name + '_config.yml'
    
    # Train net
    train_err = np.zeros((epochs,))
    test_err = np.zeros((epochs,))
    y_test_approx_all = torch.zeros(test_end-test_start,d_out)


    for ep in tqdm(range(epochs)):
        train_loss = 0.0
        test_loss  = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            y_approx = net(x).squeeze()
        
            # For forward net: 
            # Input shape (of x):     (batch, channels_in, nx_in, ny_in)
            # Output shape:           (batch, channels_out, nx_out, ny_out)
            
            # The input resolution is determined by x.shape[-2:]
            # The output resolution is determined by self.s_outputspace
            loss = loss_func(y_approx,y)
            loss.backward()
            train_loss = train_loss + loss.item()

            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            b=0
            for x,y in test_loader:
                if USE_CUDA:
                    x = x.cuda()
                    y = y.cuda()
                y_test_approx = net(x)
                t_loss = loss_func(y_test_approx,y)
                test_loss = test_loss + t_loss.item()
                if ep == epochs - 1:
                    y_test_approx_all[b*b_size:(b+1)*b_size,:] = torch.squeeze(y_test_approx).cpu()
                    b = b+1

        train_err[ep] = train_loss/len(train_loader)
        test_err[ep]  = test_loss/len(test_loader)
        print(ep, train_err[ep],test_err[ep])

    # Save model
    model_path = '/groups/astuart/mtrautne/FNM/FourierNeuralMappings/homogenization/trainModels/trainedModels/' + model_name
    torch.save({'epoch': epochs,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_err,
            'test_loss_history': test_err,
            }, model_path)

    # Save model info
    model_info_path = 'trainedModels/' + model_name + '_config.yml'
    # save model config
    # convert config to a dict that will be readable when saved as a .json
    
    with open(model_info_path, 'w') as fp:
        yaml.dump(config, fp)

    # Compute and save errors
    model_path = 'trainedModels/' + model_name
    loss_report_f2v(y_test_approx_all, y_test, x_test, model_path)
    

if __name__ == "__main__":
    # Take in user arguments 
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Check if there's a second argument
    if len(sys.argv) > 2:
        model_index = sys.argv[2]
        config['model_name'] = config['model_name'] + '_' + str(model_index)
    
    train_model(config)
