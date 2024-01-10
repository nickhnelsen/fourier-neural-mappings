import yaml
import os

modes = 12
batch_size = 15
epochs = 300
lr = 0.001
# Data information
data_path= '/groups/astuart/mtrautne/FNMdata/vor_data_Abar.pkl'
model_name= 'vor_f2v_data_size'

# Parameters for training
Ntotal= 10000
data_sizes = [10,50,250,1000,2000, 4000, 6000, 8000, 9500]
USE_CUDA: True

index = 1

for data_size in data_sizes:
    width = 1152//modes
    N_modes = modes
    epochs = epochs
    b_size = batch_size
    lr = lr
    USE_CUDA = True
    config = {
        'data_path': data_path,
        'model_name': 'hyperparam_compare/'+ model_name + '_' + str(data_size),
        'Ntotal': Ntotal,
        'N_train': data_size,
        'N_modes': N_modes,
        'width': width,
        'epochs': epochs,
        'batch_size': b_size,
        'lr': lr,
        'USE_CUDA': USE_CUDA
    }
    with open(model_name + '_' + str(data_size) + '.yaml', 'w') as f:
        yaml.dump(config, f)
    index = index + 1

