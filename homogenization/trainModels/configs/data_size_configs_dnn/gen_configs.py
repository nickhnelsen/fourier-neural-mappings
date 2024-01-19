import yaml
import os

width = 576
n_layers = 2
epochs = 200
batch_size = 35
lr = 0.001
# Data information
data_path= '/groups/astuart/mtrautne/FNMdata/vor_data_v2v.pkl'
model_name= 'vor_model_dnn_size'


# Parameters for training
Ntotal= 10000
N_train= 9500
USE_CUDA: True

data_sizes  = [10,50,250,1000,2000,4000,6000,8000,9500]
index = 1
for data_size in data_sizes:
    b_size = batch_size
    lr = lr
    USE_CUDA = True
    config = {
        'data_path': data_path,
        'model_name': 'size_compare/'+ model_name + '_' + str(data_size),
        'Ntotal': Ntotal,
        'N_train': N_train,
        'width': width,
        'epochs': epochs,
        'batch_size': b_size,
        'lr': lr,
        'USE_CUDA': USE_CUDA,
        'n_layers': n_layers
    }
    with open(model_name + '_' + str(data_size) + '.yaml', 'w') as f:
        yaml.dump(config, f)
    index = index + 1

