import yaml
import os

widths = [576,384,288,192]
n_layers_array = [2,3,4,6]
epoch_array = [200,300,400]
batch_sizes = [20,35,50]
lr = 0.001
# Data information
data_path= '/groups/astuart/mtrautne/FNMdata/vor_data_v2v.pkl'
model_name= 'vor_model_dnn_hyp'


# Parameters for training
Ntotal= 10000
N_train= 9500
USE_CUDA: True

index = 1
for i, n_layer in enumerate(n_layers_array):
    for batch_size in batch_sizes:
        for epochs in epoch_array:
                width = widths[i]
                n_layers = n_layer
                epochs = epochs
                b_size = batch_size
                lr = lr
                USE_CUDA = True
                config = {
                    'data_path': data_path,
                    'model_name': 'hyperparam_compare/'+ model_name + '_' + str(index),
                    'Ntotal': Ntotal,
                    'N_train': N_train,
                    'width': width,
                    'epochs': epochs,
                    'batch_size': b_size,
                    'lr': lr,
                    'USE_CUDA': USE_CUDA,
                    'n_layers': n_layers
                }
                with open(model_name + '_' + str(index) + '.yaml', 'w') as f:
                    yaml.dump(config, f)
                index = index + 1

