import yaml
import numpy as np
import matplotlib.pyplot as plt
import pdb
"""
Read error .yml files and process them into a table for comparison between different hyperparameters
"""
config_c = 36
name_start = 'vor_model_hyp_size_'
errors = np.zeros((config_c, 2))
b_sizes = np.zeros((config_c, 1))
modes = np.zeros((config_c, 1))
epochs = np.zeros((config_c, 1))

for index in range(config_c):
    index = index + 1
    try:
        config = yaml.load(open(name_start + str(index) + '_config.yml', 'r'), Loader=yaml.FullLoader)
        b_sizes[index - 1] = config['batch_size']
        modes[index - 1] = config['N_modes']
        epochs[index - 1] = config['epochs']
        error_dict = yaml.load(open(name_start + str(index) + '_errors.yml', 'r'), Loader=yaml.FullLoader)
        errors[index - 1, 0] = error_dict['Abar_rel_error_med']
        errors[index - 1, 1] = error_dict['Abar_rel_error2_med']
    except:
        print('Error in reading file ' + str(index))
        

err_ind =1
# Create error heatmaps for pairs of hyperparameters
# Batch size vs. modes
fig, ax = plt.subplots(3,2)
ax00 = ax[0,0].scatter(b_sizes, modes, c=errors[:,err_ind ])
ax[0,0].set_xlabel('Batch size')
ax[0,0].set_ylabel('Modes')
# add colorbar
fig.colorbar(ax00,ax = ax[0,0])

# Batch size vs. epochs
ax10 = ax[1,0].scatter(b_sizes, epochs, c=errors[:, err_ind])
ax[1,0].set_xlabel('Batch size')
ax[1,0].set_ylabel('Epochs')
# add colorbar
fig.colorbar(ax10,ax = ax[1,0])

# Epochs vs modes
ax11 = ax[1,1].scatter(epochs, modes, c=errors[:, err_ind])
ax[1,1].set_xlabel('Epochs')
ax[1,1].set_ylabel('Modes')
# add colorbar
fig.colorbar(ax11,ax = ax[1,1])

# Modes vs epochs
ax21 = ax[2,1].scatter(modes, epochs, c=errors[:, err_ind])
ax[2,1].set_xlabel('Modes')
ax[2,1].set_ylabel('Epochs')
# add colorbar
fig.colorbar(ax21,ax = ax[2,1])


# Label axes with modes corresponding to each row/column
# Add space between figures
fig.tight_layout(pad=1.0)
plt.savefig('../../Figures/hyp_f2v_eval.pdf')

# Print minimum errors and corresponding hyperparameters
min_ind = np.argmin(errors[:, err_ind])
print('Minimum error: ' + str(errors[min_ind, err_ind]))
print('Batch size: ' + str(b_sizes[min_ind]))
print('Modes: ' + str(modes[min_ind]))
print('Epochs: ' + str(epochs[min_ind]))
