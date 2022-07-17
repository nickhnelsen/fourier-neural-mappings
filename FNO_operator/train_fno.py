import torch
import numpy as np
import os, sys; sys.path.append(os.path.join('..'))
from timeit import default_timer

from models import FNO2d
from util.Adam import Adam
from util.utilities_module import LpLoss, LppLoss, count_params, validate, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)

################################################################
#
# %% user configuration
#
################################################################
# TODO: add command line
# Process command line arguments
print(sys.argv)
# save_prefix = sys.argv[1]   # e.g., robustness, scalability, efficiency
# data_suffix = sys.argv[2]   # e.g., 'nu_inf_ell_p05_torch/' or 'nu_1p5_ell_p25_torch/'
# N_train = int(sys.argv[3])  # training sample size
# d_str = sys.argv[4]         # KLE dimension of training inputs
# sigma = int(sys.argv[5])    # index between 0 and 8 that defines the noise standard deviation

save_prefix = 'robustness_TEST_npy/'    # e.g., robustness, scalability, efficiency
data_suffix = 'nu_inf_ell_p25_torch/'
N_train = 100
d_str = '5'
sigma = 0                   # index between 0 and 8

# TODO: MC loop

# File I/O
data_prefix = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/training/'      # local
# data_prefix = '/groups/astuart/nnelsen/data/raise/training/'                            # HPC
data_prefix_eval = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/validation/'      # local
# data_prefix_eval = '/groups/astuart/nnelsen/data/raise/validation/'                            # HPC
FLAG_save_model = True
FLAG_save_plots = True

# Sample size  
N_test = 100        # number of validation samples to monitor during training

# Resolution subsampling
sub_in = 2**6       # input subsample factor (power of two) from s_max_in = 4097
sub_out = 2**0      # output subsample factor (power of two) from s_max_out = 33

# FNO
modes1 = 12
modes2 = 12
width = 32

# Training
batch_size = 20
epochs = 500
learning_rate = 1e-3
weight_decay = 1e-4
scheduler_step = 100
scheduler_gamma = 0.5

################################################################
#
# %% load and process data
#
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)

# File IO
obj_suffix = '_n' + str(N_train) + '_d' + d_str + '_s' + str(sigma) + '.npy'
data_folder = data_prefix + data_suffix + d_str + 'd/'
savepath = './results/' + save_prefix + data_suffix
os.makedirs(savepath, exist_ok=True)

# Load
y_train = torch.load(data_folder + 'velocity.pt')['velocity'][:,::sub_in]
N_max, s = y_train.shape
assert max(N_train, N_test) <= N_max
x_train = torch.zeros(N_max, 2, s)
x_train[:, 0, :] = y_train
y_train = torch.load(data_folder + 'state.pt')['state'][:,::sub_out,::sub_out,-1] # final time state only

# Shuffle
dataset_shuffle_idx = torch.randperm(N_max)
np.save(savepath + 'idx_shuffle' + obj_suffix, dataset_shuffle_idx.numpy())
# torch.save({'shuffle_idx': dataset_shuffle_idx}, savepath + 'shuffle_idx' + obj_suffix)
x_train = x_train[dataset_shuffle_idx, ...]
y_train = y_train[dataset_shuffle_idx, ...]

# Extract
x_test = x_train[-N_test:,...].unsqueeze(-1).repeat(1, 1, 1, s) # velocity is constant in y=x_2 direction
x_train = x_train[:N_train,...].unsqueeze(-1).repeat(1, 1, 1, s) # velocity is constant in y=x_2 direction
y_test = y_train[-N_test:,...]
y_train = y_train[:N_train,...]

# Noise
stdevs = torch.arange(0.0, 2.01, 0.25)
if sigma not in [i for i in range(stdevs.shape[0])]:
    raise ValueError("sigma must be an index from 0 to 8")
elif sigma > 0: # add noise in the output space
    stdev = stdevs[sigma]
    y_test = y_test + stdev*torch.randn(y_test.shape)
    y_train = y_train + stdev*torch.randn(y_train.shape)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
#
# %% training
#
################################################################
s_outputspace = tuple(y_train.shape[-2:])   # same output shape as the output dataset

model = FNO2d(modes1, modes2, width, s_outputspace).to(device)
print("FNO parameter count:", count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

loss_f = LpLoss(size_average=False)

errors = torch.zeros((epochs, 2))

t0 = default_timer()
for ep in range(epochs):
    t1 = default_timer()

    train_loss = 0.0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = loss_f(out, y)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)

            test_loss += loss_f(out, y).item()

    train_loss /= N_train
    test_loss /= N_test
    
    scheduler.step()

    errors[ep,0] = train_loss
    errors[ep,1] = test_loss

    if FLAG_save_model:
        torch.save(model.state_dict(), savepath + 'model' + obj_suffix[:-3] + 'pt')

    t2 = default_timer()

    print("Epoch:", ep, "Train L2:", train_loss, "Test L2:", test_loss, "Epoch time:", t2-t1)
    np.save(savepath + 'train_errors' + obj_suffix, errors.numpy())
    # torch.save({'train_errors': errors}, savepath + 'train_errors' + obj_suffix)

print("Total time elapsed (min):", (default_timer()-t0)/60., "Total epochs trained:", epochs)

################################################################
#
# %% evaluation
#
################################################################
# File IO
if d_str == '1000':
    d_test_str = '2'    # test on d=2 case to show resolution-invariance
else:
    d_test_str = d_str
data_folder = data_prefix_eval + d_test_str + 'd_torch/'
obj_suffix_eval = '_TESTd' + d_test_str + obj_suffix

# Load
y_test = torch.load(data_folder + 'velocity.pt')['velocity'][:,::sub_in]
N_test_max, s_test = y_test.shape
x_test = torch.zeros(N_test_max, 2, s_test)
x_test[:, 0, :] = y_test
y_test = torch.load(data_folder + 'state.pt')['state'][:,::sub_out,::sub_out,-1] # final time state only
qoi_test = torch.load(data_folder + 'qoi.pt')['qoi']

# Process
s_outputtest = y_test.shape[-2:]
x_test = x_test.unsqueeze(-1).repeat(1, 1, 1, s_test) # velocity is constant in y=x_2 direction
test_loader = DataLoader(TensorDatasetID(x_test, y_test), batch_size=batch_size, shuffle=False)

# Evaluate
model.eval()
loss_vec = LpLoss(size_average=False, reduction=False) # relative L^2 error (not summed)
loss_abs = LppLoss(size_average=False, reduction=True) # absolute squared L^2 error

t1 = default_timer()
er_test_loss = 0.0
num = 0.0
den = 0.0
test_out = torch.zeros(y_test.shape)
qoi_out = torch.zeros(N_test_max)
idx_qoi = torch.div(torch.tensor(s_outputtest), 2, rounding_mode="floor")
errors_test = torch.zeros(y_test.shape[0])
with torch.no_grad():
    for x, y, idx_test in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)

        er_test_loss += loss_f(out, y).item()
        num += loss_abs.abs(out, y).item()
        den += loss_abs.abs(y, 0*y).item()
        
        errors_test[idx_test] = loss_vec(out, y).cpu()
        
        out = out.squeeze().cpu()
        test_out[idx_test, ...] = out
        qoi_out[idx_test] = out[..., idx_qoi[-2], idx_qoi[-1]]

er_test_loss /= N_test_max
er_test_bochner = (num/den)**(0.5)
er_test_qoi = validate(qoi_test, qoi_out)
t2 = default_timer()
print("Time to evaluate", N_test_max, "samples (sec):", t2-t1)
print("Average relative L2 test:", er_test_loss, "Relative Bochner L2 test:", er_test_bochner)
print("Relative L2 QoI test error:", er_test_qoi)

# Save test errors
test_errors = np.array([er_test_qoi, er_test_bochner, er_test_loss])
np.savez(savepath + 'test_errors' + obj_suffix_eval[:-3] + 'npz', qoi_bochner_loss=test_errors,\
         rel_test_error_list=errors_test.numpy())
# torch.save({'qoi_bochner_loss': test_errors, 'test_list': errors_test},\
#            savepath + 'test_errors' + obj_suffix_eval)

# TODO: remove for public version of code
# Evaluate trained model on 2D parameter grid and save result to .pt file
x_tmp = torch.load(data_prefix_eval + '2d_qoi_plot/' + 'velocity.pt')['velocity'][:,::sub_in]
N_grid, s_grid = x_tmp.shape
x_grid = torch.zeros(N_grid, 2, s_grid)
x_grid[:, 0, :] = x_tmp
del x_tmp
x_grid = x_grid.unsqueeze(-1).repeat(1, 1, 1, s_grid) # velocity is constant in y=x_2 direction
grid_loader = DataLoader(TensorDatasetID(x_grid, 0*x_grid), batch_size=batch_size, shuffle=False)
qoi_grid = torch.zeros(x_grid.shape[0])
with torch.no_grad():
    for x, _, idx_grid in grid_loader:
        x = x.to(device)
        qoi_grid[idx_grid] = model(x).squeeze().cpu()[..., idx_qoi[-2], idx_qoi[-1]]
np.save(savepath + 'qoi_grid' + obj_suffix, qoi_grid.numpy())
# torch.save({'qoi_grid': qoi_grid}, savepath + 'qoi_grid' + obj_suffix)

################################################################
#
# %% plotting
#
################################################################
if FLAG_save_plots:
    from util.configure_plots import plt, mpl
    from mpl_toolkits.axes_grid1 import ImageGrid

    plt.rcParams['figure.figsize'] = [6.0, 4.0]
    plt.rcParams['font.size'] = 16
    plt.rcParams['figure.dpi'] = 250
    plt.rcParams['savefig.dpi'] = 250
    
    plot_folder = savepath + "figures/"
    os.makedirs(plot_folder, exist_ok=True)
    
    # Plot train and test errors
    plt.close()
    plt.semilogy(errors)
    plt.grid()
    plt.xlim(0, epochs)
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Error')
    plt.legend(["Train", "Test"])
    plt.savefig(plot_folder + "epochs" + obj_suffix[:-3] + "pdf", format='pdf')
    
    # Make 2D QoI grid plot
    grid = torch.load(data_prefix_eval + '2d_qoi_plot/' + 'params.pt')['params']
    grid = grid.reshape(s_outputtest[-2], s_outputtest[-1], -1)
    plt.close()
    plt.contourf(grid[...,-2], grid[...,-1],\
                  qoi_grid.reshape(s_outputtest), cmap=mpl.cm.viridis)
    plt.title(r'FNO $(N=%d, d_{\mathrm{tr}}=%d, \sigma=0)$' % (N_train, int(d_str)))
    plt.xlabel(r'$\xi_1$')
    plt.ylabel(r'$\xi_2$')
    plt.colorbar(label=r'QoI')
    plt.savefig(plot_folder + 'qoi_grid' + obj_suffix[:-3] + 'pdf', format='pdf')
    
    # Begin three errors plots
    idx_worst = torch.argmax(errors_test).item()
    idx_median = torch.argsort(errors_test)[errors_test.shape[0]//2].item()
    idx_best = torch.argmin(errors_test).item()
    idxs = [idx_worst, idx_median, idx_best]
    np.save(savepath + 'idx_min_med_max' + obj_suffix_eval, np.array(idxs))
    names = ["worst", "median", "best"]
    XX = torch.linspace(0, 1, y_test.shape[-1])
    (YY, XX) = torch.meshgrid(XX, XX)
    for i, idx in enumerate(idxs):
        true_testsort = y_test[idx,...].squeeze()
        plot_testsort = test_out[idx,...].squeeze()
        er_testsort = torch.abs(plot_testsort - true_testsort).squeeze()
        
        plt.close()
        fig = plt.figure(figsize=(15, 15))
        grid = ImageGrid(fig, 211,
                         nrows_ncols=(1,2),
                         axes_pad=1.5,
                         share_all=True,
                         aspect=True,
                         label_mode="L",
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.2
                         )
        
        grid[0].contourf(XX, YY, plot_testsort, vmin=0, vmax=15, cmap=mpl.cm.viridis)
        grid[0].set_title('FNO Output', fontsize=16)
        grid[0].set_xlabel(r'$x_1$')
        grid[0].set_ylabel(r'$x_2$')
        
        ax01 = grid[1].contourf(XX, YY, true_testsort, vmin=0, vmax=15, cmap=mpl.cm.viridis)
        grid[1].set_title('True Final State', fontsize=16)
        grid[1].set_xlabel(r'$x_1$')
        grid[1].axes.yaxis.set_visible(False)
        cb01 = grid[1].cax.colorbar(ax01)
        grid[1].cax.toggle_label(True)
        cb01.set_label(r'State')
        
        grid = ImageGrid(fig, 223,
                         nrows_ncols=(1,1),
                         aspect=False,
                         label_mode="all"
                         )
        
        grid[0].plot(torch.linspace(0, 1, s_test), x_test[idx, 0, :, 0].squeeze())
        grid[0].grid(visible=True)
        grid[0].set_xlim(0,1)
        grid[0].set_ylim(1,5)
        grid[0].set_title('Velocity Input Profile ' + '(' + names[i] + ')', fontsize=16)
        grid[0].set_xlabel(r'$x_1$')
        grid[0].set_ylabel(r'$v_1(x_1, 0; \xi)$')
        
        grid = ImageGrid(fig, 224,
                         nrows_ncols=(1,1),
                         aspect=True,
                         label_mode="all",
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.2
                         )
  
        ax11 = grid[0].imshow(er_testsort, origin="lower", interpolation="spline16",\
                               extent=(0,1,0,1), cmap=mpl.cm.viridis)
        grid[0].set_title('Pointwise Absolute Error', fontsize=16)
        grid[0].set_xlabel(r'$x_1$')
        grid[0].set_ylabel(r'$x_2$')
        cb11 = grid[0].cax.colorbar(ax11)
        grid[0].cax.toggle_label(True)
        cb11.set_label(r'Error')
        
        # Save min, median, max error plots (contour)
        plt.savefig(plot_folder + "eval_" + names[i] + obj_suffix[:-3] + "pdf", format='pdf')
    
    # Close open figure if running interactively
    plt.close()
