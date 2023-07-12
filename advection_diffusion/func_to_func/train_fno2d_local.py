import torch
import numpy as np
import os, sys; sys.path.append(os.path.join('../..'))
from timeit import default_timer

from models import FNO2d as my_model
from util import Adam
from util.utilities_module import LpLoss, LppLoss, count_params, validate, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)

from advection_diffusion.helpers import get_qoi, trapz2, process_velocity

################################################################
#
# %% user configuration
#
################################################################
print(sys.argv)

save_prefix = 'FNM_TEST_LAYERS2/'    # e.g., 'robustness/', 'scalability/', 'efficiency/'
data_suffix = 'nu_1p5_ell_p25_torch/' # 'nu_1p5_ell_p25_torch/', 'nu_inf_ell_p05_torch/'
N_train = 100
d_str = '1000'

# File I/O
data_prefix = '/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/'      # local
FLAG_save_model = not True
FLAG_save_plots = True

# Sample size  
N_test = 500        # number of validation samples to monitor during training

# Resolution subsampling
sub_in = 2**6       # input subsample factor (power of two) from s_max_in = 4097
sub_out = 2**0      # output subsample factor (power of two) from s_max_out = 33

# FNO
modes1 = 12
modes2 = 12
width = 32
d_in = 1
n_layers = 2

# Training
batch_size = 20
epochs = 502
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
obj_suffix = '_n' + str(N_train) + '_d' + d_str + '.npy'
data_folder = data_prefix + data_suffix + 'training/' + d_str + 'd/'
data_folder_test = data_prefix + data_suffix + 'testing/' + d_str + 'd/'
savepath = './results/' + save_prefix + data_suffix
os.makedirs(savepath, exist_ok=True)

# Load train
y_train = torch.load(data_folder + 'velocity.pt')['velocity'][:,::sub_in].clone()
N_max = y_train.shape[0]
assert N_train <= N_max
x_train = y_train.unsqueeze(1)
y_train = torch.load(data_folder + 'state.pt')['state'][:,::sub_out,::sub_out].clone()

# Shuffle training set selection
dataset_shuffle_idx = torch.randperm(N_max)
np.save(savepath + 'idx_shuffle' + obj_suffix, dataset_shuffle_idx.numpy())
x_train = x_train[dataset_shuffle_idx, ...]
y_train = y_train[dataset_shuffle_idx, ...]

# Extract
x_train = process_velocity(x_train[:N_train,...], True) # velocity is constant in y=x_2 direction
y_train = y_train[:N_train,...]

# Load QoI test data
qoi_test_all = torch.load(data_folder_test + 'qoi.pt')['qoi']
N_qoi = qoi_test_all.shape[-1]

# Load validation test data to monitor during training
x_test_all = torch.load(data_folder_test + 'velocity.pt')['velocity'].clone().unsqueeze(1)
x_test = x_test_all[...,::sub_in]
N_test_max = x_test.shape[0]
assert N_test < N_test_max
y_test_all = torch.load(data_folder_test + 'state.pt')['state'].clone()

x_test = process_velocity(x_test[-N_test:,...], True)
y_test = y_test_all[-N_test:,::sub_out,::sub_out]

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
#
# %% training
#
################################################################
s_outputspace = tuple(y_train.shape[-2:])   # same output shape as the output dataset

model = my_model(modes1, modes2, width, s_outputspace, d_in=d_in, n_layers=n_layers).to(device)
print(model)
print("FNO parameter count:", count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=scheduler_step,
                                            gamma=scheduler_gamma)

loss_f = LpLoss(size_average=False)

errors = torch.zeros((epochs, 2 + 2 + 2*N_qoi))

t0 = default_timer()
for ep in range(epochs):
    t1 = default_timer()

    train_loss = 0.0
    train_qoi = 0.0
    train_qoi_vec = torch.zeros((N_qoi), device=device)
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = loss_f(out, y)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        
        with torch.no_grad():
            out_qoi = get_qoi(out.detach()).squeeze()
            y_qoi = get_qoi(y.detach()).squeeze()
            train_qoi += loss_f(out_qoi, y_qoi).item()
            train_qoi_vec += torch.sum(torch.abs(out_qoi - y_qoi) / torch.abs(y_qoi), dim=0)

    model.eval()
    test_loss = 0.0
    test_qoi = 0.0
    test_qoi_vec = torch.zeros((N_qoi), device=device)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)

            test_loss += loss_f(out, y).item()
            
            out_qoi = get_qoi(out).squeeze()
            y_qoi = get_qoi(y).squeeze()
            test_qoi += loss_f(out_qoi, y_qoi).item()
            test_qoi_vec += torch.sum(torch.abs(out_qoi - y_qoi) / torch.abs(y_qoi), dim=0)

    train_loss /= N_train
    test_loss /= N_test
    train_qoi /= N_train
    test_qoi /= N_test
    train_qoi_vec /= N_train
    test_qoi_vec /= N_test
    
    scheduler.step()

    errors[ep,0] = train_loss
    errors[ep,1] = test_loss
    errors[ep,2] = train_qoi
    errors[ep,3] = test_qoi
    for i in range(N_qoi):    
        errors[ep,4+2*i:5+2*i+1] = torch.tensor([train_qoi_vec[i], test_qoi_vec[i]]).cpu()

    if FLAG_save_model:
        torch.save(model.state_dict(), savepath + 'model' + obj_suffix[:-3] + 'pt')

    t2 = default_timer()

    print("Epoch:", ep, "Epoch time:", t2-t1)
    print("^Train L2:", train_loss, "Test L2:", test_loss, "Train QoI:", train_qoi, "Test QoI:", test_qoi)
    np.save(savepath + 'train_errors' + obj_suffix, errors.numpy())

print("Total time elapsed (min):", (default_timer()-t0)/60., "Total epochs trained:", epochs)

################################################################
#
# %% evaluation
#
################################################################
# File IO
d_test_str = d_str
obj_suffix_eval = '_TESTd' + d_test_str + obj_suffix

# Use all test data
qoi_test = qoi_test_all[:-N_test,...]
N_eval = N_test_max - N_test
x_test = process_velocity(x_test_all[:-N_test,...,::sub_in], True)
y_test = y_test_all[:-N_test,...,::sub_out,::sub_out]
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
qoi_out = torch.zeros(N_eval, N_qoi)
errors_test = torch.zeros(y_test.shape[0])
with torch.no_grad():
    for x, y, idx_test in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)

        er_test_loss += loss_f(out, y).item()
        num += loss_abs.abs(out, y).item()
        den += loss_abs.abs(y, 0*y).item()
        
        errors_test[idx_test] = loss_vec(out, y).cpu()
        
        out = out.squeeze()
        qoi_out[idx_test, ...] = get_qoi(out).cpu()
        test_out[idx_test, ...] = out.cpu()

# State error
er_test_loss /= N_eval
er_test_bochner = (num/den)**(0.5)

# QoI errors
er_test_qoi_bochner = validate(qoi_test, qoi_out)
er_test_qoi_loss = torch.mean(torch.linalg.norm(qoi_test - qoi_out, dim=-1)
                              / torch.linalg.norm(qoi_test, dim=-1), dim=0).item()
er_test_qoi_bochner_vec = torch.linalg.norm(qoi_test - qoi_out, dim=0) \
                            / torch.linalg.norm(qoi_test, dim=0)
er_test_qoi_loss_vec = torch.mean(torch.abs(qoi_test - qoi_out)
                              / torch.abs(qoi_test), dim=0)

t2 = default_timer()
print("Time to evaluate", N_eval, "samples (sec):", t2-t1)
print("Average relative L2 test:", er_test_loss)
print("Relative Bochner L2 test:", er_test_bochner)
print("QoI average relative L2 test (total):", er_test_qoi_loss)
print("QoI relative Bochner L2 test (total):", er_test_qoi_bochner)
print("QoI average relative test:", er_test_qoi_loss_vec)
print("QoI relative Bochner test:", er_test_qoi_bochner_vec)

# Save test errors
test_errors = np.array([er_test_qoi_bochner, er_test_qoi_loss, er_test_bochner, er_test_loss])
qoi_errors = torch.cat((er_test_qoi_bochner_vec[:,None], er_test_qoi_loss_vec[:,None]), dim=-1)
np.savez(savepath + 'test_errors' + obj_suffix_eval[:-3] + 'npz', 
         qoib_qoil_bochner_loss=test_errors, 
         rel_test_error_list=errors_test.numpy(),
         qoibl_vec=qoi_errors.numpy())

################################################################
#
# %% plotting
#
################################################################
if FLAG_save_plots:
    from util import plt, mpl
    from mpl_toolkits.axes_grid1 import ImageGrid

    plt.rcParams['figure.figsize'] = [6.0, 4.0]
    plt.rcParams['font.size'] = 16
    plt.rcParams['figure.dpi'] = 250
    plt.rcParams['savefig.dpi'] = 250
    
    plot_folder = savepath + "figures/"
    os.makedirs(plot_folder, exist_ok=True)
    
    # Plot train and test errors
    plt.close()
    plt.semilogy(errors[...,:2])
    plt.grid(True, which="both")
    plt.xlim(0, epochs)
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Error')
    plt.legend(["Train", "Val"])
    plt.savefig(plot_folder + "epochs" + obj_suffix[:-3] + "pdf", format='pdf')
        
    plt.close()
    plt.semilogy(errors[...,2:4])
    plt.grid(True, which="both")
    plt.xlim(0, epochs)
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Error')
    plt.legend(["Train QoI All", "Val QoI All"])
    plt.savefig(plot_folder + "epochs_qoiALL" + obj_suffix[:-3] + "pdf", format='pdf')
    
    for i in range(N_qoi):    
        plt.close()
        plt.semilogy(errors[...,4+2*i:5+2*i+1])
        plt.grid(True, which="both")
        plt.xlim(0, epochs)
        plt.xlabel(r'Epoch')
        plt.ylabel(r'Error')
        plt.legend(["Train QoI-" + str(i), "Val QoI-" + str(i)])
        plt.savefig(plot_folder + "epochs_qoi" + str(i) + obj_suffix[:-3] + "pdf", format='pdf')
    
    # Begin five errors plots
    idx_worst = torch.argmax(errors_test).item()
    idx_Q3 = torch.argsort(errors_test)[errors_test.shape[0]*3//4].item()
    idx_median = torch.argsort(errors_test)[errors_test.shape[0]//2].item()
    idx_Q1 = torch.argsort(errors_test)[errors_test.shape[0]//4].item()
    idx_best = torch.argmin(errors_test).item()
    idxs = [idx_worst, idx_Q3, idx_median, idx_Q1, idx_best]
    np.save(savepath + 'idx_min_Q1_med_Q3_max' + obj_suffix_eval, np.array(idxs))
    names = ["worst", "3rd quartile", "median", "1st quartile", "best"]
    XX = torch.linspace(0, 1, y_test.shape[-1])
    (YY, XX) = torch.meshgrid(XX, XX)
    for i, idx in enumerate(idxs):
        true_testsort = y_test[idx,...].squeeze()
        plot_testsort = test_out[idx,...].squeeze()
        er_testsort = torch.abs(plot_testsort - true_testsort).squeeze() \
            / trapz2(true_testsort.squeeze()**2, dx=1./(true_testsort.shape[-2] - 1),
                     dy=1./(true_testsort.shape[-1] - 1))**0.5
        
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
        
        grid[0].plot(torch.linspace(0, 1, x_test_all.shape[-1]), x_test_all[idx, 0, :].squeeze())
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
        grid[0].set_title('Normalized Pointwise Error', fontsize=16)
        grid[0].set_xlabel(r'$x_1$')
        grid[0].set_ylabel(r'$x_2$')
        cb11 = grid[0].cax.colorbar(ax11)
        grid[0].cax.toggle_label(True)
        cb11.set_label(r'Error')
        
        # Save min, median, max error plots (contour)
        plt.savefig(plot_folder + "eval_" + names[i] + obj_suffix_eval[:-3] + "pdf", format='pdf')
    
    # Close open figure if running interactively
    plt.close()
