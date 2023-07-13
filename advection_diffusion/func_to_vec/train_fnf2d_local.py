import torch
import numpy as np
import os, sys; sys.path.append(os.path.join('../..'))
from timeit import default_timer

from models import FNF2d as my_model
from util import Adam
from util.utilities_module import LpLoss, count_params, validate, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)

from advection_diffusion.helpers import process_velocity

################################################################
#
# %% user configuration
#
################################################################
print(sys.argv)

save_prefix = 'FNM_TEST_FNF2d/'    # e.g., 'robustness/', 'scalability/', 'efficiency/'
data_suffix = 'nu_1p5_ell_p25_torch/' # 'nu_1p5_ell_p25_torch/', 'nu_inf_ell_p05_torch/'
N_train = 500
d_str = '1000'
FLAG_2D = True

# File I/O
data_prefix = '/media/nnelsen/SharedHDD2TB/datasets/FNM/low_res/'      # local
FLAG_save_model = not True
FLAG_save_plots = True

# Sample size  
N_test = 500        # number of validation samples to monitor during training

# Resolution subsampling
sub_in = 2**6       # input subsample factor (power of two) from s_max_in = 4097

# QoI indices (0 through 5) to learn
idx_qoi = [0, 1, 2, 3, 4, 5]
# idx_qoi = [1,2,3,4,5]
# idx_qoi = [3]
# idx_qoi = [3,4]

# FNF
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
obj_suffix = ''.join(str(x) for x in idx_qoi)
obj_suffix = '_n' + str(N_train) + '_d' + d_str + '_qoi' + obj_suffix + '.npy'
data_folder = data_prefix + data_suffix + 'training/' + d_str + 'd/'
data_folder_test = data_prefix + data_suffix + 'testing/' + d_str + 'd/'
savepath = './results/' + save_prefix + data_suffix
os.makedirs(savepath, exist_ok=True)

# Load train
y_train = torch.load(data_folder + 'velocity.pt')['velocity'][:,::sub_in].clone()
N_max = y_train.shape[0]
assert N_train <= N_max
x_train = y_train.unsqueeze(1)
y_train = torch.load(data_folder + 'qoi.pt')['qoi'][..., idx_qoi].clone()

# Shuffle training set selection
dataset_shuffle_idx = torch.randperm(N_max)
np.save(savepath + 'idx_shuffle' + obj_suffix, dataset_shuffle_idx.numpy())
x_train = x_train[dataset_shuffle_idx, ...]
y_train = y_train[dataset_shuffle_idx, ...]

# Extract
x_train = process_velocity(x_train[:N_train,...], FLAG_2D) # velocity is constant in y=x_2 direction
y_train = y_train[:N_train,...]

# Load validation test data to monitor during training
x_test_all = torch.load(data_folder_test + 'velocity.pt')['velocity'].clone().unsqueeze(1)
x_test = x_test_all[...,::sub_in]
N_test_max = x_test.shape[0]
assert N_test <= N_test_max
y_test_all = torch.load(data_folder_test + 'qoi.pt')['qoi'][..., idx_qoi].clone()

x_test = process_velocity(x_test[-N_test:,...], FLAG_2D)
y_test = y_test_all[-N_test:,...]

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
#
# %% training
#
################################################################
N_qoi = len(idx_qoi)

model = my_model(modes1, modes2, width, d_in=d_in, d_out=N_qoi, n_layers=n_layers).to(device)
print(model)
print("FNF parameter count:", count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=scheduler_step,
                                            gamma=scheduler_gamma)

loss_f = LpLoss(size_average=False)

FLAG_ONE = N_qoi==1
errors = torch.zeros((epochs, 2 + 2*N_qoi*(1 - FLAG_ONE)))

t0 = default_timer()
for ep in range(epochs):
    t1 = default_timer()

    train_loss = 0.0
    if not FLAG_ONE:
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
        
        if not FLAG_ONE:
            with torch.no_grad():
                out_qoi = out.detach()
                y_qoi = y.detach()
                train_qoi_vec += torch.sum(torch.abs(out_qoi - y_qoi) / torch.abs(y_qoi), dim=0)

    model.eval()
    test_loss = 0.0
    if not FLAG_ONE:
        test_qoi_vec = torch.zeros((N_qoi), device=device)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)

            test_loss += loss_f(out, y).item()
            
            if not FLAG_ONE:
                test_qoi_vec += torch.sum(torch.abs(out - y) / torch.abs(y), dim=0)

    train_loss /= N_train
    test_loss /= N_test
    
    scheduler.step()

    errors[ep,0] = train_loss
    errors[ep,1] = test_loss
    if FLAG_ONE:
        errors = errors[...,:2]
    else:
        train_qoi_vec /= N_train
        test_qoi_vec /= N_test
        for i in range(N_qoi):    
            errors[ep,2+2*i:3+2*i+1] = torch.tensor([train_qoi_vec[i], test_qoi_vec[i]]).cpu()

    if FLAG_save_model:
        torch.save(model.state_dict(), savepath + 'model' + obj_suffix[:-3] + 'pt')

    t2 = default_timer()

    print("Epoch:", ep, "Train L2:", train_loss, "Test L2:", test_loss, "Epoch time:", t2-t1)
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
N_eval = N_test_max - N_test
x_test = process_velocity(x_test_all[:-N_test,...,::sub_in], FLAG_2D)
y_test = y_test_all[:-N_test,...]
test_loader = DataLoader(TensorDatasetID(x_test, y_test), batch_size=batch_size, shuffle=False)

# Evaluate
model.eval()
loss_vec = LpLoss(size_average=False, reduction=False) # relative L^2 error (not summed)

t1 = default_timer()
er_test_loss = 0.0
qoi_out = torch.zeros(N_eval, N_qoi)
errors_test = torch.zeros(y_test.shape[0])
with torch.no_grad():
    for x, y, idx_test in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)

        er_test_loss += loss_f(out, y).item()
        
        errors_test[idx_test] = loss_vec(out, y).cpu()
        
        qoi_out[idx_test, ...] = out.cpu()

# QoI errors
er_test_bochner = validate(y_test, qoi_out)
er_test_loss /= N_eval

t2 = default_timer()
print("Time to evaluate", N_eval, "samples (sec):", t2-t1)
print("QoI average relative L2 test (total):", er_test_loss)
print("QoI relative Bochner L2 test (total):", er_test_bochner)
er_test_bochner_vec = torch.linalg.norm(y_test - qoi_out, dim=0) \
                            / torch.linalg.norm(y_test, dim=0)
er_test_loss_vec = torch.mean(torch.abs(y_test - qoi_out)
                              / torch.abs(y_test), dim=0)
if not FLAG_ONE:
    print("QoI average relative test:", er_test_loss_vec)
    print("QoI relative Bochner test:", er_test_bochner_vec)

# Save test errors
test_errors = np.array([er_test_bochner, er_test_loss])
qoi_errors = torch.cat((er_test_bochner_vec[:,None], er_test_loss_vec[:,None]), dim=-1)
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
    from util import plt

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
    
    if not FLAG_ONE:
        for i in range(N_qoi):    
            plt.close()
            plt.semilogy(errors[...,2+2*i:3+2*i+1])
            plt.grid(True, which="both")
            plt.xlim(0, epochs)
            plt.xlabel(r'Epoch')
            plt.ylabel(r'Error')
            plt.legend(["Train QoI-" + str(idx_qoi[i]), "Val QoI-" + str(idx_qoi[i])])
            plt.savefig(plot_folder + "epochs_qoi" + str(idx_qoi[i]) + obj_suffix[:-3] + "pdf", format='pdf')
    
    # Begin five errors plots
    idx_worst = torch.argmax(errors_test).item()
    idx_Q3 = torch.argsort(errors_test)[errors_test.shape[0]*3//4].item()
    idx_median = torch.argsort(errors_test)[errors_test.shape[0]//2].item()
    idx_Q1 = torch.argsort(errors_test)[errors_test.shape[0]//4].item()
    idx_best = torch.argmin(errors_test).item()
    idxs = [idx_worst, idx_Q3, idx_median, idx_Q1, idx_best]
    np.save(savepath + 'idx_min_Q1_med_Q3_max' + obj_suffix_eval, np.array(idxs))
    names = ["worst", "3rd quartile", "median", "1st quartile", "best"]
    for i, idx in enumerate(idxs):
        plt.close()
        plt.plot(torch.linspace(0, 1, x_test_all.shape[-1]), x_test_all[idx, 0, :].squeeze())
        plt.grid(visible=True)
        plt.xlim([0,1])
        plt.ylim([1,5])
        plt.title('Velocity Input Profile ' + '(' + names[i] + ')', fontsize=16)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$v_1(x_1, 0; \xi)$')
        
        plt.savefig(plot_folder + "eval_" + names[i] + obj_suffix_eval[:-3] + "pdf", format='pdf')
    
    # Close open figure if running interactively
    plt.close()
