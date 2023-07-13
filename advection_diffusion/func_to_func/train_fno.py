import torch
import numpy as np
import os, sys; sys.path.append(os.path.join('../..'))
from importlib import import_module
from timeit import default_timer

from util import Adam
from util.utilities_module import LpLoss, LppLoss, count_params, validate, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)

from advection_diffusion.helpers import get_qoi, trapz2, process_velocity

################################################################
#
# user configuration
#
################################################################
# Process command line arguments
print(sys.argv)
save_prefix = sys.argv[1]   # e.g., 'modelsize/', 'scalability/', 'efficiency/'
data_suffix = sys.argv[2]   # e.g., 'nu_inf_ell_p05_torch/' or 'nu_1p5_ell_p25_torch/'
N_train = int(sys.argv[3])  # training sample size
d_str = sys.argv[4]         # KLE dimension of training inputs (d = 1, 2, 5, 10, 15, 20, or 1000)
FNM_model = sys.argv[5]     # model name: 'FNO2d', etc
FNM_layers = int(sys.argv[6])
FNM_modes = int(sys.argv[7])
FNM_width = int(sys.argv[8])
FNM_modes1d = int(sys.argv[9])
FNM_width1d = int(sys.argv[10])

# File I/O
data_prefix = '/groups/astuart/nnelsen/data/FNM/low_res/'
FLAG_save_model = True
FLAG_save_plots = True
SAVE_AFTER = 10

# Number of independent Monte Carlo loops over training trials
N_MC = 5

# Sample size  
N_test = 500        # number of validation samples to monitor during training

# Resolution subsampling
sub_in = 2**5       # input subsample factor (power of two) from s_max_in = 4097
sub_out = 2**0      # output subsample factor (power of two) from s_max_out = 33

# Training
batch_size = 20
epochs = 502
learning_rate = 1e-3
weight_decay = 1e-4
scheduler_step = 100
scheduler_gamma = 0.5

# Import FNO model
d_in = 1
modes1 = FNM_modes
modes2 = FNM_modes
width = FNM_width
modes1d = FNM_modes1d
width1d = FNM_width1d
n_layers = FNM_layers
my_model = getattr(import_module('models'), FNM_model)
if FNM_model == 'FNO2d':
    FLAG_2D = True
    modes_width_list = [modes1, modes2, width]
elif FNM_model == 'FNO1d2':
    FLAG_2D = False
    modes_width_list = [modes1d, width1d, modes1, modes2, width]
else:
    raise ValueError("Only models FNO2d and FNO1d2 are currently supported.")

################################################################
#
# load and process data
#
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)

# File IO
obj_suffix = '_' + FNM_model + '_L' + str(FNM_layers) + '_m' + str(FNM_modes) + '_w' + str(FNM_width) 
obj_suffix = '_n' + str(N_train) + '_d' + d_str + obj_suffix + '.npy'
data_folder = data_prefix + data_suffix + 'training/' + d_str + 'd/'
data_folder_test = data_prefix + data_suffix + 'testing/' + d_str + 'd/'
savepath = './results/' + save_prefix + data_suffix
os.makedirs(savepath, exist_ok=True)

# Load training data
y_train_all = torch.load(data_folder + 'velocity.pt')['velocity'][:,::sub_in].clone()
N_max = y_train_all.shape[0]
assert N_train <= N_max
x_train_all = y_train_all.unsqueeze(1)
y_train_all = torch.load(data_folder + 'state.pt')['state'][:,::sub_out,::sub_out].clone()
    
# Training objects
loss_f = LpLoss(size_average=False)

# Evaluation objects
loss_vec = LpLoss(size_average=False, reduction=False) # relative L^2 error (not summed)
loss_abs = LppLoss(size_average=False, reduction=True) # absolute squared L^2 error

# File IO evaluation
d_test_str = d_str
obj_suffix_eval = '_TESTd' + d_test_str + obj_suffix

# Load test data
x_test_all = torch.load(data_folder_test + 'velocity.pt')['velocity'].clone().unsqueeze(1)
x_eval = x_test_all[...,::sub_in]
N_test_max = x_eval.shape[0]
assert N_test <= N_test_max
y_test_all = torch.load(data_folder_test + 'state.pt')['state'].clone()

# Process evaluation
N_eval = N_test_max - N_test
x_eval = process_velocity(x_eval, FLAG_2D)
y_eval = y_test_all[...,::sub_out,::sub_out]
test_loader = DataLoader(TensorDataset(x_eval[-N_test:,...], y_eval[-N_test:,...]),
                         batch_size=batch_size, shuffle=False)
x_eval = x_eval[:-N_test,...]
y_eval = y_eval[:-N_test,...]
eval_loader = DataLoader(TensorDatasetID(x_eval, y_eval),
                         batch_size=batch_size, shuffle=False)

# Load QoI evaluation data
qoi_eval = torch.load(data_folder_test + 'qoi.pt')['qoi'][:-N_test,...]
N_qoi = qoi_eval.shape[-1]

# Initialize output files
idx_shuffle_all = []
model_dict = {}
optimizer_dict = {}
errors_all = []
test_errors_all = []
qoi_errors_all = []
errors_test_list= []

# Begin simple MC loops
for loop in range(N_MC):
    print('######### Beginning MC loop %d/%d' % (loop + 1, N_MC))

    # File IO
    keym = 'model' + str(loop)
    keyo = 'optimizer' + str(loop)
    
    # Shuffle
    dataset_shuffle_idx = torch.randperm(N_max)
    idx_shuffle_all.append(dataset_shuffle_idx.numpy())
    np.save(savepath + 'idx_shuffle_all' + obj_suffix, np.asarray(idx_shuffle_all))
    x_train = x_train_all[dataset_shuffle_idx, ...]
    y_train = y_train_all[dataset_shuffle_idx, ...]
    
    # Extract
    x_train = process_velocity(x_train[:N_train,...], FLAG_2D)
    y_train = y_train[:N_train,...]
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    
    ################################################################
    #
    # training
    #
    ################################################################
    s_outputspace = tuple(y_train.shape[-2:])   # same output shape as the output dataset
    
    model = my_model(*modes_width_list, s_outputspace, d_in=d_in, n_layers=n_layers).to(device)
    print(model)
    print("FNO parameter count:", count_params(model))
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=scheduler_step,
                                                gamma=scheduler_gamma)
        
    errors = torch.zeros((epochs, 2 + 2 + 2*N_qoi))
    errors_all.append(errors.numpy())
    
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
            if ep % SAVE_AFTER == 0:
                model_dict.update({keym: model.state_dict()})
                optimizer_dict.update({keyo: optimizer.state_dict()})
                torch.save(model_dict, savepath + 'model_dict' + obj_suffix[:-3] + 'pt')
                torch.save(optimizer_dict, savepath + 'optimizer_dict' + obj_suffix[:-3] + 'pt')
    
        t2 = default_timer()
    
        print("Epoch:", ep, "Epoch time:", t2-t1)
        print("^Train L2:", train_loss, "Test L2:", test_loss, "Train QoI:", train_qoi,
              "Test QoI:", test_qoi)
        errors_all[loop] = errors.numpy()
        np.save(savepath + 'train_errors_all' + obj_suffix, np.asarray(errors_all))
    
    # End training loop
    model_dict.update({keym: model.state_dict()})
    optimizer_dict.update({keyo: optimizer.state_dict()})
    torch.save(model_dict, savepath + 'model_dict' + obj_suffix[:-3] + 'pt')
    torch.save(optimizer_dict, savepath + 'optimizer_dict' + obj_suffix[:-3] + 'pt')
    print("Total time elapsed (min):", (default_timer()-t0)/60., "Total epochs trained:", epochs)
    
    ################################################################
    #
    # evaluation
    #
    ################################################################    
    # Evaluate
    model.eval()
    t1 = default_timer()
    er_test_loss = 0.0
    num = 0.0
    den = 0.0
    state_out = torch.zeros(y_eval.shape)
    qoi_out = torch.zeros((N_eval, N_qoi))
    errors_test = torch.zeros(y_eval.shape[0])
    with torch.no_grad():
        for x, y, idx_test in eval_loader:
            x, y = x.to(device), y.to(device)
    
            out = model(x)
    
            er_test_loss += loss_f(out, y).item()
            num += loss_abs.abs(out, y).item()
            den += loss_abs.abs(y, 0*y).item()
            
            errors_test[idx_test] = loss_vec(out, y).cpu()
            
            out = out.squeeze()
            qoi_out[idx_test, ...] = get_qoi(out).cpu()
            state_out[idx_test, ...] = out.cpu()
    
    # State error
    er_test_loss /= N_eval
    er_test_bochner = (num/den)**(0.5)
    
    # QoI errors
    er_test_qoi_bochner = validate(qoi_eval, qoi_out)
    er_test_qoi_loss = torch.mean(torch.linalg.norm(qoi_eval - qoi_out, dim=-1)
                                  / torch.linalg.norm(qoi_eval, dim=-1), dim=0).item()
    er_test_qoi_bochner_vec = torch.linalg.norm(qoi_eval - qoi_out, dim=0) \
                                / torch.linalg.norm(qoi_eval, dim=0)
    er_test_qoi_loss_vec = torch.mean(torch.abs(qoi_eval - qoi_out)
                                  / torch.abs(qoi_eval), dim=0)

    t2 = default_timer()
    print("Time to evaluate", N_eval, "samples (sec):", t2-t1)
    print("Average relative L2 test:", er_test_loss)
    print("Relative Bochner L2 test:", er_test_bochner)
    print("QoI average relative L2 test (total):", er_test_qoi_loss)
    print("QoI relative Bochner L2 test (total):", er_test_qoi_bochner)
    print("QoI average relative test:", er_test_qoi_loss_vec)
    print("QoI relative Bochner test:", er_test_qoi_bochner_vec)
    
    # Save test errors
    test_errors_all.append(np.asarray([er_test_qoi_bochner, er_test_qoi_loss,
                                       er_test_bochner, er_test_loss]))
    qoi_errors_all.append(torch.cat((er_test_qoi_bochner_vec[:,None], er_test_qoi_loss_vec[:,None]),
                                    dim=-1).numpy())
    errors_test_list.append(errors_test.numpy())
    np.savez(savepath + 'test_errors_all' + obj_suffix_eval[:-3] + 'npz',
             qoib_qoil_bochner_loss=np.asarray(test_errors_all),
             rel_test_error_list=np.asarray(errors_test_list),
             qoibl_vec=np.asarray(qoi_errors_all)
             )

print('######### End of all', N_MC, 'MC loops\n')

################################################################
#
# plotting last MC trial
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
    XX = torch.linspace(0, 1, y_eval.shape[-1])
    (YY, XX) = torch.meshgrid(XX, XX)
    for i, idx in enumerate(idxs):
        true_testsort = y_eval[idx,...].squeeze()
        plot_testsort = state_out[idx,...].squeeze()
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
