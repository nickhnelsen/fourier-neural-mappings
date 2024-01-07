import torch
import numpy as np
import os, sys; sys.path.append(os.path.join('../..'))
from importlib import import_module
from timeit import default_timer

from util import Adam
from util.utilities_module import LpLoss, count_params, validate, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)

from advection_diffusion.helpers import process_velocity

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
FNM_model = sys.argv[5]     # model name: 'FNF2d' or 'FNF2d'
FNM_layers = int(sys.argv[6])
FNM_modes = int(sys.argv[7])
FNM_width = int(sys.argv[8])
FNM_modes1d = int(sys.argv[9])
FNM_width1d = int(sys.argv[10])
idx_qoi = [int(x) for x in sys.argv[11]]    # QoI indices (0 through 5) to learn, e.g.: "012345"

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
sub_in1d = 2**0

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
if FNM_model == 'FNF2d':
    FLAG_2D = True
    modes_width_list = [modes1, modes2, width]
elif FNM_model == 'FNF1d':
    FLAG_2D = False
    modes_width_list = [modes1d, width1d]
    sub_in = sub_in1d
else:
    raise ValueError("Only models FNF2d and FNF1d are currently supported.")

################################################################
#
# load and process data
#
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)

# File IO
obj_suffix = ''.join(str(x) for x in idx_qoi)
obj_suffix = '_' + FNM_model + '_qoi' + obj_suffix + '_L' + str(FNM_layers) + '_m' + str(FNM_modes)\
    + '_w' + str(FNM_width) + '_md' + str(FNM_modes1d) + '_wd' + str(FNM_width1d)
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
y_train_all = torch.load(data_folder + 'qoi.pt')['qoi'][...,idx_qoi].clone()
    
# Training objects
loss_f = LpLoss(size_average=False)

# Evaluation objects
loss_vec = LpLoss(size_average=False, reduction=False) # relative L^2 error (not summed)

# File IO evaluation
d_test_str = d_str
obj_suffix_eval = '_TESTd' + d_test_str + obj_suffix

# Load test data
x_test_all = torch.load(data_folder_test + 'velocity.pt')['velocity'].clone().unsqueeze(1)
x_eval = x_test_all[...,::sub_in]
N_test_max = x_eval.shape[0]
assert N_test <= N_test_max
y_test_all = torch.load(data_folder_test + 'qoi.pt')['qoi'][...,idx_qoi].clone()

# Process evaluation
N_qoi = len(idx_qoi)
N_eval = N_test_max - N_test
x_eval = process_velocity(x_eval, FLAG_2D)
y_eval = y_test_all
test_loader = DataLoader(TensorDataset(x_eval[-N_test:,...], y_eval[-N_test:,...]),
                         batch_size=batch_size, shuffle=False)
x_eval = x_eval[:-N_test,...]
y_eval = y_eval[:-N_test,...]
eval_loader = DataLoader(TensorDatasetID(x_eval, y_eval),
                         batch_size=batch_size, shuffle=False)

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
    model = my_model(*modes_width_list, d_in=d_in, d_out=N_qoi, n_layers=n_layers).to(device)
    print(model)
    print("FNF parameter count:", count_params(model))
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=scheduler_step,
                                                gamma=scheduler_gamma)
    
    FLAG_ONE = N_qoi==1
    errors = torch.zeros((epochs, 2 + 2*N_qoi*(1 - FLAG_ONE)))
    errors_all.append(errors.numpy())
    
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
            if ep % SAVE_AFTER == 0:
                model_dict.update({keym: model.state_dict()})
                optimizer_dict.update({keyo: optimizer.state_dict()})
                torch.save(model_dict, savepath + 'model_dict' + obj_suffix[:-3] + 'pt')
                torch.save(optimizer_dict, savepath + 'optimizer_dict' + obj_suffix[:-3] + 'pt')
    
        t2 = default_timer()
    
        print("Epoch:", ep, "Train L2:", train_loss, "Test L2:", test_loss, "Epoch time:", t2-t1)
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
    qoi_out = torch.zeros((N_eval, N_qoi))
    errors_test = torch.zeros(y_eval.shape[0])
    with torch.no_grad():
        for x, y, idx_test in eval_loader:
            x, y = x.to(device), y.to(device)
    
            out = model(x)
    
            er_test_loss += loss_f(out, y).item()
            
            errors_test[idx_test] = loss_vec(out, y).cpu()
            
            qoi_out[idx_test, ...] = out.cpu()
    
    # QoI errors
    er_test_bochner = validate(y_eval, qoi_out)
    er_test_loss /= N_eval
    
    t2 = default_timer()
    print("Time to evaluate", N_eval, "samples (sec):", t2-t1)
    print("QoI average relative L2 test (total):", er_test_loss)
    print("QoI relative Bochner L2 test (total):", er_test_bochner)
    er_test_bochner_vec = torch.linalg.norm(y_eval - qoi_out, dim=0) \
                                / torch.linalg.norm(y_eval, dim=0)
    er_test_loss_vec = torch.mean(torch.abs(y_eval - qoi_out)
                                  / torch.abs(y_eval), dim=0)
    if not FLAG_ONE:
        print("QoI average relative test:", er_test_loss_vec)
        print("QoI relative Bochner test:", er_test_bochner_vec)
    
    # Save test errors
    test_errors_all.append(np.asarray([er_test_bochner, er_test_loss]))
    qoi_errors_all.append(torch.cat((er_test_bochner_vec[:,None], er_test_loss_vec[:,None]),
                                    dim=-1).numpy())
    errors_test_list.append(errors_test.numpy())
    np.savez(savepath + 'test_errors_all' + obj_suffix_eval[:-3] + 'npz',
             bochner_loss=np.asarray(test_errors_all),
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
