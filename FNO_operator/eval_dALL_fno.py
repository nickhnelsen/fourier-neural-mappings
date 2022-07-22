import torch
import numpy as np
import os, sys; sys.path.append(os.path.join('..', 'util'))
from timeit import default_timer

from models import FNO2d
from utilities_module import LpLoss, LppLoss, count_params, validate, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)

################################################################
#
# user configuration
#
################################################################
# Process command line arguments
print(sys.argv)
save_prefix = sys.argv[1]   # e.g., 'robustness/', 'scalability/', 'efficiency/'
data_suffix = sys.argv[2]   # e.g., 'nu_inf_ell_p05_torch/' or 'nu_1p5_ell_p25_torch/'
N_train = int(sys.argv[3])  # training sample size used in model to load
d_str = sys.argv[4]         # KLE dimension of model to load (d = 1, 2, 5, 10, 15, 20, or 1000)
sigma = int(sys.argv[5])    # index between 0 and 8 that defines the noise standard deviation

# TODO: remove local paths once it runs on HPC without errors
# File I/O
data_prefix_eval = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/validation/'      # local
# data_prefix_eval = '/groups/astuart/nnelsen/data/raise/validation/'                            # HPC
FLAG_save_plots = True

# Resolution subsampling
sub_in = 2**6       # input subsample factor (power of two) from s_max_in = 4097
sub_out = 2**0      # output subsample factor (power of two) from s_max_out = 33

# FNO
modes1 = 12
modes2 = 12
width = 32

# Evaluation
batch_size = 20
s_outputspace = tuple((33, 33))
d_test_str_list = ['1', '2', '5', '10', '15', '20']
N_val = 2000

################################################################
#
# load and process data
#
################################################################
# File IO
obj_suffix = '_n' + str(N_train) + '_d' + d_str + '_s' + str(sigma) + '.npy'
savepath = './results/' + save_prefix + data_suffix

# Setup model and get model dictionary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)
model = FNO2d(modes1, modes2, width, s_outputspace).to(device)
print("FNO parameter count:", count_params(model))
model_dict = torch.load(savepath + 'model_dict' + obj_suffix[:-3] + 'pt')

# Evaluation objects
loss_f = LpLoss(size_average=False)
loss_vec = LpLoss(size_average=False, reduction=False) # relative L^2 error (not summed)
loss_abs = LppLoss(size_average=False, reduction=True) # absolute squared L^2 error

# Initialize output files
N_MC = len(model_dict)
N_d = len(d_test_str_list)
test_errors_all_dall = np.zeros((N_MC, N_d, 3))
errors_test_list_dall = np.zeros((N_MC, N_d, N_val))

# Double loop over MC and d_test
t0 = default_timer()
for idx_d, d_test_str in enumerate(d_test_str_list):
    data_folder_eval = data_prefix_eval + d_test_str + 'd_torch/'
    obj_suffix_eval = '_TESTd' + d_test_str + obj_suffix

    # Load evaluation
    y_eval = torch.load(data_folder_eval + 'velocity.pt')['velocity'][:,::sub_in]
    N_eval_max, s_eval = y_eval.shape
    x_eval = torch.zeros(N_eval_max, 2, s_eval)
    x_eval[:, 0, :] = y_eval
    y_eval = torch.load(data_folder_eval + 'state.pt')['state'][:,::sub_out,::sub_out,-1]
    qoi_eval = torch.load(data_folder_eval + 'qoi.pt')['qoi']

    # Process evaluation
    s_outputtest = y_eval.shape[-2:]
    idx_qoi = torch.div(torch.tensor(s_outputtest), 2, rounding_mode="floor") # center of grid
    x_eval = x_eval.unsqueeze(-1).repeat(1, 1, 1, s_eval) # velocity is constant in y=x_2 direction
    eval_loader = DataLoader(TensorDatasetID(x_eval, y_eval), batch_size=batch_size, shuffle=False)

    for loop in range(N_MC):    
        # Load model corresponding to each MC trial
        model.load_state_dict(model_dict['model' + str(loop)])
        model.eval()
        if tuple(s_outputspace) != tuple(s_outputtest):
            model.set_outputspace_resolution(tuple(s_outputtest))
    
        ################################################################
        #
        # evaluation on all test sets with the same model
        #
        ################################################################    
        # Evaluate
        t1 = default_timer()
        er_test_loss = 0.0
        num = 0.0
        den = 0.0
        state_out = torch.zeros(y_eval.shape)
        qoi_out = torch.zeros(N_eval_max)
        errors_test = torch.zeros(y_eval.shape[0])
        with torch.no_grad():
            for x, y, idx_test in eval_loader:
                x, y = x.to(device), y.to(device)
        
                out = model(x)
        
                er_test_loss += loss_f(out, y).item()
                num += loss_abs.abs(out, y).item()
                den += loss_abs.abs(y, 0*y).item()
                
                errors_test[idx_test] = loss_vec(out, y).cpu()
                
                out = out.squeeze().cpu()
                state_out[idx_test, ...] = out
                qoi_out[idx_test] = out[..., idx_qoi[-2], idx_qoi[-1]]
        
        er_test_loss /= N_eval_max
        er_test_bochner = (num/den)**(0.5)
        er_test_qoi = validate(qoi_eval, qoi_out)
        t2 = default_timer()
        pid = "d" + str(d_test_str) + "_MC" + str(loop) + " "
        print(pid + "Time to evaluate", N_eval_max, "samples (sec):", t2-t1)
        print(pid + "Average relative L2 test:", er_test_loss, "Relative Bochner L2 test:", er_test_bochner)
        print(pid + "Relative L2 QoI test error:", er_test_qoi)
        
        test_errors_all_dall[loop, idx_d, :] = np.asarray([er_test_qoi, er_test_bochner, er_test_loss])
        errors_test_list_dall[loop, idx_d, :] = errors_test.numpy()
        
        ################################################################
        #
        # plotting last MC trial
        #
        ################################################################
        if FLAG_save_plots and loop == N_MC - 1:
            from configure_plots import plt, mpl
            from mpl_toolkits.axes_grid1 import ImageGrid
        
            plt.rcParams['figure.figsize'] = [6.0, 4.0]
            plt.rcParams['font.size'] = 16
            plt.rcParams['figure.dpi'] = 250
            plt.rcParams['savefig.dpi'] = 250
            
            plot_folder = savepath + "figures/"
            os.makedirs(plot_folder, exist_ok=True)
            
            # Begin three errors plots
            idx_worst = torch.argmax(errors_test).item()
            idx_median = torch.argsort(errors_test)[errors_test.shape[0]//2].item()
            idx_best = torch.argmin(errors_test).item()
            idxs = [idx_worst, idx_median, idx_best]
            np.save(savepath + 'idx_min_med_max' + obj_suffix_eval, np.asarray(idxs))
            names = ["worst", "median", "best"]
            XX = torch.linspace(0, 1, y_eval.shape[-1])
            (YY, XX) = torch.meshgrid(XX, XX)
            for i, idx in enumerate(idxs):
                true_testsort = y_eval[idx,...].squeeze()
                plot_testsort = state_out[idx,...].squeeze()
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
                grid[0].set_title('FNO Output')
                grid[0].set_xlabel(r'$x_1$')
                grid[0].set_ylabel(r'$x_2$')
                
                ax01 = grid[1].contourf(XX, YY, true_testsort, vmin=0, vmax=15, cmap=mpl.cm.viridis)
                grid[1].set_title('True Final State')
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
                
                grid[0].plot(torch.linspace(0, 1, s_eval), x_eval[idx, 0, :, 0].squeeze())
                grid[0].grid(visible=True)
                grid[0].set_xlim(0,1)
                grid[0].set_ylim(1,5)
                grid[0].set_title('Velocity Input Profile ' + '(' + names[i] + ')')
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
                grid[0].set_title('Pointwise Absolute Error')
                grid[0].set_xlabel(r'$x_1$')
                grid[0].set_ylabel(r'$x_2$')
                cb11 = grid[0].cax.colorbar(ax11)
                grid[0].cax.toggle_label(True)
                cb11.set_label(r'Error')
                
                # Save min, median, max error plots (contour)
                plt.savefig(plot_folder + "eval_" + names[i] + obj_suffix_eval[:-3] + "pdf", format='pdf')
            
            # Close open figure if running interactively
            plt.close()

# Write errors to file
np.savez(savepath + 'test_errors_all_dALL' + obj_suffix[:-3] + 'npz',\
         qoi_bochner_loss_dALL=test_errors_all_dall,\
             rel_test_error_list_dALL=errors_test_list_dall)
print("Total time elapsed (min):", (default_timer()-t0)/60.)
