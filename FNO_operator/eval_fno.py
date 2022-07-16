import torch

from models import FNO2d
import os, sys; sys.path.append(os.path.join('..'))
from util.utilities_module import LpLoss, UnitGaussianNormalizer, count_params, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)

# from util.plot_suiteSIAM import Plotter
import matplotlib.pyplot as plt

from timeit import default_timer

################################################################
#
# user configuration
#
################################################################
# Process command line arguments
print(sys.argv)
save_prefix = sys.argv[1]   # e.g., robustness, scalability, efficiency
data_suffix = sys.argv[2]   # e.g., 'nu_inf_ell_p05_torch/' or 'nu_1p5_ell_p25_torch/'
N_train = int(sys.argv[3])  # training sample size
d_str = sys.argv[4]         # KLE dimension of training inputs
sigma = int(sys.argv[5])    # index between 0 and 8 that defines the noise standard deviation

# File I/O
data_prefix = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/validation/'      # local
# data_prefix = '/groups/astuart/nnelsen/data/raise/validation/'                            # HPC

# Resolution subsampling
sub_in = 2**6       # input subsample factor (power of two) from s_max_in = 4097
sub_out = 2**0      # output subsample factor (power of two) from s_max_out = 33

# FNO
modes1 = 12
modes2 = 12
width = 32

# TODO: loop to evaluate on all test d values

# Eval
batch_size = 20

################################################################
#
# load and process data
#
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File IO
obj_suffix = '_n' + str(N_train) + '_d' + d_str + '_s' + str(sigma) + '.pt'
data_folder = data_prefix + d_str + 'd_torch/'
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
torch.save({'shuffle_idx': dataset_shuffle_idx}, savepath + 'shuffle_idx' + obj_suffix)
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


x_train = torch.load(data_folder + 'kernel.pt')['kernel'][...,::sub_in,::sub_in]
y_train = torch.load(data_folder + 'conductivity.pt')['conductivity'][...,::sub_out,::sub_out]
mask = torch.load(data_folder + 'mask.pt')['mask'][::sub_out,::sub_out]
mask = mask.to(device)

x_test = x_train[-N_test:,...]
x_train = x_train[:N_train,...]

y_test = y_train[-N_test:,...]
y_train = y_train[:N_train,...]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

# Make the singleton channel dimension match the FNO2D model input shape requirement
x_train = torch.unsqueeze(x_train, 1)
x_test = torch.unsqueeze(x_test, 1)
x_test3 = x_normalizer.encode(torch.unsqueeze(x_test3, 1))

train_loader = DataLoader(TensorDatasetID(x_train, y_train), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDatasetID(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
#
# evaluation on all test sets with the same model
#
################################################################
s_outputspace = tuple(y_train.shape[-2:])   # same output shape as the output dataset

model = FNO2d(modes1, modes2, width, s_outputspace=s_outputspace).to(device)
model.load_state_dict(torch.load('model.pt'))
# model.load_state_dict(torch.load('model_last.pt'))
print(count_params(model))
model.eval()
loss_f = LpLoss(size_average=False)
loss_vec = LpLoss(size_average=False, reduction=False)

t1 = default_timer()
test_loss = 0.0
out_test = torch.zeros(y_test.shape)
errors_test = torch.zeros(y_test.shape[0])
with torch.no_grad():
    for x, y, idx_test in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

        test_loss += loss_f(out, y).item()
        
        errors_test[idx_test] = loss_vec(out,y).cpu()
        
        out_test[idx_test,...] = out.squeeze().cpu()

test_loss /= N_test
t2 = default_timer()
print(test_loss, t2-t1)

################################################################
#
# plotting
#
################################################################
plot_folder = "figures/"
os.makedirs(plot_folder, exist_ok=True)

    
# %% Worst, median, best case inputs (test)
plt.close("all")

idx_worst = torch.argmax(errors_test).item()
idx_median = torch.argsort(errors_test)[errors_test.shape[0]//2].item()
idx_best = torch.argmin(errors_test).item()

idxs = [idx_worst, idx_median, idx_best]
names = ["worst", "median", "best"]

for i in range(3):
    idx = idxs[i]
    true_testsort = y_test[idx,...].squeeze()
    true_testsort[~mask_plot] = float('nan')
    plot_testsort = out_test[idx,...].squeeze()
    er_testsort = torch.abs(plot_testsort - true_testsort).squeeze()
    
    plt.close()
    plt.figure(3, figsize=(9, 9))
    plt.subplot(2,2,1)
    plt.title('Test Output')
    plt.imshow(plot_testsort, origin='lower', interpolation='none')
    plt.box(False)
    plt.subplot(2,2,2)
    plt.title('Test Truth')
    plt.imshow(true_testsort, origin='lower', interpolation='none')
    plt.box(False)
    plt.subplot(2,2,3)
    plt.title('Test Input')
    plt.imshow(x_test[idx,...].squeeze(), origin='lower')
    plt.subplot(2,2,4)
    plt.title('Test PW Error')
    plt.imshow(er_testsort, origin='lower')
    plt.box(False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()

    # plt.savefig(plot_folder + "eval_test_" + names[i] + ".png", format='png', dpi=300, bbox_inches='tight')

# %% Save train or not
# plt.savefig(plot_folder + "eval_train" + str(pind_train) + ".png", format='png', dpi=300, bbox_inches='tight')

# %% Test point
plt.close("all")

pind_test = torch.randint(N_test, [1]).item()

true_test = y_test[pind_test,...].squeeze()
true_test[~mask_plot] = float('nan')
plot_test = out_test[pind_test,...].squeeze()
er_test = torch.abs(plot_test - true_test).squeeze()

plt.figure(11, figsize=(9, 9))
plt.subplot(2,2,1)
plt.title('Test Output')
plt.imshow(plot_test, origin='lower', interpolation='none')
plt.box(False)
plt.subplot(2,2,2)
plt.title('Test Truth')
plt.imshow(true_test, origin='lower', interpolation='none')
plt.box(False)
plt.subplot(2,2,3)
plt.title('Test Input')
plt.imshow(x_test[pind_test,...].squeeze(), origin='lower')
plt.subplot(2,2,4)
plt.title('Test PW Error')
plt.imshow(er_test, origin='lower')
plt.box(False)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.tight_layout()
plt.show()

# %% Save test or not
# plt.savefig(plot_folder + "eval_test" + str(pind_test) + ".png", format='png', dpi=300, bbox_inches='tight')

# %% Non-random phantoms of varying contrast
plt.close("all")

for i in range(3):
    true_test3 = y_test3[i,...].squeeze()
    true_test3[~mask_plot] = float('nan')
    plot_test3 = out3[i,...].squeeze()
    er_test3 = torch.abs(plot_test3 - true_test3).squeeze()
    
    plt.close()
    plt.figure(22, figsize=(9, 9))
    plt.subplot(2,2,1)
    plt.title('Test Output')
    plt.imshow(plot_test3, origin='lower', interpolation='none')
    plt.box(False)
    plt.subplot(2,2,2)
    plt.title('Test Truth')
    plt.imshow(true_test3, origin='lower', interpolation='none')
    plt.box(False)
    plt.subplot(2,2,3)
    plt.title('Test Input')
    plt.imshow(x_test3[i,...].squeeze(), origin='lower')
    plt.subplot(2,2,4)
    plt.title('Test PW Error')
    plt.imshow(er_test3, origin='lower')
    plt.box(False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()

    # plt.savefig(plot_folder + "eval_phantom_rhop7_" + str(i) + ".png", format='png', dpi=300, bbox_inches='tight')
    # plt.savefig(plot_folder + "eval_phantom" + str(i) + ".png", format='png', dpi=300, bbox_inches='tight')