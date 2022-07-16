import torch

from models import FNO2d
import os, sys; sys.path.append(os.path.join('..'))
from util.Adam import Adam
from util.utilities_module import LpLoss, LppLoss, count_params, validate, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)
from timeit import default_timer
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

################################################################
#
# user configuration
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

save_prefix = 'robustness_TEST_eval/'    # e.g., robustness, scalability, efficiency
data_suffix = 'nu_inf_ell_p25_torch/'
N_train = 1000
d_str = '1000'
sigma = 0                   # index between 0 and 8

# TODO: MC loop

# File I/O
data_prefix = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/training/'      # local
# data_prefix = '/groups/astuart/nnelsen/data/raise/training/'                            # HPC
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
epochs = 15
learning_rate = 1e-3
weight_decay = 1e-4
scheduler_step = 100
scheduler_gamma = 0.5

################################################################
#
# load and process data
#
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)

# File IO
obj_suffix = '_n' + str(N_train) + '_d' + d_str + '_s' + str(sigma) + '.pt'
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

################################################################
#
# training
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
        torch.save(model.state_dict(), savepath + 'model' + obj_suffix)

    t2 = default_timer()

    print("Epoch:", ep, "Train L2:", train_loss, "Test L2:", test_loss, "Epoch time:", t2-t1)
    torch.save({'train_errors': errors}, savepath + 'train_errors' + obj_suffix)

print("Total time elapsed (min):", (default_timer()-t0)/60., "Total epochs trained:", epochs)

################################################################
#
# evaluation
#
################################################################
data_prefix = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/validation/'      # local
# data_prefix = '/groups/astuart/nnelsen/data/raise/validation/'                            # HPC

# File IO
if d_str == '1000':
    d_test_str = '2'    # test on d=2 case to show resolution-invariance
else:
    d_test_str = d_str
data_folder = data_prefix + d_test_str + 'd_torch/'
obj_suffix = '_TESTd' + d_test_str + obj_suffix

# Load
y_test = torch.load(data_folder + 'velocity.pt')['velocity'][:,::sub_in]
N_test_max, s_test = y_test.shape
x_test = torch.zeros(N_test_max, 2, s_test)
x_test[:, 0, :] = y_test
y_test = torch.load(data_folder + 'state.pt')['state'][:,::sub_out,::sub_out,-1] # final time state only
qoi_test = torch.load(data_folder + 'qoi.pt')['qoi']

# Process
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
idx_qoi = torch.tensor(s_outputspace)//2 + 1
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
test_errors = [er_test_qoi, er_test_bochner, er_test_loss]
torch.save({'qoi_bochner_loss': test_errors, 'test_list': errors_test},\
           savepath + 'test_errors' + obj_suffix)

# TODO: remove for public version of code
# Evaluate trained model on 2D parameter grid
x_test = torch.load(data_prefix + '2d_qoi_plot/' + 'velocity.pt')['velocity'][:,::sub_in]
N_grid, s_grid = x_test.shape
x_grid = torch.zeros(N_grid, 2, s_grid)
x_grid[:, 0, :] = x_test
x_grid = x_grid.unsqueeze(-1).repeat(1, 1, 1, s_grid) # velocity is constant in y=x_2 direction
grid_loader = DataLoader(TensorDatasetID(x_grid, 0*x_grid), batch_size=batch_size, shuffle=False)
qoi_grid = torch.zeros(x_grid.shape[0])
with torch.no_grad():
    for x, _, idx_grid in grid_loader:
        x = x.to(device)
        qoi_grid[idx_grid] = model(x).squeeze().cpu()[..., idx_qoi[-2], idx_qoi[-1]]
torch.save({'qoi_grid': qoi_grid}, savepath + 'qoi_grid' + obj_suffix)
grid = torch.load(data_prefix + '2d_qoi_plot/' + 'params.pt')['params']
grid = grid.reshape(s_outputspace[-2], s_outputspace[-1], -1)
plt.contourf(grid[...,-2], grid[...,-1],\
             qoi_grid.reshape(s_outputspace[-2], s_outputspace[-1]), cmap=cm.viridis)
plt.title(r'FNO $(N=%d, d_{\mathrm{tr}}=%d, \sigma=0)$' % (N_train, int(d_str)))
plt.xlabel(r'$\xi_1$')
plt.ylabel(r'$\xi_2$')
plt.colorbar(label=r'QoI')
if FLAG_save_plots:
    plt.savefig(savepath + 'qoi_grid' + obj_suffix[:-2] + 'pdf', format='pdf')