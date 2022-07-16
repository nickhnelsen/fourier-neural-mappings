import torch

from models import FNO2d
import os, sys; sys.path.append(os.path.join('..'))
from util.Adam import Adam
from util.utilities_module import LpLoss, UnitGaussianNormalizer, count_params
from torch.utils.data import TensorDataset, DataLoader

from timeit import default_timer

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

save_prefix = 'robustness_TEST/'    # e.g., robustness, scalability, efficiency
data_suffix = 'nu_inf_ell_p25_torch/'
N_train = 1000
d_str = '1000'
sigma = 0                   # index between 0 and 8

# TODO: MC loop

# File paths
data_prefix = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/training/'      # local
# data_prefix = '/groups/astuart/nnelsen/data/raise/training/'                            # HPC
FLAG_save_model = True

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
epochs = 150
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
    torch.save({'errors': errors}, savepath + 'errors' + obj_suffix)

print("Total time elapsed (min):", (default_timer()-t0)/60., "Total epochs trained:", epochs)

################################################################
#
# evaluation
#
################################################################
data_prefix = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/validation/'      # local
# data_prefix = '/groups/astuart/nnelsen/data/raise/validation/'                            # HPC

# File IO
data_folder = data_prefix + d_str + 'd_torch/'

# Load
y_test = torch.load(data_folder + 'velocity.pt')['velocity'][:,::sub_in]
N_test_max, s_test = y_test.shape
x_test = torch.zeros(N_test_max, 2, s_test)
x_test[:, 0, :] = y_test
y_test = torch.load(data_folder + 'state.pt')['state'][:,::sub_out,::sub_out,-1] # final time state only
qoi_test = torch.load(data_folder + 'qoi.pt')['qoi']

# Process
x_test = x_test.unsqueeze(-1).repeat(1, 1, 1, s) # velocity is constant in y=x_2 direction

test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)



