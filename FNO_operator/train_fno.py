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
# data_suffix = sys.argv[1]   # e.g., 'nu_inf_ell_p05_torch/2d/' or 'nu_1p5_ell_p25_torch/1000d/'
# N_train = int(sys.argv[2])  # number of solves (training sample size)
# save_suffix = sys.argv[3]   # e.g., robustness, scalability, efficiency
data_suffix = 'nu_inf_ell_p25_torch/2d/'
save_suffix = '_TEST_bothVel/'
save_prefix = 'results/'        # e.g., robustness, scalability, efficiency
N_train = 1000
sigma = 0 # TODO: noise stdev

# File I/O
data_prefix = '/media/nnelsen/SharedNHN/documents/datasets/Sandia/raise/training/'      # local
# data_prefix = '/groups/astuart/nnelsen/data/raise/training/'                            # HPC
FLAG_save_model = True
FLAG_reduce = False

# Sample size  
N_test = 100

# Resolution subsampling
sub_in = 2**6       # input subsample factor (power of two) from s_max_in = 4097
sub_out = 2**0      # output subsample factor (power of two) from s_max_out = 33

# FNO
modes1 = 12
modes2 = 12
width = 32

# Training
batch_size = 20
epochs = 100
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
data_folder = data_prefix + data_suffix
savepath = './' + save_prefix + data_suffix[:-1] + save_suffix
os.makedirs(savepath, exist_ok=True)

# TODO: monitor qoi during training?
# TODO: add noise flag to outputs
y_train = torch.load(data_folder + 'velocity.pt')['velocity'][:,::sub_in]
N_max, s = y_train.shape
assert max(N_train, N_test) <= N_max
x_train = torch.zeros(N_max, 2, s)
x_train[:, 0, :] = y_train
y_train = torch.load(data_folder + 'state.pt')['state'][:,::sub_out,::sub_out,-1] # final time state only
# qoi_test = torch.load(data_folder + 'qoi.pt')['qoi']

# TODO: torch.permute to get random training indices like in TORCH_RFM_UQ
x_test = x_train[-N_test:,...].unsqueeze(-1).repeat(1, 1, 1, s) # velocity is constant in y=x_2 direction
x_train = x_train[:N_train,...].unsqueeze(-1).repeat(1, 1, 1, s) # velocity is constant in y=x_2 direction

y_test = y_train[-N_test:,...]
y_train = y_train[:N_train,...]
# qoi_test = qoi_test[-N_test:,...]

# TODO: decide if normalizer is needed on inputs and/or outputs
# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
#
# training and evaluation
#
################################################################
s_outputspace = tuple(y_train.shape[-2:])   # same output shape as the output dataset

model = FNO2d(modes1, modes2, width, s_outputspace).to(device)
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
if FLAG_reduce:
    scheduler_val = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_gamma)

loss_f = LpLoss(size_average=False)

errors = torch.zeros((epochs, 2))

# TODO: decide if early stopping is needed
# lowest_test = 10.0 # initialize a test loss threshold
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
    if FLAG_reduce:
        scheduler_val.step(test_loss)

    errors[ep,0] = train_loss
    errors[ep,1] = test_loss

    if FLAG_save_model:
        torch.save(model.state_dict(), savepath + 'model_N' + str(N_train) + '.pt')
        # if test_loss < lowest_test:
        #     torch.save(model.state_dict(), 'model.pt')
        #     lowest_test = test_loss

    t2 = default_timer()

    print("Epoch:", ep, "Train L2:", train_loss, "Test L2:", test_loss, "Epoch time:", t2-t1)
    torch.save({'errors': errors}, savepath + 'errors_N' + str(N_train) + '.pt')

print("Total time elapsed (min):", (default_timer()-t0)/60., "Total epochs trained:", epochs)