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
# File I/O
#data_folder = '/media/nnelsen/SharedNHN/documents/datasets/eit/'    # local
data_folder = '/groups/astuart/nnelsen/data/eit/'                    # HPC
FLAG_save_model = True
FLAG_reduce = False

# Sample size
N_train = 9500      # N_train_max = 10000
N_test = 500

# Resolution subsampling
sub_in = 2**1       # input subsample factor (power of two) from s_max_out = 512
sub_out = 2**0      # output subsample factor (power of two) from s_max_out = 256

# FNO
modes1 = 12
modes2 = 12
width = 32

# Training
batch_size = 20
epochs = 100*0 + 3000*0 + 500
learning_rate = 1e-3
weight_decay = 1e-4
scheduler_step = 25*0 + 375*0 + 100
scheduler_gamma = 0.5

################################################################
#
# load and process data
#
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)

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

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True)

################################################################
#
# training and evaluation
#
################################################################
s_outputspace = tuple(y_train.shape[-2:])   # same output shape as the output dataset

model = FNO2d(modes1, modes2, width, s_outputspace=s_outputspace).to(device)
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
if FLAG_reduce:
    scheduler_val = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_gamma)

loss_f = LpLoss(size_average=False)

errors = torch.zeros((epochs,2))

lowest_test = 10.0 # initialize a test loss threshold
for ep in range(epochs):
    t1 = default_timer()

    train_loss = 0.0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

        loss = loss_f(out, y)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

            test_loss += loss_f(out, y).item()

    train_loss /= N_train
    test_loss /= N_test
    
    scheduler.step()
    if FLAG_reduce:
        scheduler_val.step(test_loss)

    errors[ep,0] = train_loss
    errors[ep,1] = test_loss

    if FLAG_save_model:
        torch.save(model.state_dict(), 'model_last.pt')
        if test_loss < lowest_test:
            torch.save(model.state_dict(), 'model.pt')
            lowest_test = test_loss

    t2 = default_timer()

    print(ep, train_loss, test_loss, t2-t1)
    torch.save({'errors': errors}, 'errors.pt')
