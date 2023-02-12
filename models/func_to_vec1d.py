import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import SpectralConv1d, LinearFunctionals1d, get_grid1d, _get_act, MLP


class FNF1d(nn.Module):
    """
    Fourier Neural Functionals for mapping functions to finite-dimensional vectors
    """
    def __init__(self,
                 modes1=16,
                 width=64,
                 width_final=128,
                 padding=8,
                 d_in=1,
                 d_out=1,
                 width_lfunc=None,
                 act='gelu'
                 ):
        """
        modes1          (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        width_final     (int): width of the final projection layer
        padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
        d_in            (int): number of input channels (NOT including grid input features)
        d_out           (int): finite number of desired outputs (number of functionals)
        width_lfunc     (int): number of intermediate linear functionals to extract in FNF layer
        act             (str): Activation function = tanh, relu, gelu, elu, or leakyrelu
        """
        super(FNF1d, self).__init__()

        self.d_physical = 1
        self.modes1 = modes1
        self.width = width
        self.width_final = width_final
        self.padding = padding
        self.d_in = d_in
        self.d_out = d_out 
        if width_lfunc is None:
            self.width_lfunc = self.width
        else:
            self.width_lfunc = width_lfunc
        self.act = _get_act(act)
        
        self.fc0 = nn.Linear(self.d_in + self.d_physical, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        
        self.lfunc0 = LinearFunctionals1d(self.width, self.width_lfunc, self.modes1)

        self.mlp0 = MLP(self.width_lfunc, self.width_final, self.d_out, act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, channels_in, nx_in)
        Output shape:           (batch, self.d_out)
        
        The input resolution is determined by x.shape[-1]
        """
        # Lifting layer
        x = x.permute(0, 2, 1)
        x = torch.cat((x, get_grid1d(x.shape, x.device)), dim=-1)    # grid ``features''
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Map from input domain into the torus
        x = F.pad(x, [0, x.shape[-1]//self.padding])

        # Fourier integral operator layers on the torus
        x = self.w0(x) + self.conv0(x)
        x = self.act(x)

        x = self.w1(x) + self.conv1(x)
        x = self.act(x)

        x = self.w2(x) + self.conv2(x)
        x = self.act(x)

        # Extract Fourier neural functionals on the torus
        x = self.lfunc0(x)
        
        # Final projection layer
        x = self.mlp0(x)

        return x