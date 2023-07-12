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
                 act='gelu',
                 n_layers=4
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
        n_layers        (int): Number of Fourier Layers, by default 4
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
        self.n_layers = n_layers
        if self.n_layers is None:
            self.n_layers = 4
        
        self.fc0 = nn.Linear(self.d_in + self.d_physical, self.width)
        
        self.speconvs = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes1)
                for _ in range(self.n_layers - 1)]
            )

        self.ws = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1)
                for _ in range(self.n_layers - 1)]
            )
        
        self.lfunc0 = LinearFunctionals1d(self.width, self.width_lfunc, self.modes1)
        self.mlpfunc0 = MLP(self.width, self.width_final, self.width_lfunc, act)

        # Expand the hidden dim by 2 because the input is also twice as large
        self.mlp0 = MLP(2*self.width_lfunc, 2*self.width_final, self.d_out, act)

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
        for speconv, w in zip(self.speconvs, self.ws):
            x = w(x) + speconv(x)
            x = self.act(x)

        # Extract Fourier neural functionals on the torus
        x_temp = self.lfunc0(x)
        
        # Retain the truncated modes (use all modes)
        x = x.permute(0, 2, 1)
        x = self.mlpfunc0(x)
        x = x.permute(0, 2, 1)
        x = torch.trapz(x, dx=1./x.shape[-1])
        
        # Combine nonlocal and local features
        x = torch.cat((x_temp, x), dim=1)
        
        # Final projection layer
        x = self.mlp0(x)

        return x