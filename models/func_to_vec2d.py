import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import SpectralConv2d, LinearFunctionals2d, get_grid2d


class FNF2d(nn.Module):
    """
    Fourier Neural Functionals for mapping functions to finite-dimensional vectors
    """
    def __init__(self, modes1, modes2, width,
                 width_final=128,
                 padding=8,
                 d_in=2, # TODO: adjust default to 1, check this does not break train scripts
                 d_out=1,
                 width_lfunc=None
                 ):
        """
        modes1, modes2  (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        width_final     (int): width of the final projection layer
        padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
        d_in            (int): number of input channels (NOT including grid input features)
        d_out           (int): finite number of desired outputs (number of functionals)
        width_lfunc     (int): number of intermediate linear functionals to extract in FNF layer
        """
        super(FNF2d, self).__init__()

        self.d_physical = 2
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width_final = width_final
        self.padding = padding
        self.d_in = d_in
        self.d_out = d_out 
        if width_lfunc is None:
            self.width_lfunc = self.width
        else:
            self.width_lfunc = width_lfunc
        
        self.fc0 = nn.Linear(self.d_in + self.d_physical, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        
        self.lfunc0 = LinearFunctionals2d(self.width, self.width_lfunc, self.modes1, self.modes2)

        self.fc1 = nn.Linear(self.width_lfunc, self.width_final)
        self.fc2 = nn.Linear(self.width_final, self.d_out)

    def forward(self, x):
        """
        Input shape (of x):     (batch, channels=2, nx_in, ny_in)
        Output shape:           (batch, self.d_out)
        
        The input resolution is determined by x.shape[-2:]
        """
        # Lifting layer
        x = x.permute(0, 2, 3, 1)
        x = torch.cat((x, get_grid2d(x.shape, x.device)), dim=-1)    # grid ``features''
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Map from input domain into the torus
        x = F.pad(x, [0, x.shape[-1]//self.padding, 0, x.shape[-2]//self.padding])

        # Fourier integral operator layers on the torus
        x = self.w0(x) + self.conv0(x)
        x = F.gelu(x)

        x = self.w1(x) + self.conv1(x)
        x = F.gelu(x)

        x = self.w2(x) + self.conv2(x)
        x = F.gelu(x)

        # Extract Fourier neural functionals on the torus
        x = self.lfunc0(x)
        
        # Final projection layer
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x