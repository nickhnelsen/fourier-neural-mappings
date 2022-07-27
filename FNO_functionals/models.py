import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

import os, sys; sys.path.append(os.path.join('..', 'util'))
from utilities_module import resize_rfft, resize_fft

def resize_rfft2(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft2(ar, s=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (n, c, N_1, N_2) tensor, must satisfy real conjugate symmetry (not checked)
        s: (2) tuple, s=(s_1, s_2) desired irfft2 output dimension (s_i >=1)
    Output
        out: (n, c, s1, s_2//2 + 1) tensor
    """
    s1, s2 = s
    out = resize_rfft(ar, s2) # last axis (rfft)
    return resize_fft(out.permute(0, 1, 3, 2), s1).permute(0, 1, 3, 2) # second to last axis (fft)

class LinearFunctionals2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier neural functionals layer for functions over the 2D torus
        Inputs:    
            in_channels  (int): number of input functions
            out_channels (int): total number of linear functionals to extract
        """
        super(LinearFunctionals2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
    
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2
    
        # Complex conjugation in L^2 inner product is absorbed into parametrization
        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, 2*self.modes1, self.modes2 + 1, dtype=torch.cfloat))
    
    def compl_mul2d(self, input_tensor, weights):
        """
        Complex multiplication:
        (batch, in_channel, nx, ny), (in_channel, out_channel, nx, ny) -> (batch, out_channel, nx, ny)
        """
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x):
        """
        Input shape (of x):     (batch, in_channels, nx_in, ny_in)
        Output shape:           (batch, out_channels)
        """
        # Compute Fourier coeffcients (scaled to approximate integration)
        x = fft.rfft2(x, norm="forward")
        
        # Truncate input modes
        x = resize_rfft2(x, (2*self.modes1, 2*self.modes2))

        # Multiply relevant Fourier modes and take the real part
        x = self.compl_mul2d(x, self.weights).real

        # Integrate the conjugate product in physical space by summing Fourier coefficients
        x = 2*torch.sum(x[..., :self.modes1, :], dim=(-2, -1)) + \
            2*torch.sum(x[..., -self.modes1:, 1:], dim=(-2, -1)) - x[..., 0:1, 0:1]

        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier integral operator layer defined for functions over the torus
        """
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input_tensor, weights):
        """
        Complex multiplication:
        (batch, in_channel, nx, ny), (in_channel, out_channel, nx, ny) -> (batch, out_channel, nx, ny)
        """
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x, s=None):
        """
        Input shape (of x):     (batch, channels, nx_in, ny_in)
        s:                      (list or tuple, length 2): desired spatial resolution (s,s) in output space
        """
        # Original resolution
        xsize = x.shape[-2:]
        
        # Compute Fourier coeffcients (un-scaled)
        x = fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(x.shape[0], self.out_channels, xsize[-2], xsize[-1]//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if s is None or tuple(s) == tuple(xsize):
            x = fft.irfft2(out_ft, s=tuple(xsize))
        else:
            x = fft.irfft2(resize_rfft2(out_ft, s), s=s, norm="forward") / (xsize[-2] * xsize[-1])

        return x

class FNF2d(nn.Module):
    def __init__(self, modes1, modes2, width,
                 width_final=128,
                 padding=8,
                 d_in=4,
                 d_out=1,
                 width_lfunc=None
                 ):
        """
        modes1, modes2  (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        width_final     (int): width of the final projection layer
        padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
        d_in            (int): number of input channels (here 2 velocity inputs + 2 space variables)
        d_out           (int): finite number of desired outputs (number of functionals)
        width_lfunc     (int): number of intermediate linear functionals to extract in FNF layer
        """
        super(FNF2d, self).__init__()

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
        
        self.fc0 = nn.Linear(self.d_in, self.width)

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
        x = torch.cat((x, self.get_grid(x.shape, x.device)), dim=-1)    # grid ``features''
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
    
    def get_grid(self, shape, device):
        """
        Returns a discretization of the 2D identity function on [0,1]^2
        """
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
