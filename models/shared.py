import torch
import torch.nn as nn
import torch.fft as fft


def compl_mul(input_tensor, weights):
    """
    Complex multiplication:
    (batch, in_channel, ...), (in_channel, out_channel, ...) -> (batch, out_channel, ...), where ``...'' represents the spatial part of the input.
    """
    return torch.einsum("bi...,io...->bo...", input_tensor, weights)


################################################################
#
# 1d helpers
#
################################################################
def resize_rfft(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft(ar, n=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N) tensor, must satisfy real conjugate symmetry (not checked)
        s: (int), desired irfft output dimension >= 1
    Output
        out: (..., s//2 + 1) tensor
    """
    N = ar.shape[-1]
    s = s//2 + 1 if s >=1 else s//2
    if s >= N: # zero pad or leave alone
        out = torch.zeros(list(ar.shape[:-1]) + [s - N], dtype=torch.cfloat, device=ar.device)
        out = torch.cat((ar[..., :N], out), dim=-1)
    elif s >= 1: # truncate
        out = ar[..., :s]
    else: # edge case
        raise ValueError("s must be greater than or equal to 1.")

    return out


def resize_fft(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.ifft(ar, n=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Reference: https://github.com/numpy/numpy/pull/7593
    Args
        ar: (..., N) tensor
        s: (int), desired ifft output dimension >= 1
    Output
        out: (..., s) tensor
    """
    N = ar.shape[-1]
    if s >= N: # zero pad or leave alone
        out = torch.zeros(list(ar.shape[:-1]) + [s - N], dtype=torch.cfloat, device=ar.device)
        out = torch.cat((ar[..., :N//2], out, ar[..., N//2:]), dim=-1)
    elif s >= 2: # truncate modes
        if s % 2: # odd
            out = torch.cat((ar[..., :s//2 + 1], ar[..., -s//2 + 1:]), dim=-1)
        else: # even
            out = torch.cat((ar[..., :s//2], ar[..., -s//2:]), dim=-1)
    else: # edge case s = 1
        if s < 1:
            raise ValueError("s must be greater than or equal to 1.")
        else:
            out = ar[..., 0:1]

    return out


def get_grid1d(shape, device):
    """
    Returns a discretization of the 1D identity function on [0,1]
    """
    size_x = shape[1]
    gridx = torch.linspace(0, 1, size_x)
    gridx = gridx.reshape(1, size_x, 1).repeat([shape[0], 1, 1])
    return gridx.to(device)


def projector1d(x, s=None):
    """
    Either truncate or zero pad the Fourier modes of x so that x has new resolution s (s is int)
    """
    if s is not None and s != x.shape[-1]:
        x = fft.irfft(resize_rfft(fft.rfft(x, norm="forward"), s), n=s, norm="forward")
        
    return x


################################################################
#
# 2d helpers
#
################################################################
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

    
def get_grid2d(shape, device):
    """
    Returns a discretization of the 2D identity function on [0,1]^2
    """
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


def projector2d(x, s=None):
    """
    Either truncate or zero pad the Fourier modes of x so that x has new resolution s (s is 2 tuple)
    """
    if s is not None and tuple(s) != tuple(x.shape[-2:]):
        x = fft.irfft2(resize_rfft2(fft.rfft2(x, norm="forward"), s), s=s, norm="forward")
        
    return x


################################################################
#
# 1d Fourier layers
#
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        """
        Fourier integral operator layer defined for functions over the torus. Maps functions to functions.
        """
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x, s=None):
        """
        Input shape (of x):     (batch, channels, nx_in, ny_in)
        s:                      (int): desired spatial resolution (s,) in output space
        """
        # Original resolution
        xsize = x.shape[-1]
        
        # Compute Fourier coeffcients (un-scaled)
        x = fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(x.shape[0], self.out_channels, xsize//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul(x[:, :, :self.modes1], self.weights1)

        # Return to physical space
        if s is None or s == xsize:
            x = fft.irfft(out_ft, n=xsize)
        else:
            x = fft.irfft(resize_rfft(out_ft, s), n=s, norm="forward") / xsize

        return x
    

class LinearFunctionals1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        """
        Fourier neural functionals (encoder) layer for functions over the torus. Maps functions to vectors.
        Inputs:    
            in_channels  (int): number of input functions
            out_channels (int): total number of linear functionals to extract
        """
        super(LinearFunctionals1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
    
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
    
        # Complex conjugation in L^2 inner product is absorbed into parametrization
        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1 + 1, dtype=torch.cfloat))

    def forward(self, x):
        """
        Input shape (of x):     (batch, in_channels, nx_in)
        Output shape:           (batch, out_channels)
        """
        # Compute Fourier coeffcients (scaled to approximate integration)
        x = fft.rfft(x, norm="forward")
        
        # Truncate input modes
        x = resize_rfft(x, 2*self.modes1)

        # Multiply relevant Fourier modes and take the real part
        x = compl_mul(x, self.weights).real

        # Integrate the conjugate product in physical space by summing Fourier coefficients
        x = 2*torch.sum(x, dim=-1) - x[..., 0]

        return x


class LinearDecoder1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, s):
        """
        Fourier neural decoder layer for functions over the torus. Maps vectors to functions.
        Inputs:    
            in_channels  (int): dimension of input vectors
            out_channels (int): total number of functions to extract
            s            (int): desired spatial resolution (nx,) of functions
        """
        super(LinearDecoder1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1 + 1, dtype=torch.cfloat))

    def forward(self, x):
        """
        Input shape (of x):     (batch, in_channels)
        Output shape:           (batch, out_channels, nx)
        """
        # Multiply relevant Fourier modes
        x = compl_mul(x.type(torch.cfloat), self.weights)
        
        # Zero pad modes
        x = resize_rfft(x, self.s)
        
        # Return to physical space
        return fft.irfft(x, n=self.s)
    

################################################################
#
# 2d Fourier layers
#
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier integral operator layer defined for functions over the torus. Maps functions to functions.
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
            compl_mul(x[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul(x[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if s is None or tuple(s) == tuple(xsize):
            x = fft.irfft2(out_ft, s=tuple(xsize))
        else:
            x = fft.irfft2(resize_rfft2(out_ft, s), s=s, norm="forward") / (xsize[-2] * xsize[-1])

        return x


class LinearFunctionals2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier neural functionals (encoder) layer for functions over the torus. Maps functions to vectors.
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
        x = compl_mul(x, self.weights).real

        # Integrate the conjugate product in physical space by summing Fourier coefficients
        x = 2*torch.sum(x[..., :self.modes1, :], dim=(-2, -1)) + \
            2*torch.sum(x[..., -self.modes1:, 1:], dim=(-2, -1)) - x[..., 0, 0]

        return x
    

class LinearDecoder2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s):
        """
        Fourier neural decoder layer for functions over the torus. Maps vectors to functions.
        Inputs:    
            in_channels  (int): dimension of input vectors
            out_channels (int): total number of functions to extract
            s (list or tuple, length 2): desired spatial resolution (nx,ny) of functions
        """
        super(LinearDecoder2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, 2*self.modes1, self.modes2 + 1, dtype=torch.cfloat))

    def forward(self, x):
        """
        Input shape (of x):     (batch, in_channels)
        Output shape:           (batch, out_channels, nx, ny)
        """
        # Multiply relevant Fourier modes
        x = compl_mul(x.type(torch.cfloat), self.weights)
        
        # Zero pad modes
        x = resize_rfft2(x, tuple(self.s))
        
        # Return to physical space
        return fft.irfft2(x, s=self.s)
