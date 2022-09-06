import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import SpectralConv2d, projector2d, get_grid2d


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for mapping functions to functions
    """
    def __init__(self, modes1, modes2, width,
                 s_outputspace=None,
                 width_final=128,
                 padding=8,
                 d_in=2, # TODO: adjust default to 1, check this does not break train scripts
                 d_out=1
                 ):
        """
        modes1, modes2  (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        s_outputspace   (list or tuple, length 2): desired spatial resolution (s,s) in output space
        width_final     (int): width of the final projection layer
        padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
        d_in            (int): number of input channels (NOT including grid input features)
        d_out           (int): one output channel (co-domain dimension of output space functions)
        """
        super(FNO2d, self).__init__()

        self.d_physical = 2
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width_final = width_final
        self.padding = padding
        self.d_in = d_in
        self.d_out = d_out 
        
        self.set_outputspace_resolution(s_outputspace)

        self.fc0 = nn.Linear(self.d_in + self.d_physical, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, self.width_final)
        self.fc2 = nn.Linear(self.width_final, self.d_out)

    def forward(self, x):
        """
        Input shape (of x):     (batch, channels=2, nx_in, ny_in)
        Output shape:           (batch, channels=1, nx_out, ny_out)
        
        The input resolution is determined by x.shape[-2:]
        The output resolution is determined by self.s_outputspace
        """
        # Lifting layer
        x_res = x.shape[-2:]
        x = x.permute(0, 2, 3, 1)
        x = torch.cat((x, get_grid2d(x.shape, x.device)), dim=-1)    # grid ``features''
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Map from input domain into the torus
        x = F.pad(x, [0, x_res[-1]//self.padding, 0, x_res[-2]//self.padding])

        # Fourier integral operator layers on the torus
        x = self.w0(x) + self.conv0(x)
        x = F.gelu(x)

        x = self.w1(x) + self.conv1(x)
        x = F.gelu(x)

        x = self.w2(x) + self.conv2(x)
        x = F.gelu(x)

        # Change resolution in function space consistent way
        x = self.w3(projector2d(x, s=self.s_outputspace)) + self.conv3(x, s=self.s_outputspace)

        # Map from the torus into the output domain
        if self.s_outputspace is not None:
            x = x[..., :-self.num_pad_outputspace[-2], :-self.num_pad_outputspace[-1]]
        else:
            x = x[..., :-x_res[-2]//self.padding, :-x_res[-1]//self.padding]
        
        # Final projection layer
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.permute(0, 3, 1, 2)

    def set_outputspace_resolution(self, s=None):
        """
        Helper to set desired output space resolution of the model at any time
        """
        if s is None:
            self.s_outputspace = None
            self.num_pad_outputspace = None
        else:
            self.s_outputspace = tuple([r + r//self.padding for r in list(s)])
            self.num_pad_outputspace = tuple([r//self.padding for r in list(s)])
