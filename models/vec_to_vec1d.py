import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import SpectralConv1d, LinearDecoder1d, LinearFunctionals1d


class FNN1d(nn.Module):
    """
    Fourier Neural Network function for mapping finite-dimensional vectors to vectors
    """
    def __init__(self, d_in, d_out,
                 s_latentspace=1024,
                 modes1=16,
                 width=64,
                 width_initial=128,
                 width_final=128,
                 width_ldec=None,
                 width_lfunc=None
                 ):
        """
        d_in            (int): number of input channels (dimension of input vectors)
        d_out           (int): finite number of desired outputs (dimension of output vectors)
        s_latentspace   (int): desired spatial resolution (s,) in latent space
        modes1          (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        width_initial   (int): width of the initial processing layer
        width_final     (int): width of the final projection layer
        width_ldec      (int): input channel width for FND layer
        width_lfunc     (int): number of intermediate linear functionals to extract in FNF layer
        """
        super(FNN1d, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.s_latentspace = s_latentspace
        self.modes1 = modes1
        self.width = width
        self.width_initial = width_initial
        self.width_final = width_final
        if width_ldec is None:
            self.width_ldec = self.width
        else:
            self.width_ldec = width_ldec
        if width_lfunc is None:
            self.width_lfunc = self.width
        else:
            self.width_lfunc = width_lfunc
        
        self.fc0 = nn.Linear(self.d_in, self.width_initial)
        self.fc1 = nn.Linear(self.width_initial, self.width_ldec)
        
        self.ldec0 = LinearDecoder1d(self.width_ldec, self.width, self.modes1)
        
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        
        self.lfunc0 = LinearFunctionals1d(self.width, self.width_lfunc, self.modes1)
        
        self.fc2 = nn.Linear(self.width_lfunc, self.width_final)
        self.fc3 = nn.Linear(self.width_final, self.d_out)

    def forward(self, x):
        """
        Input shape (of x):     (batch, self.d_in)
        Output shape:           (batch, self.d_out)
        """
        # Nonlinear processing layer
        x = self.fc0(x)
        x = F.gelu(x)
        x = self.fc1(x)
        
        # Decode into functions on the torus
        x = self.ldec0(x, self.s_latentspace)

        # Fourier integral operator layers on the torus
        x = self.w0(x) + self.conv0(x)
        x = F.gelu(x)

        x = self.w1(x) + self.conv1(x)
        x = F.gelu(x)

        # Extract Fourier neural functionals on the torus
        x = self.lfunc0(x)
        
        # Final projection layer
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)

        return x
    
    def set_latentspace_resolution(self, s):
        """
        Helper to set desired output space resolution of the model at any time
        """
        self.s_latentspace = s
