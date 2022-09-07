import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import SpectralConv1d, LinearDecoder1d


class FND1d(nn.Module):
    """
    Fourier Neural Decoder for mapping finite-dimensional vectors to functions
    """
    def __init__(self, d_in, s_outputspace,
                 modes1=16,
                 width=64,
                 width_initial=128,
                 width_final=128,
                 padding=8,
                 d_out=1,
                 width_ldec=None
                 ):
        """
        d_in            (int): number of input channels (dimension of input vectors)
        s_outputspace   (int): desired spatial resolution (s,) in output space
        modes1          (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        width_initial   (int): width of the initial processing layer
        width_final     (int): width of the final projection layer
        padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
        d_out           (int): finite number of desired outputs (number of functions)
        width_ldec      (int): input channel width for FND layer
        """
        super(FND1d, self).__init__()

        self.d_in = d_in
        self.modes1 = modes1
        self.width = width
        self.width_initial = width_initial
        self.width_final = width_final
        self.padding = padding
        self.d_out = d_out 
        if width_ldec is None:
            self.width_ldec = self.width
        else:
            self.width_ldec = width_ldec
            
        self.set_outputspace_resolution(s_outputspace)
        
        self.fc0 = nn.Linear(self.d_in, self.width_initial)
        self.fc1 = nn.Linear(self.width_initial, self.width_ldec)
        
        self.ldec0 = LinearDecoder1d(self.width_ldec, self.width, self.modes1)
        
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        
        self.fc2 = nn.Linear(self.width, self.width_final)
        self.fc3 = nn.Linear(self.width_final, self.d_out)

    def forward(self, x):
        """
        Input shape (of x):     (batch, self.d_in)
        Output shape:           (batch, self.d_out, nx_out)
        
        The output resolution is determined by self.s_outputspace
        """
        # Nonlinear processing layer
        x = self.fc0(x)
        x = F.gelu(x)
        x = self.fc1(x)
        
        # Decode into functions on the torus
        x = self.ldec0(x, self.s_outputspace)

        # Fourier integral operator layers on the torus
        x = self.w0(x) + self.conv0(x)
        x = F.gelu(x)

        x = self.w1(x) + self.conv1(x)
        x = F.gelu(x)

        x = self.w2(x) + self.conv2(x)
           
        # Map from the torus into the output domain
        x = x[..., :-self.num_pad_outputspace]
 
        # Final projection layer
        x = x.permute(0, 2, 1)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
           
        return x.permute(0, 2, 1)
    
    def set_outputspace_resolution(self, s):
        """
        Helper to set desired output space resolution of the model at any time
        """
        self.s_outputspace = s + s//self.padding
        self.num_pad_outputspace = s//self.padding
