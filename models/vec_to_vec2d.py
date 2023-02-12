import torch.nn as nn

from .shared import SpectralConv2d, LinearDecoder2d, LinearFunctionals2d, _get_act, MLP


class FNN2d(nn.Module):
    """
    Fourier Neural Network function for mapping finite-dimensional vectors to vectors
    """
    def __init__(self, d_in, d_out,
                 s_latentspace=(64,64),
                 modes1=12,
                 modes2=12,
                 width=32,
                 width_initial=128,
                 width_final=128,
                 width_ldec=None,
                 width_lfunc=None,
                 act='gelu'
                 ):
        """
        d_in            (int): number of input channels (dimension of input vectors)
        d_out           (int): finite number of desired outputs (dimension of output vectors)
        s_latentspace   (list or tuple, length 2): desired spatial resolution (s,s) in latent space
        modes1, modes2  (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        width_initial   (int): width of the initial processing layer
        width_final     (int): width of the final projection layer
        width_ldec      (int): input channel width for FND layer
        width_lfunc     (int): number of intermediate linear functionals to extract in FNF layer
        act             (str): Activation function = tanh, relu, gelu, elu, or leakyrelu
        """
        super(FNN2d, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.s_latentspace = s_latentspace
        self.modes1 = modes1
        self.modes2 = modes2
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
        self.act = _get_act(act)
        
        self.mlp0 = MLP(self.d_in, self.width_initial, self.width_ldec, act)
        
        self.ldec0 = LinearDecoder2d(self.width_ldec, self.width, self.modes1, self.modes2)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        
        self.lfunc0 = LinearFunctionals2d(self.width, self.width_lfunc, self.modes1, self.modes2)
        
        self.mlp1 = MLP(self.width_lfunc, self.width_final, self.d_out, act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, self.d_in)
        Output shape:           (batch, self.d_out)
        """
        # Nonlinear processing layer
        x = self.mlp0(x)
        
        # Decode into functions on the torus
        x = self.ldec0(x, self.s_latentspace)

        # Fourier integral operator layers on the torus
        x = self.w0(x) + self.conv0(x)
        x = self.act(x)

        x = self.w1(x) + self.conv1(x)
        x = self.act(x)

        # Extract Fourier neural functionals on the torus
        x = self.lfunc0(x)
        
        # Final projection layer
        x = self.mlp1(x)

        return x
    
    def set_latentspace_resolution(self, s):
        """
        Helper to set desired output space resolution of the model at any time
        """
        self.s_latentspace = s
