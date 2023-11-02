import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import SpectralConv1d, projector1d, get_grid1d, _get_act, MLP


class FNO1d(nn.Module):
    """
    Fourier Neural Operator for mapping functions to functions
    """
    def __init__(self,
                 modes1=16,
                 width=64,
                 s_outputspace=None,
                 width_final=128,
                 padding=8,
                 d_in=1,
                 d_out=1,
                 act='gelu',
                 n_layers=4
                 ):
        """
        modes1          (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        s_outputspace   (int): desired spatial resolution (s,) in output space
        width_final     (int): width of the final projection layer
        padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
        d_in            (int): number of input channels (NOT including grid input features)
        d_out           (int): number of output channels (co-domain dimension of output space functions)
        act             (str): Activation function = tanh, relu, gelu, elu, or leakyrelu
        n_layers        (int): Number of Fourier Layers, by default 4
        """
        super(FNO1d, self).__init__()

        self.d_physical = 1
        self.modes1 = modes1
        self.width = width
        self.width_final = width_final
        self.padding = padding
        self.d_in = d_in
        self.d_out = d_out 
        self.act = _get_act(act)
        self.n_layers = n_layers
        if self.n_layers is None:
            self.n_layers = 4
        
        self.set_outputspace_resolution(s_outputspace)

        self.fc0 = nn.Linear(self.d_in + self.d_physical, self.width)
        
        self.speconvs = nn.ModuleList([SpectralConv1d(self.width, self.width, self.modes1)
                                       for _ in range(self.n_layers)])

        self.ws = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1)
                for _ in range(self.n_layers)]
            )

        self.mlp0 = MLP(self.width, self.width_final, self.d_out, act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, channels_in, nx_in)
        Output shape:           (batch, channels_out, nx_out)
        
        The input resolution is determined by x.shape[-1]
        The output resolution is determined by self.s_outputspace
        """
        # Lifting layer
        x_res = x.shape[-1]
        x = x.permute(0, 2, 1)
        x = torch.cat((x, get_grid1d(x.shape, x.device)), dim=-1)    # grid ``features''
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Map from input domain into the torus
        x = F.pad(x, [0, x_res//self.padding])
        
        # Fourier integral operator layers on the torus
        for idx_layer, (speconv, w) in enumerate(zip(self.speconvs, self.ws)):
            if idx_layer != self.n_layers - 1:
                x = w(x) + speconv(x)
                x = self.act(x)
            else:
                # Change resolution in function space consistent way
                x = w(projector1d(x, s=self.s_outputspace)) + speconv(x, s=self.s_outputspace)

        # Map from the torus into the output domain
        if self.s_outputspace is not None:
            x = x[..., :-self.num_pad_outputspace]
        else:
            x = x[..., :-(x_res//self.padding)]
        
        # Final projection layer
        x = x.permute(0, 2, 1)
        x = self.mlp0(x)

        return x.permute(0, 2, 1)

    def set_outputspace_resolution(self, s=None):
        """
        Helper to set desired output space resolution of the model at any time
        """
        if s is None:
            self.s_outputspace = None
            self.num_pad_outputspace = None
        else:
            self.s_outputspace = s + s//self.padding
            self.num_pad_outputspace = s//self.padding
