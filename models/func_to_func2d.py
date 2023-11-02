import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import SpectralConv2d, projector2d, get_grid2d, _get_act, MLP


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for mapping functions to functions
    """
    def __init__(self,
                 modes1=12,
                 modes2=12,
                 width=32,
                 s_outputspace=None,
                 width_final=128,
                 padding=8,
                 d_in=1,
                 d_out=1,
                 act='gelu',
                 n_layers=4,
                 get_grid=True
                 ):
        """
        modes1, modes2  (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        s_outputspace   (list or tuple, length 2): desired spatial resolution (s,s) in output space
        width_final     (int): width of the final projection layer
        padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
        d_in            (int): number of input channels (NOT including grid input features)
        d_out           (int): number of output channels (co-domain dimension of output space functions)
        act             (str): Activation function = tanh, relu, gelu, elu, or leakyrelu
        n_layers        (int): Number of Fourier Layers, by default 4
        get_grid        (bool): Whether or not append grid coordinate as a feature for the input
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
        self.act = _get_act(act)
        self.n_layers = n_layers
        self.get_grid = get_grid
        if self.n_layers is None:
            self.n_layers = 4
        
        self.set_outputspace_resolution(s_outputspace)

        self.fc0 = nn.Linear((self.d_in + self.d_physical if get_grid else self.d_in), self.width)
        
        self.speconvs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
                for _ in range(self.n_layers)]
            )

        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
                for _ in range(self.n_layers)]
            )

        self.mlp0 = MLP(self.width, self.width_final, self.d_out, act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, channels_in, nx_in, ny_in)
        Output shape:           (batch, channels_out, nx_out, ny_out)
        
        The input resolution is determined by x.shape[-2:]
        The output resolution is determined by self.s_outputspace
        """
        # Lifting layer
        x_res = x.shape[-2:]
        x = x.permute(0, 2, 3, 1)
        if self.get_grid:
            x = torch.cat((x, get_grid2d(x.shape, x.device)), dim=-1)   # grid ``features''
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Map from input domain into the torus
        x = F.pad(x, [0, x_res[-1]//self.padding, 0, x_res[-2]//self.padding])

        # Fourier integral operator layers on the torus
        for idx_layer, (speconv, w) in enumerate(zip(self.speconvs, self.ws)):
            if idx_layer != self.n_layers - 1:
                x = w(x) + speconv(x)
                x = self.act(x)
            else:
                # Change resolution in function space consistent way
                x = w(projector2d(x, s=self.s_outputspace)) + speconv(x, s=self.s_outputspace)

        # Map from the torus into the output domain
        if self.s_outputspace is not None:
            x = x[..., :-self.num_pad_outputspace[-2], :-self.num_pad_outputspace[-1]]
        else:
            x = x[..., :-(x_res[-2]//self.padding), :-(x_res[-1]//self.padding)]
        
        # Final projection layer
        x = x.permute(0, 2, 3, 1)
        x = self.mlp0(x)

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
