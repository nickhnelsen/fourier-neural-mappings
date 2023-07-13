import torch.nn as nn

from .shared import SpectralConv2d, LinearDecoder2d, _get_act, MLP


class FND2d(nn.Module):
    """
    Fourier Neural Decoder for mapping finite-dimensional vectors to functions
    """
    def __init__(self, d_in, s_outputspace,
                 modes1=12,
                 modes2=12,
                 width=32,
                 width_initial=128,
                 width_final=128,
                 padding=8,
                 d_out=1,
                 width_ldec=None,
                 act='gelu',
                 n_layers=4
                 ):
        """
        d_in            (int): number of input channels (dimension of input vectors)
        s_outputspace   (list or tuple, length 2): desired spatial resolution (s,s) in output space
        modes1, modes2  (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        width_initial   (int): width of the initial processing layer
        width_final     (int): width of the final projection layer
        padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
        d_out           (int): finite number of desired outputs (number of functions)
        width_ldec      (int): input channel width for FND layer
        act             (str): Activation function = tanh, relu, gelu, elu, or leakyrelu
        n_layers        (int): Number of Fourier Layers, by default 4
        """
        super(FND2d, self).__init__()

        self.d_in = d_in
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width_initial = width_initial
        self.width_final = width_final
        self.padding = padding
        self.d_out = d_out 
        if width_ldec is None:
            self.width_ldec = self.width
        else:
            self.width_ldec = width_ldec
        self.act = _get_act(act)
        self.n_layers = n_layers
        if self.n_layers is None:
            self.n_layers = 4
            
        self.set_outputspace_resolution(s_outputspace)
        
        self.mlp0 = MLP(self.d_in, self.width_initial, self.width_ldec, act)
        
        self.ldec0 = LinearDecoder2d(self.width_ldec, self.width, self.modes1, self.modes2)
        
        self.speconvs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2) 
                for _ in range(self.n_layers - 1)]
            )

        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
                for _ in range(self.n_layers - 1)]
            )
        
        self.mlp1 = MLP(self.width, self.width_final, self.d_out, act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, self.d_in)
        Output shape:           (batch, self.d_out, nx_out, ny_out)
        
        The output resolution is determined by self.s_outputspace
        """
        # Nonlinear processing layer
        x = self.mlp0(x)
        
        # Decode into functions on the torus
        x = self.ldec0(x, self.s_outputspace)
        
        # Fourier integral operator layers on the torus
        for idx_layer, (speconv, w) in enumerate(zip(self.speconvs, self.ws)):
            x = w(x) + speconv(x)
            if idx_layer != self.n_layers - 2:
                x = self.act(x)
           
        # Map from the torus into the output domain
        x = x[..., :-self.num_pad_outputspace[-2], :-self.num_pad_outputspace[-1]]
        
        # Final projection layer
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)

        return x.permute(0, 3, 1, 2)
    
    def set_outputspace_resolution(self, s):
        """
        Helper to set desired output space resolution of the model at any time
        """
        self.s_outputspace = tuple([r + r//self.padding for r in list(s)])
        self.num_pad_outputspace = tuple([r//self.padding for r in list(s)])
