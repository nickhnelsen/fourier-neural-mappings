import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input_batch, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input_batch, weights)

    def forward(self, x, s=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients
        x_ft = fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if s is None:
            x = fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        else:
            x = fft.irfft2(out_ft, s=s)

        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, s_outputspace, width_final=128, padding=8, d_out=1):
        """
        modes1, modes2 (int): Fourier mode truncation levels
        width (int): dimension of channel space
        s_outputspace (list or tuple, length 2): desired spatial resolution (s,s) in output space
        width_final (int): width of the final projection layer
        padding (int or float): (1.0/padding) is fraction of domain to zero pad (the input is not periodic)
        d_out (int): one output channel (co-domain dimension of output space functions)
        """
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width_final = width_final
        self.padding = padding
        self.d_out = d_out 
        self.num_pad_outputspace = tuple([s//self.padding for s in list(s_outputspace)])
        self.s_outputspace = tuple([s + s//self.padding for s in list(s_outputspace)])

        self.fc0 = nn.Linear(4, self.width) # input channel size is 4 (2 velocity inputs + 2 space variables)

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
        Input shape: (batch, channels=1, x, y)
        Output shape: (batch, channels=1, x, y)
        """
        # Lifting layer
        x = x.permute(0, 2, 3, 1)
        x = torch.cat((x, self.get_grid(x.shape, x.device)), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, x.shape[-1]//self.padding, 0, x.shape[-2]//self.padding])

        # Four Fourier integral operator layers
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x, s=self.s_outputspace)
        x2 = self.w3(x)
        x2 = self.projector(x2, s=self.s_outputspace)
        x = x1 + x2

        # Final projection layer
        x = x[..., :-self.num_pad_outputspace[-2], :-self.num_pad_outputspace[-1]]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x.permute(0, 3, 1, 2)
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def projector(self, x, s=None):
        """
        Either truncate or zero pad the Fourier modes of x to resolution s
        """
        if s is not None:
            x = fft.irfft2(fft.rfft2(x), s=s)
            
        return x