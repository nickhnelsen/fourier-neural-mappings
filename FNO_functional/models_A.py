import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

# TODO: simplify with real part of sum and split parameter into two weights
class LinearFunctional2d(nn.Module):
    def __init__(self, in_channels, modes1, modes2):
        super(LinearFunctional2d, self).__init__()

        self.in_channels = in_channels

        #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = (modes1//2)*2 # ensure even
        self.modes2 = (modes2//2)*2 # ensure even

        self.scale = 1.0/in_channels
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, 2*self.modes1, self.modes2 + 1, dtype=torch.cfloat))
    
    def half_to_full(self, rfft):
        """
        Takes Fourier mode weight matrix in half rfft format and contructs the full ``fft'' format matrix.
            Ensures the result corresponds to the Fourier transform of a real-valued function.
        Input
        ----
        rfft_mat: (..., 2*jmax, kmax + 1), matrix in rfft format with last dimension missing negative modes
            NOTE: 2*jmax must be even and kmax + 1 must be odd
            First index j ordering: j=0,1,...,jmax | -(jmax-1),-(jmax-2),...,-1
            Second index k ordering: k=0,1,...,kmax
        """
        jmax = rfft.shape[-2]//2
        kmax = rfft.shape[-1] - 1
        
        # Set special modes to zero imaginary part to make IFFT2 real-valued
        rfft_mat = torch.clone(rfft) # avoid in place operation on leaf tensors
        rfft_mat[..., 0, 0] = torch.real(rfft_mat[..., 0, 0]) + 0.j
        rfft_mat[..., jmax, 0] = torch.real(rfft_mat[..., jmax, 0]) + 0.j
        rfft_mat[..., jmax, kmax] = torch.real(rfft_mat[..., jmax, kmax]) + 0.j
        rfft_mat[..., 0, kmax] = torch.real(rfft_mat[..., 0, kmax]) + 0.j
        
        Mjk = rfft_mat[..., 1:jmax, 1:-1] # bottom right block
        Mmjk = rfft_mat[..., jmax:, 1:-1] # top right block
        Mjk = torch.cat((torch.flip(Mmjk.conj(),[-2,-1]), torch.flip(Mjk.conj(),[-2,-1])), -2)
        Mjk = torch.cat((torch.flip(rfft_mat[..., 0:1, 1:-1].conj(), [-1]), Mjk), -2)
        return torch.cat((rfft_mat, Mjk), -1)
    
    # Complex inner product
    def compl_integrate2d(self, input_func, weights):
        # (batch, in_channel, x,y ), (in_channel, x,y) -> (batch, in_channel)
        return torch.einsum("bixy,ixy->bi", input_func.conj(), weights)

    def forward(self, x):
        # Compute Fourier coeffcients
        x_ft = fft.rfft2(x, norm="forward") # project into L^2 orthonormal complex exponential basis
        x_ft = torch.cat((x_ft[..., :self.modes1 + 1, :self.modes2 + 1], \
                            x_ft[..., -self.modes1 + 1:, :self.modes2 + 1]), -2) # truncate modes
        
        x_ft = self.half_to_full(x_ft)
        
        # # Integrate in Fourier space
        # return torch.real(self.compl_integrate2d(x_ft, self.half_to_full(self.weights)))
        # Integrate in Fourier space
        return self.compl_integrate2d(x_ft, self.half_to_full(self.weights))

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients
        x_ft = fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        return fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FNO2dfinite(nn.Module):
    def __init__(self, modes1, modes2, width, width_final=128, padding=9, d_out=1):
        super(FNO2dfinite, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width_final = width_final # width of the final projection layer
        self.padding = padding # pad the domain if input is non-periodic
        self.d_out = d_out # number of output functionals

        self.fc0 = nn.Linear(3, self.width) # input channel size is 3 (scalar input + two space variables)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        
        self.func0 = LinearFunctional2d(self.width, self.modes1, self.modes2)

        self.fc1 = nn.Linear(self.width, self.width_final)
        self.fc2 = nn.Linear(self.width_final, self.d_out)

    def forward(self, x):
        """
        Input shape: (batch, channels=1, x, y)
        Output shape: (batch, channels=self.d_out)
        """
        
        # Lifting layer
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        # Three Fourier integral operator layers
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
        
        # One Fourier neural functional layer
        x = self.func0(x)

        # Final projection layer
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)