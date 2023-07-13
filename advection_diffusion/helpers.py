import torch
import torch.nn.functional as F


def process_velocity(v, FLAG_2D=False):
    """
    Process velocity input for appropriate FNM model
    Input
        v:      (N, d_c, nx) torch tensor
    """
    if v.ndim != 3:
        raise ValueError("input must have ndim=3 with shape (N, d_c, nx)")
    elif FLAG_2D:
        v = v.unsqueeze(-1).repeat(1, 1, 1, v.shape[-1])
    return v

def trapz2(x, dx, dy=None):
    """
    2D trapezoid rule quadrature on a square
    Input
        x:      (..., nx, ny)
    Output
        x:      (...,)
    """
    if dy is None:
        dy = dx
    return torch.trapz(torch.trapz(x, dx=dy), dx=dx)

def get_qoi(y):
    """
    Extract all 6 QoIs from final time solution state on [0,1]^2
    Input
        y:      (..., nx1, nx2) torch tensor
    Output
        qoi:    (..., 6) torch tensor
    """
    N_qoi = 6
    dev = y.device
    qoi = torch.zeros(*y.shape[:-2], N_qoi, device=dev)

    s = y.shape[-2:]
    dx1 = 1. / (s[0] - 1)
    dx2 = 1. / (s[1] - 1)

    # QoI 0: point evaluation at the center of the square    
    idx_qoi = torch.div(torch.tensor(s, device=dev), 2, rounding_mode="floor")
    qoi[...,0] = y[..., idx_qoi[-2], idx_qoi[-1]]

    # QoI 1: mean
    y_bar = trapz2(y, dx=dx1, dy=dx2)[...,None,None]
    qoi[...,1] = y_bar[...,0,0]
    
    # QoI 2: standard deviation
    y_std = torch.sqrt(trapz2((y - y_bar)**2, dx=dx1, dy=dx2))
    qoi[...,2] = y_std
    
    # QoI 3: skewness
    qoi[...,3] = trapz2((y - y_bar)**3, dx=dx1, dy=dx2) / (y_std**3)
    
    # QoI 4: excess kurtosis
    qoi[...,4] = -3. + trapz2((y - y_bar)**4, dx=dx1, dy=dx2) / (y_std**4)
    
    # QoI 5: relative (concentration) entropy
    y = F.relu(y)
    y_bar = trapz2(y, dx=dx1, dy=dx2)
    qoi[...,5] = torch.special.entr(y_bar) - trapz2(torch.special.entr(y), dx=dx1, dy=dx2)
    
    return qoi