import torch
import torch.nn.functional as F


def process_Abar(Abar, FLAG_2D=False):
    """
    Process 2by2 matrix Abar to a dim=3 vector
    Input
        v:      (N, d_c, 2, 2) torch tensor
    """
    pass

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

def get_qoi(A, Chi):
    """
    Extract all 6 QoIs from final time solution state on [0,1]^2
    Input
        y:      (..., nx1, nx2) torch tensor
    Output
        qoi:    (..., 6) torch tensor
    """
    
    return None
