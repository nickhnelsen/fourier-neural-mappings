import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, widths, act="gelu"):
        super(DNN, self).__init__() 
        
        self.widths = widths
        self.layer = len(widths) - 1
        self.fcs = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(widths[0:-1], widths[1:])])
        self.act = _get_act(act)

    # x represents our data
    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i != self.layer - 1:
                x = self.act(x)
        return x

    
    

    
def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    elif act == 'none':
        func = None
    else:
        raise ValueError(f'{act} is not supported')
    return func