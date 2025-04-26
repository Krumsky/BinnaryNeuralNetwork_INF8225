import torch
from torch import nn
from torch.nn import functional as F

from qnn.modules import QATLinear,QATConv2d,clip

class ModelBase(nn.Module):
    
    def __init__(self):
        super(ModelBase, self).__init__()
        self.blocked = dict()
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_tensor_list(self):
        raise NotImplementedError