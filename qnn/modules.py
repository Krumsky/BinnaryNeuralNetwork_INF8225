import torch
from torch import nn
from torch.nn import functional as F

from torch import Tensor

class WeightBinarizeFunc(torch.autograd.Function):
    """
    A function used to quantize a parameter during a forward pass and backward pass,
    but keeping the non-quantized parameter during weight update
    """
    @staticmethod
    def forward(ctx, tensor: Tensor):
        with torch.no_grad(): # don't retain the operations to keep the gradient of the quantized tensor for the non-quantized
            qTensor = tensor.clone()
            qTensor[qTensor < 0] = -1
            qTensor[qTensor == 0] = 0 # A zero value means the weight has been pruned
            qTensor[qTensor > 0] = 1
        return qTensor
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
def weightBinarize(tensor):
    return WeightBinarizeFunc.apply(tensor)


class ActivationBinarizeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: Tensor):
        with torch.no_grad(): # don't retain the operations to keep the gradient of the quantized tensor for the non-quantized
            qTensor = tensor.clone()
            ctx.saturation_mask = (qTensor.abs() > 1)
            qTensor[qTensor < 0] = -1
            qTensor[qTensor >= 0] = 1
        return qTensor
    
    @staticmethod
    def backward(ctx, grad_output):
        out = grad_output.clone()
        out[ctx.saturation_mask] = 0
        return out, None
    
def activationBinarize(tensor):
    return ActivationBinarizeFunc.apply(tensor)
    

def clip(tensor, bits):
    with torch.no_grad():
        if bits == 1:
            tensor[tensor < -1] = -1
            tensor[tensor > 1] = 1
        else:
            m = -2**(bits-1)
            M = 2**(bits-1)-1
            tensor[tensor < m] = m
            tensor[tensor > M] = M
        

class QATLinear(nn.Linear):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None :
        super().__init__(in_features, out_features, bias, device, dtype)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            clip(self.weight, 1)
        return F.linear(input, weightBinarize(self.weight), self.bias)
    
class QATConv2d(nn.Conv2d):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros') -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            clip(self.weight, 1)
        return F.conv2d(input, weightBinarize(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        
class Binarize(nn.Module):
    """
    A module that binarizes the input tensor
    """
    def __init__(self):
        super(Binarize, self).__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        return activationBinarize(input)
    
    
class AutoPruningLinear(QATLinear):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None :
        super().__init__(in_features, out_features, bias, device, dtype)
        
    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            clip(self.weight, 1)
        return F.linear(input, weightBinarize(self.weight), self.bias)