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
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, preserve_norm=True) -> None :
        super().__init__(in_features, out_features, bias, device, dtype)
        self.binary = True
        
        # Norm Preservation
        self.preserve_norm = preserve_norm
        # Norm Parameter
        norm = self.weight.norm(1)
        norm = norm/(self.in_features * self.out_features) # Mean of the weight norm
        self.norm_param = nn.Parameter(norm, requires_grad=False)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.binary:
            if self.training:
                clip(self.weight, 1)
            wb = weightBinarize(self.weight)
            if self.preserve_norm:
                if not self.norm_param.requires_grad:
                    norm = self.weight.norm(1)
                    self.norm_param.data = norm/(self.in_features * self.out_features) # Mean of the weight norm
                    self.norm_param.requires_grad = True
                wb = wb * self.norm_param # Preserve the norm of the weights
        else:
            wb = self.weight
        return F.linear(input, wb, self.bias)
    
    def binary_mode(self, mode=3):
        """
        Set the model to binary mode.
        0 - full precision mode (no binarization)
        1 - weight binarization mode (weights are binarized, but activations are not)
        2 - activation binarization mode (activations are binarized, but weights are not)
        3 - full binarization mode (both weights and activations are binarized)
        Useful to perform floating point pre-training before quantization.
        """
        self.binary = (mode == 1) or (mode == 3)
    
class QATConv2d(nn.Conv2d):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', preserve_norm=True) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.binary = True
        
        # Norm Preservation
        self.preserve_norm = preserve_norm
        # Norm Parameter
        norm = self.weight.flatten(start_dim=-2, end_dim=-1).norm(1, dim=-1, keepdim=True).view((self.out_channels, self.in_channels, 1, 1)).repeat(1, 1, *self.kernel_size)
        norm = norm/(self.kernel_size[0] * self.kernel_size[1]) # Mean of the weight norm
        self.norm_param = nn.Parameter(norm, requires_grad=False)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.binary:
            if self.training:
                clip(self.weight, 1)
            wb = weightBinarize(self.weight)
            if self.preserve_norm:
                if not self.norm_param.requires_grad:
                    norm = self.weight.flatten(start_dim=-2, end_dim=-1).norm(1, dim=-1, keepdim=True).view((self.out_channels, self.in_channels, 1, 1)).repeat(1, 1, *self.kernel_size)
                    self.norm_param.data = norm/(self.kernel_size[0] * self.kernel_size[1]) # Mean of the weight norm
                    self.norm_param.requires_grad = True
                wb = wb * self.norm_param # Preserve the norm of the weights
        else:
            wb = self.weight
        return F.conv2d(input, wb, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
    def binary_mode(self, mode=3):
        """
        Set the model to binary mode.
        0 - full precision mode (no binarization)
        1 - weight binarization mode (weights are binarized, but activations are not)
        2 - activation binarization mode (activations are binarized, but weights are not)
        3 - full binarization mode (both weights and activations are binarized)
        Useful to perform floating point pre-training before quantization.
        """
        self.binary = (mode == 1) or (mode == 3)
        
        
class Binarize(nn.Module):
    """
    A module that binarizes the input tensor
    """
    def __init__(self, base_activation_func=F.relu):
        super(Binarize, self).__init__()
        self.base_activation_func = base_activation_func
        self.binary = True
    
    def forward(self, input: Tensor) -> Tensor:
        return activationBinarize(input) if self.binary else self.base_activation_func(input)
    
    def binary_mode(self, mode=3):
        """
        Set the model to binary mode.
        0 - full precision mode (no binarization)
        1 - weight binarization mode (weights are binarized, but activations are not)
        2 - activation binarization mode (activations are binarized, but weights are not)
        3 - full binarization mode (both weights and activations are binarized)
        Useful to perform floating point pre-training before quantization.
        """
        self.binary = (mode == 2) or (mode == 3)
    
    
class AutoPruningLinear(QATLinear):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None :
        super().__init__(in_features, out_features, bias, device, dtype)
        
    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            clip(self.weight, 1)
        return F.linear(input, weightBinarize(self.weight), self.bias)