import torch
from torch import nn
from typing import Dict

from qnn.modules import QATLinear,QATConv2d,Binarize

class ModelBase(nn.Module):
    
    def __init__(self):
        super(ModelBase, self).__init__()
        self.blocked = dict()
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_binary_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get the binary weights of the model.
        Returns a dict of number of modules as keys and tensors with the binary weights of the model as values.
        """
        binary_weights = {}
        for name, submodule in self.named_modules():
            if submodule == self: # Skip the model itself
                continue
            if isinstance(submodule, (QATLinear, QATConv2d)):
                # Find the parameter index
                idx = None
                for param_idx, (param_name, _) in enumerate(self.named_parameters()):
                    if param_name == name + '.weight':
                        idx = param_idx
                        break
                if idx is None:
                    raise ValueError(f"Could not find parameter for module {name}.")
                binary_weights[idx] = submodule.weight
        return binary_weights
    
    def binary_mode(self, mode=3):
        """
        Set the model to binary mode.
        0 - full precision mode (no binarization)
        1 - weight binarization mode (weights are binarized, but activations are not)
        2 - activation binarization mode (activations are binarized, but weights are not)
        3 - full binarization mode (both weights and activations are binarized)
        Useful to perform floating point pre-training before quantization.
        """
        for idx, module in enumerate(self.modules()):
            if idx == 0: # Self
                continue
            if isinstance(module, ModelBase):
                module.binary_mode(mode)
            elif isinstance(module, QATLinear) or isinstance(module, QATConv2d) or isinstance(module, Binarize):
                module.binary_mode(mode)