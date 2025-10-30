import torch
from torch import nn
from qnn.modules import QATLinear, QATConv2d

class CrossEntropyBinReg():
    def __init__(self, lbda=0.01, model=None):
        self.lbda = lbda
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
    
    def __call__(self, output, target):
        # Compute the cross entropy loss
        ce_loss = self.criterion(output, target)
        
        # Compute the regularization term
        reg_loss = 0
        n_params = 0
        for module in self.model.modules():
            if isinstance(module, (QATLinear, QATConv2d)):
                # # Check if the module is in binary mode
                # if not module.binary: # If not, skip the regularization (the model is probably in fp pre-training mode)
                #     continue
                # Apply the regularization only to the weights that are binarized
                reg_loss += torch.sum(torch.abs(module.weight.abs() - torch.ones_like(module.weight)))
                n_params += module.weight.numel()
        reg_loss /= n_params if n_params > 0 else 1 # Average over all parameters
        
        # Compute the total loss
        total_loss = ce_loss + self.lbda * reg_loss
        
        return total_loss
    