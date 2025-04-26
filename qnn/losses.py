import torch
from torch import nn

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
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'BN' not in name:
                # Apply the regularization only to the weights that are binarized
                reg_loss += torch.mean(torch.abs(param.abs() - torch.ones_like(param)))
        
        # Compute the total loss
        total_loss = ce_loss + self.lbda * reg_loss
        
        return total_loss
    