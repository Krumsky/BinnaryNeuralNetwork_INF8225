import torch
from qnn.models.basic_models import ModelBase

class FreezeAgent:
    def __init__(self, trainer, model: ModelBase, *args, **kwargs):
        self.trainer = trainer
        self.model = model
        weight_dict = model.get_binary_weights()
        self.weights = list(weight_dict.values())
        self.weights_idx = list(weight_dict.keys())
        self.saved_weights = [w.clone().detach() for w in self.weights]
        self.masks = [torch.ones_like(w) < 0 for w in self.weights]
    
    def update_mask_step(self):
        pass
    
    def update_mask_epoch(self):
        pass
    
    def restore_tensors(self):
        # Restore the weights from the saved tensors
        for w, s, m in zip(self.weights, self.saved_weights, self.masks):
            w.data[m] = s.data[m]
    
    def get_frozen(self):
        # Count the number of frozen weights in the mask
        cnt = 0
        for mask in self.masks:
            cnt += mask.sum().item()
        return cnt
    
    def get_pfrozen(self):
        # Calculate the percentage of frozen weights
        total_weights = sum(mask.numel() for mask in self.masks)
        if total_weights == 0:
            return 0.0
        return (self.get_frozen() / total_weights) * 100.0
    
class NoFreeze(FreezeAgent):
    def __init__(self, trainer, model:ModelBase, *args, **kwargs):
        super().__init__(trainer, model)
    
    def update_mask_step(self):
        return
    
    def update_mask_epoch(self):
        return
    
    def restore_tensors(self):
        return
    
    def get_frozen(self):
        return 0
    
    def get_pfrozen(self):
        return 0.0