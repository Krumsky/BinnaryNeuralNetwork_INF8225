import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class TwoStepBinaryTraining():
    def __init__(self, scheduler=None, fp_pretraining_end=300, model=None, *args, **kwargs):
        self.scheduler = scheduler
        self.epoch_cnt = 0
        self.fp_pretraining_end = fp_pretraining_end
        self.model = model
        self.model.binary_mode(2) # Go into activation binarization mode
        
    def step(self):
        self.epoch_cnt += 1
        if self.scheduler:
            self.scheduler.step() # Step the inner scheduler too
        if self.epoch_cnt == self.fp_pretraining_end:
            self.model.binary_mode(3) # Go into full binarization mode
            self.reset_optimizer_buffers() # Reset the optimizer to the base state dict
            self.scheduler.optimizer.weight_decay = 0.0 # Disable weight decay in full binarization mode
            
    def reset_optimizer_buffers(self):
        """
        Reset the optimizer to the base state dict.
        This is useful to reset the optimizer after the floating point pre-training phase.
        """
        if self.scheduler:
            state_dict = deepcopy(self.scheduler.optimizer.state_dict())
            state_dict['state'] = {} # Reset the state of the optimizer
            self.scheduler.optimizer.load_state_dict(state_dict)
        else:
            raise ValueError("No scheduler found, cannot reset optimizer.")