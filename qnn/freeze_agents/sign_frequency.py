import torch
from qnn.models.basic_models import ModelBase
from qnn.freeze_agents.freeze import FreezeAgent

# For debugging purposes
from time import sleep


class SignFrequency(FreezeAgent):
    def __init__(self, trainer, model: ModelBase, warmup=0, start=0, low_freq=0.25, high_freq=0.75, momentum=0.9, prob=0.1, *args, **kwargs):
        super().__init__(trainer, model)
        # Freeze agent hyperparameters
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.warmup = warmup
        self.start = start
        self.momentum = momentum
        self.prob = prob
        self.record_length = len(trainer.train_set)
        
        # Agent's saved tensors
        self.last_sign = [w.sign() for w in self.weights]
        self.sign_change = [torch.zeros_like(w) for w in self.weights] # Sign change count on one epoch
        self.velocity = [torch.zeros_like(w) for w in self.weights] # Velocity of the sign change count over all epochs

    def update_mask_step(self):
        """
        Update the sign change count for each weight tensor.
        """
        
        if self.trainer.epoch < self.warmup and self.trainer.epoch < self.start:
            # Not in warmup phase nor in training phase, skip the update
            return
        
        # Update the sign change count and last sign for each weight tensor
        for w, m, s, sc, ls in zip(self.weights, self.masks, self.saved_weights, self.sign_change, self.last_sign):
            # Update the number of sign changes
            sc += (w.sign() != ls).float().abs()
            # Update the last sign
            ls = w.sign()
            # Save the tensors
            s.data = w.clone().detach()
                

    def update_mask_epoch(self):
        """
        Update the masks based on the sign change frequency.
        High frequencies are frozen to zero, because they add noise to the weight matrix and can be nullified.
        Low frequencies are frozen to +-1, because they are almost always the same sign and therefore are likely to converge to the said value.
        """
        
        if self.trainer.epoch < self.warmup and self.trainer.epoch < self.start:
            # Not in warmup phase nor in training phase, skip the update
            return
        
        debug = True
        for w, m, s, sc, v in zip(self.weights, self.masks, self.saved_weights, self.sign_change, self.velocity):
            # Calculate the frequency of sign changes
            freq = sc / self.record_length
            
            # Accumulate the frequency in the velocity tensor
            v.data = (self.momentum * v + freq)/(1 + self.momentum)
            
            if debug:
                print(sc.flatten().max().item())
                print(sc.flatten().mean().item())
                print(v.flatten().max().item())
                print(v.flatten().mean().item())
                debug = False
            
            if self.trainer.epoch >= self.start: # Change the mask only after the start epoch
                # Update the mask for high frequencies
                hf = freq > self.high_freq
                hf = hf & (torch.rand_like(freq) < self.prob) # Randomly freeze some high frequencies
                m[hf] = True
                s[hf] = 0.0
                
                # Update the mask for low frequencies
                lf = freq < self.low_freq
                lf = lf & (torch.rand_like(freq) < self.prob) # Randomly freeze some low frequencies
                m[lf] = True
                # s[lf] = w.sign()[lf] # Don't change the saved weights to preserve norm
                
            # Reset the sign change count for the next recording
            sc.data = torch.zeros_like(w)
        
    def restore_tensors(self):
        
        if self.trainer.epoch < self.start:
            # Not in freezing phase, skip the restoration
            return
        
        with torch.no_grad():
            # Restore the weights from the saved tensors
            for w, s, m in zip(self.weights, self.saved_weights, self.masks):
                w.data[m] = s.data[m]


         
                
class SignFrequencyGV(FreezeAgent):
    def __init__(self, trainer, model: ModelBase, warmup=0, start=0, low_freq=0.25, high_freq=0.75, momentum=0.9, prob=0.1, *args, **kwargs):
        super().__init__(trainer, model)
        # Freeze agent hyperparameters
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.warmup = warmup
        self.start = start
        self.momentum = momentum
        self.prob = 1-prob # Save the inverted probability to scale the gradients
        self.record_length = len(trainer.train_set)
        
        # Reference on the weight-specific learning rate scales
        self.lr_scaling = [torch.ones_like(w) for w in self.weights] # Learning rate scaling tensors for each weight
        
        # Agent's saved tensors
        self.last_sign = [w.sign() for w in self.weights]
        self.sign_change = [torch.zeros_like(w) for w in self.weights] # Sign change count on one epoch
        self.velocity = [torch.zeros_like(w) for w in self.weights] # Velocity of the sign change count over all epochs
        
    def get_scale_list(self, parameters):
        """
        Get the learning rate dictionary for the optimizer.
        This is useful to set different learning rates for different weights.
        """
        # Generate the base learning rate scaling tensors
        for idx, param in enumerate(parameters):
            if idx in self.weights_idx:
                param.lr_scale = self.lr_scaling[self.weights_idx.index(idx)] # Assign the learning rate scaling tensor for the weight
            else:
                param.lr_scale = torch.ones_like(param) # Initialize the learning rate scaling tensors to ones

    def update_mask_step(self):
        """
        Update the sign change count for each weight tensor.
        """
        
        if self.trainer.epoch < self.warmup and self.trainer.epoch < self.start:
            # Not in warmup phase nor in training phase, skip the update
            return
        
        # Update the sign change count and last sign for each weight tensor
        for w, s, sc, ls in zip(self.weights, self.saved_weights, self.sign_change, self.last_sign):
            # Update the number of sign changes
            sc += (w.sign() != ls).float().abs()
            # Update the last sign
            ls = w.sign()
            # Save the tensors
            s.data = w.clone().detach()
                

    def update_mask_epoch(self):
        """
        Update the masks based on the sign change frequency.
        High frequencies are frozen to zero, because they add noise to the weight matrix and can be nullified.
        Low frequencies are frozen to +-1, because they are almost always the same sign and therefore are likely to converge to the said value.
        """
        
        if self.trainer.epoch < self.warmup and self.trainer.epoch < self.start:
            # Not in warmup phase nor in training phase, skip the update
            return
        
        debug = True
        for w, m, lr, sc, v in zip(self.weights, self.masks, self.lr_scaling, self.sign_change, self.velocity):
            # Calculate the frequency of sign changes
            freq = sc / self.record_length
            
            # Accumulate the frequency in the velocity tensor
            v.data = (self.momentum * v + freq)/(1 + self.momentum)
            
            if debug:
                print(sc.flatten().max().item())
                print(sc.flatten().mean().item())
                print(v.flatten().max().item())
                print(v.flatten().mean().item())
                debug = False
            
            if self.trainer.epoch >= self.start: # Change the mask only after the start epoch
                # Update the mask for high frequencies
                hf = freq > self.high_freq
                m[hf] = True
                with torch.no_grad():
                    w[hf] *= self.prob # Save the weights scaled by the probability                
                
                # Update the mask for low frequencies
                lf = freq < self.low_freq
                m[lf] = True
                # s[lf] = w.sign()[lf] # Don't change the saved weights to preserve norm
                
                # Scale down the gradient mask
                lr[m] *= self.prob
                # Zero out the scaling values if they are too small
                lr[lr < 1e-6] = 0.0
                # Zero the weights if their scaling is zero and they are high frequency
                with torch.no_grad():
                    w[(lr < 1e-6) & hf] = 0.0
                
            # Reset the sign change count for the next recording
            sc.data = torch.zeros_like(w)
        
    def restore_tensors(self):
        
        # No restoration to perform
        return
    
    def get_frozen(self):
        # Sum the lr scaling values to get the mean number of frozen weights
        # (e.g. a weight with a scaling of 0.5 is frozen, but not completely and count as 0.5 frozen weights)
        cnt = 0
        for scale in self.lr_scaling:
            cnt += (1-scale).sum().item()
        return cnt
    
    def get_pfrozen(self):
        # Calculate the percentage of frozen weights
        total_weights = sum(mask.numel() for mask in self.masks)
        if total_weights == 0:
            return 0.0
        return (self.get_frozen() / total_weights) * 100.0