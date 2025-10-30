import torch
from qnn.models.basic_models import ModelBase
from qnn.freeze_agents.freeze import FreezeAgent
from tomlkit.toml_file import TOMLFile
from math import exp


class ThresholdFreezer(FreezeAgent):
    def __init__(self, trainer, model:ModelBase, threshold=0, activation_time=100, *args, **kwargs):
        super().__init__(trainer,model)
        self.threshold = threshold
        self.activation_time = activation_time
        self.tensor_list = model.get_tensor_list()
        self.saved_tensor_list = [torch.zeros_like(tensor) for tensor in self.tensor_list]
        self.mask_list = [(torch.ones_like(tensor) < 0) for tensor in self.tensor_list]
    
    def update_mask_step(self):
        if self.trainer.epoch > self.activation_time:
            for tensor,saved_tensor,mask in zip(self.tensor_list,self.saved_tensor_list,self.mask_list):
                mask[tensor.abs() > self.threshold] = True
                saved_tensor.data[mask] = tensor.data[mask]
    
    def update_mask_epoch(self):
        pass
    
    def restore_tensors(self):
        if self.trainer.epoch > self.activation_time:
            for tensor,saved_tensor,mask in zip(self.tensor_list,self.saved_tensor_list,self.mask_list):
                tensor.data[mask] = saved_tensor.data[mask]
    
    def get_frozen(self):
        frozen = 0
        for mask in self.mask_list:
            frozen += mask.sum().item()
        return frozen
    
    def get_pfrozen(self):
        total_weights = 0
        for tensor in self.tensor_list:
            total_weights += tensor.numel()
        return (self.get_frozen()/total_weights)*100
            
class GradientFreezer(FreezeAgent):
    def __init__(self, trainer, model:ModelBase, threshold=50, activation_time=100, *args, **kwargs):
        super().__init__(trainer,model)
        self.activation_time = activation_time
        self.threshold = threshold*(len(trainer.train_set))
        self.tensor_list = model.get_tensor_list()
        self.saved_tensor_list = [torch.zeros_like(tensor) for tensor in self.tensor_list]
        self.momentum_list = [torch.zeros_like(tensor,dtype=torch.int64) for tensor in self.tensor_list]
        self.mask_list = [(torch.ones_like(tensor) < 0) for tensor in self.tensor_list]
        
        if self.trainer.manual_freeze:
            mf = TOMLFile("config_freeze/freeze0.toml").read()
            if mf['coordinates']:
                None
            if mf['slices']:
                for idx,slice in enumerate(mf['slices']):
                    if slice:
                        x1,x2,y1,y2 = mf['slices']
                        self.mask_list[idx][x1:x2,y1:y2] = True
            if mf['tensors']:
                None
                
    def update_mask_step(self):
        if self.trainer.epoch > self.activation_time:
            for tensor,saved_tensor,mask,momentum in zip(self.tensor_list,self.saved_tensor_list,self.mask_list,self.momentum_list):
                temp = torch.ones_like(tensor,dtype=torch.int64)
                temp[tensor.grad < 0] = -1
                momentum += temp
                mask[momentum.abs() > self.threshold] = True
                saved_tensor.data[mask] = tensor.data[mask]
                
    def update_mask_epoch(self):
        pass
                
    def restore_tensors(self):
        if self.trainer.epoch > self.activation_time:
            for tensor,saved_tensor,mask in zip(self.tensor_list,self.saved_tensor_list,self.mask_list):
                tensor.data[mask] = saved_tensor.data[mask]
    
    def get_frozen(self):
        return sum([mask.sum().item() for mask in self.mask_list])
    
    def get_pfrozen(self):
        total_weights = 0
        for tensor in self.tensor_list:
            total_weights += tensor.numel()
        return (self.get_frozen()/total_weights)*100
    

class OscillatorFreezer(FreezeAgent):
    def __init__(self, trainer, model:ModelBase, threshold=50, activation_time=100, osc_threshold=0.1, probability=0.25, *args, **kwargs):
        super().__init__(trainer,model)
        self.activation_time = activation_time
        self.threshold = threshold*(len(trainer.train_set))
        self.osc_threshold = osc_threshold*(len(trainer.train_set))
        self.probability = probability
        self.tensor_list = model.get_tensor_list()
        self.saved_tensor_list = [torch.zeros_like(tensor) for tensor in self.tensor_list]
        self.gradient_accumulated_sign_list = [torch.zeros_like(tensor,dtype=torch.int64) for tensor in self.tensor_list]
        self.weight_accumulated_sign_list = [torch.zeros_like(tensor,dtype=torch.int64) for tensor in self.tensor_list]
        self.le_gradient_accumulated_sign_list = [torch.zeros_like(tensor,dtype=torch.int64) for tensor in self.tensor_list]
        self.le_weight_accumulated_sign_list = [torch.zeros_like(tensor,dtype=torch.int64) for tensor in self.tensor_list]
        self.mask_list = [(torch.ones_like(tensor) < 0) for tensor in self.tensor_list]

    def update_mask_step(self):
        if self.trainer.epoch > self.activation_time:
            for tensor,saved_tensor,mask,grad_acc_sign,weight_acc_sign in zip(self.tensor_list,self.saved_tensor_list,self.mask_list,self.gradient_accumulated_sign_list,self.weight_accumulated_sign_list):
                grad_sign = torch.ones_like(tensor,dtype=torch.int64)
                grad_sign[tensor.grad < 0] = -1
                grad_acc_sign += grad_sign
                weight_sign = torch.ones_like(tensor,dtype=torch.int64)
                weight_sign[tensor < 0] = -1
                weight_acc_sign += grad_sign
                mask[(grad_acc_sign.abs() > self.threshold) & (weight_acc_sign.abs() > self.threshold)] = True
                saved_tensor.data[mask] = tensor.data[mask]
                
    def update_mask_epoch(self):
        if self.trainer.epoch > self.activation_time:
            for le_grad_acc_sign, grad_acc_sign, le_weight_acc_sign, weight_acc_sign, mask, tensor in zip(self.le_gradient_accumulated_sign_list, self.gradient_accumulated_sign_list, self.le_weight_accumulated_sign_list, self.weight_accumulated_sign_list, self.mask_list, self.tensor_list):
                # Detect the oscillators close to 0, and set them to 0 (equivalent to pruning them)
                oscillators = (le_grad_acc_sign - grad_acc_sign).abs() < self.osc_threshold
                zero_oscillators = oscillators & ((le_weight_acc_sign - weight_acc_sign).abs() < self.osc_threshold)
                zero_oscillators &= torch.rand_like(tensor) < self.probability
                mask[zero_oscillators] = True
                tensor.data[zero_oscillators] = 0.0
                #print(zero_oscillators.sum().item())
                # Detect oscillators far enough from 0, and set them to +-1
                non_zero_oscillators = oscillators & (weight_acc_sign.abs() > self.threshold)
                mask[non_zero_oscillators] = True
                #print(non_zero_oscillators.sum().item())
                
    def restore_tensors(self):
        if self.trainer.epoch > self.activation_time:
            for tensor,saved_tensor,mask in zip(self.tensor_list,self.saved_tensor_list,self.mask_list):
                tensor.data[mask] = saved_tensor.data[mask]
    
    def get_frozen(self):
        return sum([mask.sum().item() for mask in self.mask_list])
    
    def get_pfrozen(self):
        total_weights = 0
        for tensor in self.tensor_list:
            total_weights += tensor.numel()
        return (self.get_frozen()/total_weights)*100
    
    
class SimplerFreezer(FreezeAgent):
    def __init__(self, trainer, model:ModelBase, lt=0.05, ht=10.0, p=1.0, activation_time=100, *args, **kwargs):
        super().__init__(trainer,model)
        self.activation_time = activation_time
        self.lt = lt*(len(trainer.train_set)) # low threshold (to freeze oscillators to 0)
        self.ht = ht*(len(trainer.train_set)) # high threshold (to freeze steadies to +-1)
        self.p = p # probability of freezing the oscillators
        self.tensor_list = model.get_tensor_list()
        self.n = len(self.tensor_list)
        self.saved_tensor_list = [torch.zeros_like(tensor) for tensor in self.tensor_list]
        self.accumulators = [torch.zeros_like(tensor,dtype=torch.float32) for tensor in self.tensor_list]
        self.mask_list = [(torch.ones_like(tensor) < 0) for tensor in self.tensor_list]
    
    def update_mask_step(self):
        for idx in range(self.n):
            #Get the positive and negative masks
            positives = self.tensor_list[idx] > 0
            negatives = self.tensor_list[idx] < 0
            # If the accumulator was negative (accumulating negative signs), set it to 0, and start to accumulate positive signs
            self.accumulators[idx][positives] = torch.max(self.accumulators[idx][positives], torch.zeros_like(self.accumulators[idx][positives])) + 1
            # If the accumulator was positive (accumulating positive signs), set it to 0, and start to accumulate negative signs
            self.accumulators[idx][negatives] = torch.min(self.accumulators[idx][negatives], torch.zeros_like(self.accumulators[idx][negatives])) - 1
    
    def update_mask_epoch(self):
        if self.trainer.epoch > self.activation_time:
            for idx in range(self.n):
                # Update the mask for the current epoch
                mask = self.mask_list[idx]
                # Detect the oscillators close to 0, and set them to 0 (equivalent to pruning them)
                oscillators = (self.accumulators[idx].abs() < self.lt)
                # Set the oscillators to 0 (with probability p)
                oscillators &= torch.rand_like(self.accumulators[idx]) < self.p
                self.saved_tensor_list[idx].data[oscillators] = 0
                mask[oscillators] = True
                # Detect steadies far enough from 0, and set them to +-1
                steadies = self.accumulators[idx].abs() > self.ht
                mask[steadies] = True
                # Set the steadies to +-1
                self.saved_tensor_list[idx].data[steadies] = torch.sign(self.saved_tensor_list[idx].data[steadies])
                
    
    def restore_tensors(self):
        if self.trainer.epoch > self.activation_time:
            for tensor,saved_tensor,mask in zip(self.tensor_list,self.saved_tensor_list,self.mask_list):
                tensor.data[mask] = saved_tensor.data[mask]
    
    def get_frozen(self):
        frozen = 0
        for mask in self.mask_list:
            frozen += mask.sum().item()
        return frozen
    
    def get_pfrozen(self):
        total_weights = 0
        for tensor in self.tensor_list:
            total_weights += tensor.numel()
        return (self.get_frozen()/total_weights)*100