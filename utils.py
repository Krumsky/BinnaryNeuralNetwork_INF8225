import torch
import torch.nn as nn
from qnn.modules import QATLinear, QATConv2d
from qnn.models.basic_models import ModelBase
from torch.linalg import norm
from typing import List

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    r"""
    Computes the acc@k for the specified values of k
    """
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k <= output.shape[1]:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(torch.zeros(1, device=target.device) - 1.0)
    return res

def normalize_output(y:torch.Tensor):
    y = y.swapaxes(0,1)
    y = y.flatten(1)
    n = norm(y, 1, 1, keepdim=True)
    y = y/n
    y = y.nan_to_num(nan=0.0) # if an element is 0 and its vector's norm is 0, it would be nan, so we set it back to 0
    return y

def optim_accumulator_lambda(optimizer, param_idx_list: List[int], func: callable):
    """Calls a function with the following signature:
    func(parameter_accumulator: Tensor) -> Tensor
    on the accumulator (momentum for sgd, exp_avg for adam, etc.) of the parameters in the list param_idx_list,
    returns the new value of the accumulator.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        param_idx_list (List[int]): List of parameter indices to apply the function to.
        func (callable): function to apply to the accumulator.
    """
    # Find the type of the optimizer
    if isinstance(optimizer, torch.optim.SGD):
        # SGD uses momentum
        accumulator_name = 'momentum_buffer'
    elif isinstance(optimizer, torch.optim.Adam):
        # Adam uses exp_avg
        accumulator_name = 'exp_avg'
    
    state_dict = optimizer.state_dict()
    for param_idx in param_idx_list:
        state_dict['state'][param_idx][accumulator_name] = func(state_dict['state'][param_idx][accumulator_name])
    