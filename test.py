import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets
from qnn.models.FracBNN import *

if __name__ == "__main__":
    dataset = {
        'train': datasets.CIFAR10("./datasets/CIFAR10", train=True, transform=T.ToTensor(), download=True),
        'test': datasets.CIFAR10("./datasets/CIFAR10", train=False, transform=T.ToTensor(), download=True)
    }
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    print(norm(dataset['train'][0]))
    #thermo = torch.Tensor([255, 223, 191, 159, 127, 95, 63, 31])
    t_list = [torch.full((3, 32, 32), i, dtype=torch.float32) for i in [255, 223, 191, 159, 127, 95, 63, 31]]
    for i in range(len(t_list)):
        t_list[i] = norm(t_list[i])
    thermo = torch.cat(t_list, dim=0).reshape(1, 24, 32, 32)
    #print(thermo)
    #print(thermo.shape)