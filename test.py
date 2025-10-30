import torch
import torch.nn as nn
from qnn.models.FracBNN import *
import matplotlib.pyplot as plt
from qnn.modules import QATLinear, QATConv2d, Binarize

def plot_weight_histogram(model:nn.Module, plot_filepath, show_plot=False):
    """
    Plots the histogram of the weights of the model and saves it to the plot_filepath.
    """
    for name, module in model.named_modules():
        if isinstance(module, (QATLinear, QATConv2d)):
            param = module.weight
            p = param.flatten()
            plt.hist(p.detach().numpy(), bins=100)
            plt.title(f'Weight histogram for {name}')
            plt.savefig(plot_filepath + name.split('.')[0] + ".png")
            plt.savefig(plot_filepath + name.split('.')[0] + ".svg")
            if show_plot:
                plt.show()
            else:
                plt.close('all')
                