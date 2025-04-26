import matplotlib.pyplot as plt
import numpy as np
import torch
from model_training.config_helper import config_splitter
from model_training.builder import Builder
from qnn.modules import WeightBinarizeFunc
from tomlkit.toml_file import TOMLFile

# Plot a numpy matrix as a heatmap and saves it to a file
def plot_matrix(matrix, title='Matrix Heatmap', filepath='heatmap.svg'):
    """
    Plots a numpy matrix as a heatmap.
    
    Parameters:
        matrix (np.ndarray): The matrix to plot.
        title (str): The title of the plot.
    """
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filepath)
    plt.show()
    plt.close()

config = TOMLFile('config/weight_freeze/WeightFreeze.toml').read()
subconfig = config_splitter(config)[0][6]
b = Builder(subconfig)
b.load_model('runs/results_WeightFreeze/bMLP_0006.pth')
qtensor = WeightBinarizeFunc.apply(b.model.entry.weight.detach())
plus_tensor = qtensor.clone()
plus_tensor[plus_tensor > 0] = 1
plus_tensor[plus_tensor < 0] = 0
minus_tensor = qtensor.clone()
minus_tensor[minus_tensor > 0] = 0
minus_tensor[minus_tensor < 0] = -1
zero_tensor = qtensor.clone()
zero_tensor[zero_tensor > 0] = 1
zero_tensor[zero_tensor < 0] = 1
plot_matrix(qtensor.numpy(), title='Entry Layer Binary Weight Map', filepath='heatmap.svg')
plot_matrix(plus_tensor.numpy(), title='Entry Layer Binary Weight Map, 1 only', filepath='heatmap_plus.svg')
plot_matrix(minus_tensor.numpy(), title='Entry Layer Binary Weight Map, -1 only', filepath='heatmap_minus.svg')
plot_matrix(zero_tensor.numpy(), title='Entry Layer Binary Weight Map, 0 only', filepath='heatmap_zero.svg')
qtensor1 = WeightBinarizeFunc.apply(b.model.hidden1.weight.detach())
plot_matrix(qtensor1.numpy(), title='Hidden Layer 1 Binary Weight Map', filepath='heatmap1.svg')
