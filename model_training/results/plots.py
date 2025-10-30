import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tomlkit.toml_file import TOMLFile
import torch.nn as nn
from qnn.modules import QATLinear, QATConv2d
from csv import reader
import os

COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'turquoise', 'lime']

def plot_accuracy_results(result_file_list, result_name_list, plot_filepath, show_plot=False):
    """
    Plots the validation accuracy of the models from the result file list in the same plot, and
    saves the plot in the plot_filepath if show_plot is False.
    """
    fig, ax = plt.subplots()
    max_val_acc = 0.0
    for result_file, name, color in zip(result_file_list, result_name_list, COLORS):
        with open(result_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            val_acc_idx = header.index('acc1')
            epochs_idx = header.index('epoch')
            val_acc = []
            epochs = []
            for row in csv_reader:
                val_acc.append(float(row[val_acc_idx]))
                epochs.append(int(row[epochs_idx]))
            ax.plot(epochs, val_acc, label=name, color=color)
            if max(val_acc) > max_val_acc:
                max_val_acc = max(val_acc)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Validation Accuracy')
    ax.legend()
    # Draw a horizontal line at the maximum validation accuracy
    ax.axhline(y=max_val_acc, color='black', linestyle='dashed', label='Max Validation Accuracy')
    # Add a text annotation for the maximum validation accuracy
    ax.text(epochs[-1], max_val_acc, f'Max: {max_val_acc:.2f}', color='black', va='bottom', ha='right')
    plt.savefig(plot_filepath + 'accuracy.png')
    plt.savefig(plot_filepath + 'accuracy.svg')
    if show_plot:
        plt.show()
    else:
        plt.close('all')
        
def plot_error_rate_results(result_file_list, result_name_list, plot_filepath, show_plot=False):
    """
    Plots the validation error rate of the models from the result file list in the same plot, and
    saves the plot in the plot_filepath if show_plot is False.
    The plot is clipped at 30% error rate.
    """
    fig, ax = plt.subplots()
    min_val_error = 100.0
    for result_file, name, color in zip(result_file_list, result_name_list, COLORS):
        with open(result_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            val_acc_idx = header.index('acc1')
            epochs_idx = header.index('epoch')
            val_error = []
            epochs = []
            for row in csv_reader:
                val_acc = float(row[val_acc_idx])
                val_error.append(100 - val_acc)
                epochs.append(int(row[epochs_idx]))
            ax.plot(epochs, val_error, label=name, color=color)
            if min(val_error) < min_val_error:
                min_val_error = min(val_error)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Validation Error Rate')
    ax.legend()
    # Draw a horizontal line at the minimum validation error rate
    ax.axhline(y=min_val_error, color='black', linestyle='dashed', label='Min Validation Error Rate')
    # Add a text annotation for the minimum validation error rate
    ax.text(epochs[-1], min_val_error, f'Min: {min_val_error:.2f}', color='black', va='top', ha='right')
    # Clip the y-axis at 30% error rate
    ax.set_ylim(0, 30)
    plt.savefig(plot_filepath + 'error_rate.png')
    plt.savefig(plot_filepath + 'error_rate.svg')
    if show_plot:
        plt.show()
    else:
        plt.close('all')
        
def plot_loss_results(result_file_list, result_name_list, plot_filepath, show_plot=False):
    """
    Plots the training and validation losses of the models from the result file list in the same plot, and
    saves the plot in the plot_filepath if show_plot is False.
    """
    fig, ax = plt.subplots()
    for result_file, name, color in zip(result_file_list, result_name_list, COLORS):
        with open(result_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            train_loss_idx = header.index('train_loss')
            val_loss_idx = header.index('test_loss')
            epochs_idx = header.index('epoch')
            train_loss = []
            val_loss = []
            epochs = []
            for row in csv_reader:
                train_loss.append(float(row[train_loss_idx]))
                val_loss.append(float(row[val_loss_idx]))
                epochs.append(int(row[epochs_idx]))
            ax.plot(epochs, train_loss, label=name + ' Train', color=color, linestyle='dashed')
            ax.plot(epochs, val_loss, label=name + ' Validation', color=color)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(plot_filepath + 'losses.png')
    plt.savefig(plot_filepath + 'losses.svg')
    if show_plot:
        plt.show()
    else:
        plt.close('all')
    
    
def plot_weight_histogram(result_file_name, model, show_plot=False):
    for name, param in model.named_parameters():
        p = param.flatten()
        if 'weight' in name and 'BN' not in name:
            plt.hist(p.detach().numpy(), bins=100)
            plt.title(f'Weight histogram for {name}')
            plt.savefig(result_file_name + '_' + name.split('.')[0] + ".png")
            plt.savefig(result_file_name + '_' + name.split('.')[0] + ".svg")
            if show_plot:
                plt.show()
            else:
                plt.close('all')
                
def plot_frozen_evolution(result_file_names, names, plot_filepath, show_plot=False):
    """
    Plots the evolution of the number of frozen weights during training, taking the number of frozen weights per epoch from the result csv file.
    The plot also contains the total number of weights in the model, tanken directly from the model itself.
    """
    
    fig, ax = plt.subplots()
    for result_file_name, name, color in zip(result_file_names, names, COLORS):
        with open(result_file_name, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            frozen_idx = header.index('%freeze')
            epochs_idx = header.index('epoch')
            frozen = []
            epochs = []
            for row in csv_reader:
                frozen.append(float(row[frozen_idx]))
                epochs.append(int(row[epochs_idx]))
        ax.plot(epochs, frozen, label=name, color=color)
        
    ax.plot(epochs, [100.0]*len(epochs), linestyle='dashed', color='black')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('% Weights')
    ax.legend()
    plt.savefig(plot_filepath + 'frozen.png')
    plt.savefig(plot_filepath + 'frozen.svg')
    if show_plot:
        plt.show()
    else:
        plt.close('all')
        
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
            filepath = plot_filepath + 'weight_histograms/' + name.replace('.', '/')
            directory = '/'.join(filepath.split('/')[:-1])
            os.makedirs(directory, exist_ok=True)
            plt.savefig(filepath + ".png")
            plt.savefig(filepath + ".svg")
            if show_plot:
                plt.show()
            else:
                plt.close('all')