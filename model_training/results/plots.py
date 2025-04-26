import matplotlib.pyplot as plt
from tomlkit.toml_file import TOMLFile
from csv import reader

COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'turquoise', 'lime']

def plot_accuracy_results(result_file_list, result_name_list, plot_filepath, show_plot=False):
    """
    Plots the validation accuracy of the models from the result file list in the same plot, and
    saves the plot in the plot_filepath if show_plot is False.
    """
    fig, ax = plt.subplots()
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
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Validation Accuracy')
    ax.legend()
    plt.savefig(plot_filepath + 'accuracy.png')
    plt.savefig(plot_filepath + 'accuracy.svg')
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
    ax.set_ylabel('Neurons')
    ax.legend()
    plt.savefig(plot_filepath + 'frozen.png')
    plt.savefig(plot_filepath + 'frozen.svg')
    if show_plot:
        plt.show()
    else:
        plt.close('all')
        
def plot_weight_histogram(model, plot_filepath, show_plot=False):
    """
    Plots the histogram of the weights of the model and saves it to the plot_filepath.
    """
    for name, param in model.named_parameters():
        p = param.flatten()
        if 'weight' in name and 'BN' not in name:
            plt.hist(p.detach().numpy(), bins=100)
            plt.title(f'Weight histogram for {name}')
            plt.savefig(plot_filepath + name.split('.')[0] + ".png")
            plt.savefig(plot_filepath + name.split('.')[0] + ".svg")
            if show_plot:
                plt.show()
            else:
                plt.close('all')