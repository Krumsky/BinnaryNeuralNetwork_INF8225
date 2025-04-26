import torch
from matplotlib import pyplot as plt
from tomlkit.toml_file import TOMLFile
from csv import reader
import argparse


from model_training import builder

def plot_accuracy_results(result_file_list, result_name_list, plot_filepath, show_plot=False):
    """
    Plots the validation accuracy of the models from the result file list in the same plot, and
    saves the plot in the plot_filepath if show_plot is False.
    """
    fig, ax = plt.subplots()
    for result_file,name in zip(result_file_list,result_name_list):
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
            ax.plot(epochs, val_acc, label=name + ' Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy top1')
    ax.legend()
    if plot_filepath is None:
        plt.savefig('plot.png')
        plt.savefig('plot.svg')
    else:
        plt.savefig(plot_filepath + '.png')
        plt.savefig(plot_filepath + '.svg')
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
    for result_file,name in zip(result_file_list,result_name_list):
        with open(result_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            #train_loss_idx = header.index('train_loss')
            val_loss_idx = header.index('test_loss')
            epochs_idx = header.index('epoch')
            #train_loss = []
            val_loss = []
            epochs = []
            for row in csv_reader:
                #train_loss.append(float(row[train_loss_idx]))
                val_loss.append(float(row[val_loss_idx]))
                epochs.append(int(row[epochs_idx]))
            #ax.plot(epochs, train_loss, label=name + ' Train', linestyle='dashed')
            ax.plot(epochs, val_loss, label=name + ' Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    if plot_filepath is None:
        plt.savefig('plot.png')
        plt.savefig('plot.svg')
    else:
        plt.savefig(plot_filepath + '.png')
        plt.savefig(plot_filepath + '.svg')
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
                
def plot_frozen_evolution(result_file_names, names, plot_filepath=None, model=None, show_plot=False):
    """
    Plots the evolution of the number of frozen weights during training, taking the number of frozen weights per epoch from the result csv file.
    The plot also contains the total number of weights in the model, tanken directly from the model itself.
    """
    
    total_weights = 0
    for name, param in model.named_parameters():
        total_weights += param.numel()
    
    fig, ax = plt.subplots()
    for result_file_name,name in zip(result_file_names,names):
        with open('runs/results_'+result_file_name+'/main.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            frozen_idx = header.index('frozen_neurons')
            epochs_idx = header.index('epoch')
            frozen = []
            epochs = []
            for row in csv_reader:
                frozen.append(min(float(row[frozen_idx])+524288,total_weights))
                epochs.append(int(row[epochs_idx]))
        ax.plot(epochs, frozen, label=name)
        
    ax.plot(epochs, [total_weights]*len(epochs), label='Total Neurons', linestyle='dashed')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Neurons')
    ax.legend()
    if plot_filepath is None:
        for result_file_name in result_file_names:
            plot_filepath = 'runs/results_' + result_file_name + '/plots/frozen_evolution'
            plt.savefig(plot_filepath + '.png')
            plt.savefig(plot_filepath + '.svg')
    else:
        plt.savefig(plot_filepath + '.png')
        plt.savefig(plot_filepath + '.svg')
    if show_plot:
        plt.show()
    else:
        plt.close('all')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Results",
        description="Generate plots and tables from the results of the training of the models.",)

    parser.add_argument('configfile',
                        help="the name of the config file to run",
                        type=str,
                        default='config_results/default.toml')

    try:
        args = parser.parse_args()

        filepath = args.configfile
    except:
        filepath = 'config_results/default.toml'

    try:
        d = TOMLFile(filepath).read()
    except:
        print("config file not found, exitting...")
        exit()
    
    for rfile_list, name_list, pfile in zip(d['files']['rfile_lists'], d['files']['name_lists'], d['files']['pfile_list']):
        if d['measures']['losses']:
            plot_loss_results(rfile_list, name_list, pfile)
        if d['measures']['accuracy']:
            plot_accuracy_results(rfile_list, name_list, pfile)
        if d['measures']['frozen']:
            config = TOMLFile('config/' + rfile_list[0] + '.toml').read()
            model = builder.Builder(config["model"], config["dataset"]).model
            plot_frozen_evolution(rfile_list, name_list, model=model)