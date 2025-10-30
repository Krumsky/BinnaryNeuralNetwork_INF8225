from model_training.results.plots import *
from model_training.config_helper import *
from model_training.builder import Builder
from copy import deepcopy
import os

class Plotter():
    def __init__(self, config, config_filepath):
        
        self.base_config = config
        self.config_filename = config_filepath.split('/')[-1].split('.')[0] # name of the config file
        self.results_config = config['results']
        for attr_name in self.results_config:
            # set model attributes
            self.__setattr__(attr_name, self.results_config[attr_name])
        
        # Re-generate the sub-configs
        self.subconfigs, self.combinations = config_splitter(config)
        self.subconfigs_dict = {idx: subconfig for idx, subconfig in enumerate(self.subconfigs)}
            
    def plot(self):
        """ 
        In the "explore" attribute, every varying hyperparameter that we want to show on the same plot is specified.
        The plot function will draw one plot with the results for each varying hyperparameter value, for each set of other hyperparameters.
        E.g. if epochs = [100, 200] and model = ["MLP", "VGG"] and explore = ["epochs"], we draw 2 plots, one for each model, 
        with the results for both epochs.
        In the "plots" attribute, we specify the plots to be drawn (e.g. accuracy, loss, %frozen... etc.]).
        To draw the plots, we use the functions from plots.py. 
        """
        ###################################
        # Draw the cross-comparison plots #
        ###################################
        
        # Get the varying parameters' names
        multi_params_names = get_multi_params(self.base_config)[1]
        
        # Loop over the parameters to explore
        for explore_param in self.explore:
            # Give a name to the directory where the plots will be saved and with pivot the parameter to explore
            explore_directory = 'pivot=' + explore_param + '/'
            # If the parameter is not in the config, skip it
            if explore_param not in multi_params_names:
                continue
            # Find the index of the parameter to explore
            explore_param_idx = multi_params_names.index(explore_param)
            other_names = deepcopy(multi_params_names)
            other_names.pop(explore_param_idx)
            
            # Build the dictionary of identical subconfigs, excepting the parameter to explore
            # The keys are the combinations of the other parameters and the values are the dictionary of subconfig numbers
            subconfigs_dict = {}
            subconfigs_dict_keys = [] # We have to build such a list since our tuples are not hashable
            for idx, combination in enumerate(self.combinations):
                # Get the explore parameter out of the combination
                other_combination = combination[:explore_param_idx] + combination[explore_param_idx+1:]
                # Create the key for the inner dictionary (with keys being the different values of the explore parameter)
                if explore_param == 'model' or explore_param == 'freeze_agent':
                    key = combination[explore_param_idx][1]['name'] # Get the name of the model from the model/freeze args
                else:
                    key = combination[explore_param_idx]
                if other_combination not in subconfigs_dict_keys:
                    subconfigs_dict_keys.append(other_combination)
                    subconfigs_dict[len(subconfigs_dict_keys)-1] = {key: self.subconfigs[idx]['model']['model_args']['name']}
                else:
                    subconfigs_dict[subconfigs_dict_keys.index(other_combination)][key] = self.subconfigs[idx]['model']['model_args']['name']

            # Base filepath for the plot
            result_filepath = f'runs/results_{self.config_filename}/'
            base_filepath = result_filepath + 'plots/' + explore_directory
            for name in other_names:
                if name == 'model':
                    base_filepath += '{}/'
                    continue
                if name == 'freeze_agent':
                    base_filepath += '{}={}/'
                    continue
                base_filepath += name + '={}/'
            # Plot the results for each combination of the other parameters
            for oc_idx, explore_subconfigs in subconfigs_dict.items():
                other_combination = subconfigs_dict_keys[oc_idx]
                # format the filepath with the correct parameter values
                if 'model' in other_names and 'freeze_agent' in other_names:
                    idx1 = other_names.index('model')
                    idx2 = other_names.index('freeze_agent')
                    new_combination = other_combination[:idx1] + (other_combination[idx1][1]['name'][:-5],) + other_combination[idx1+1:idx2] + (other_combination[idx2][0],other_combination[idx2][1]['name']) + other_combination[idx+1:]
                elif 'model' in other_names:
                    idx = other_names.index('model')
                    new_combination = other_combination[:idx] + (other_combination[idx][1]['name'][:-5],) + other_combination[idx+1:]
                elif 'freeze_agent' in other_names:
                    idx = other_names.index('freeze_agent')
                    new_combination = other_combination[:idx] + (other_combination[idx][0],other_combination[idx][1]['name']) + other_combination[idx+1:]
                else:
                    new_combination = other_combination
                filepath = base_filepath.format(*new_combination)
                # Create the directory if it doesn't exist
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                # Build the list of result files and labels
                result_file_list = []
                result_label_list = []
                for explore_value,subconfig_name in explore_subconfigs.items():
                    # Get the result file
                    result_file_list.append(result_filepath + 'raw_csv/' + subconfig_name + '.csv')
                    # Get the label for the plot
                    result_label_list.append(parameter_to_plot_label(explore_param, explore_value))
                # Loop over all plots to draw
                for plot_type in self.plots:
                    if plot_type == 'loss':
                        plot_loss_results(result_file_list, result_label_list, filepath, show_plot=False)
                    elif plot_type == 'accuracy':
                        plot_accuracy_results(result_file_list, result_label_list, filepath, show_plot=False)
                    elif plot_type == 'error_rate':
                        plot_error_rate_results(result_file_list, result_label_list, filepath, show_plot=False)
                    elif plot_type == 'frozen':
                        plot_frozen_evolution(result_file_list, result_label_list, filepath, show_plot=False)
                    
                        
        #####################################
        # Draw the subconfig-specific plots #
        #####################################
        # Loop over the subconfigs and the combinations of varying parameters for each subconfig
        for subconfig, combination in zip(self.subconfigs, self.combinations):
            # Draw every needed plot for each subconfig
            for plot_type in self.plots:
                # Filepath of the result files
                result_filepath = f'runs/results_{self.config_filename}/'
                # Get the plot filepath
                filepath = result_filepath + 'plots/'
                # Add directories for the subconfig varying parameters
                for name in multi_params_names:
                    if name == 'model':
                        filepath += '{}/'
                        continue
                    if name == 'freeze_agent':
                        filepath += '{}={}/'
                        continue
                    filepath += name + '={}/'
                # Format the filepath with the subconfig values
                if 'model' in multi_params_names and 'freeze_agent' in multi_params_names:
                    idx1 = multi_params_names.index('model')
                    idx2 = multi_params_names.index('freeze_agent')
                    new_combination = combination[:idx1] + (combination[idx1][1]['name'],) + combination[idx1+1:idx2] + (combination[idx2][0],combination[idx2][1]['name']) + combination[idx+1:]
                elif 'model' in multi_params_names:
                    idx = multi_params_names.index('model')
                    new_combination = combination[:idx] + (combination[idx][1]['name'],) + combination[idx+1:]
                elif 'freeze_agent' in multi_params_names:
                    idx = multi_params_names.index('freeze_agent')
                    new_combination = combination[:idx] + (combination[idx][0],combination[idx][1]['name']) + combination[idx+1:]
                else:
                    new_combination = combination
                filepath = filepath.format(*new_combination) if new_combination[0] is not None else filepath
                subconfig_name = subconfig['model']['model_args']['name']
                # Create the directory if it doesn't exist
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                if plot_type == "weight_histogram": ### WEIGHT HISTOGRAM ###
                    # Get the model
                    builder = Builder(subconfig)
                    builder.load_model(result_filepath + 'weights/' + subconfig['model']['model_args']['name'] + '.pth')
                    model = builder.model
                    # Plot the weight histogram
                    plot_weight_histogram(model, filepath, show_plot=False)
                elif plot_type == 'loss': ### LOSS ###
                    plot_loss_results([result_filepath + 'raw_csv/' + subconfig['model']['model_args']['name'] + '.csv'], [subconfig['model']['model_args']['name']], filepath, show_plot=False)
                elif plot_type == 'accuracy': ### ACCURACY ###
                    plot_accuracy_results([result_filepath + 'raw_csv/' + subconfig['model']['model_args']['name'] + '.csv'], [subconfig['model']['model_args']['name']], filepath, show_plot=False)
                elif plot_type == 'error_rate': ### ERROR RATE ###
                    plot_error_rate_results([result_filepath + 'raw_csv/' + subconfig['model']['model_args']['name'] + '.csv'], [subconfig['model']['model_args']['name']], filepath, show_plot=False)
                elif plot_type == 'frozen': ### FROZEN ###
                    plot_frozen_evolution([result_filepath + 'raw_csv/' + subconfig['model']['model_args']['name'] + '.csv'], [subconfig['model']['model_args']['name']], filepath, show_plot=False)

                