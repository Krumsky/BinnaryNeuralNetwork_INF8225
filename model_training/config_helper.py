from itertools import product
from copy import deepcopy

def parameter_to_plot_label(param_name: str, param_value) -> str:
    """
    Converts a parameter name and value to a string label for plotting.
    """
    if param_name == 'model':
        return param_value # Assuming param_value is a string like 'MLP', 'VGG', etc.
    if param_name == 'freeze_agent':
        return param_value # Assuming param_value is a string like 'NullFreezer', 'GradientFreezer', etc.
    if param_name == 'optimizer':
        return param_value # Assuming param_value is a string like 'SGD', 'Adam', etc.
    return f"{param_name}={param_value}" # For other parameters, just return the name and value

def get_multi_params(config: dict) -> tuple:
    # Build an iterator over all the varying parameters
    multi_params = []
    multi_params_names = []
    if isinstance(config['model']['model_args'], list):
        multi_params_names.append('model')
        if isinstance(config['model']['model'], list):
            if len(config['model']['model']) != len(config['model']['model_args']):
                raise ValueError("The number of models and model arguments dictionaries must match.")
            multi_params.append(list(zip(config['model']['model'], config['model']['model_args']))) # [(model1, args1), (model2, args2)]
        else:
            multi_params.append(list(product([config['model']['model']], config['model']['model_args']))) # [(model, args1), (model, args2)]
    if isinstance(config['train']['freeze_args'], list):
        multi_params_names.append('freeze_agent')
        if isinstance(config['train']['freeze_agent'], list):
            if len(config['train']['freeze_agent']) != len(config['train']['freeze_args']):
                raise ValueError("The number of freeze agents and freeze arguments dictionaries must match.")
            multi_params.append(list(zip(config['train']['freeze_agent'], config['train']['freeze_args']))) # [(freeze1, args1), (freeze2, args2)]
        else:
            multi_params.append(list(product([config['train']['freeze_agent']], config['train']['freeze_args']))) # [(freeze, args1), (freeze, args2)]
    if isinstance(config['train']['optimizer'], list):
        multi_params_names.append('optimizer')
        if isinstance(config['train']['optim_args'], list):
            if len(config['train']['optimizer']) != len(config['train']['optim_args']):
                raise ValueError("The number of optimizers and optimizer arguments dictionaries must match.")
            multi_params.append(list(zip(config['train']['optimizer'], config['train']['optim_args'])))
        else:
            multi_params.append(list(product([config['train']['optimizer']], config['train']['optim_args'])))
    if isinstance(config['dataset']['dataset'], list):
        multi_params_names.append('dataset')
        multi_params.append(config['dataset']['dataset'])
    if isinstance(config['dataset']['batch_size'], list):
        multi_params_names.append('batch_size')
        multi_params.append(config['train']['batch_size'])
    if isinstance(config['train']['epochs'], list):
        multi_params_names.append('epochs')
        multi_params.append(config['train']['epochs'])
        
    return multi_params, multi_params_names

def config_splitter(config: dict) -> list:
    """
    Splits the config into subconfigs when parameters are lists (multiple models, datasets, etc.).
    Supported multi-parameters are: model (if model_args is a list, each dictionary goes to the corresponding model), freeze_agent
    (same as model_args, for freeze_args with freeze_agent), dataset, batch_size, epochs
    Args:
        config (dict): The configuration dictionary to split.
    Returns:
        list: A list of subconfigs.
    """
    # Get the multi-parameters
    multi_params, multi_params_names = get_multi_params(config)
    
    # Generate the iterator over the multi-parameters
    iterator = product(*multi_params)
    
    if len(multi_params) == 0:
        # No multi-parameters found, return the original config
        return [config], [(None,)]
    
    # Build the subconfigs
    subconfigs = []
    combinations = []
    for idx,params in enumerate(iterator):
        subconfig = deepcopy(config)
        for param_name, param in zip(multi_params_names, params):
            if param_name == 'model':
                subconfig['model']['model'] = param[0]
                subconfig['model']['model_args'] = deepcopy(param[1]) # Deepcopy because this is a dictionary
            elif param_name == 'freeze_agent':
                subconfig['train']['freeze_agent'] = param[0]
                subconfig['train']['freeze_args'] = deepcopy(param[1]) # Deepcopy because this is a dictionary
            elif param_name == 'optimizer':
                subconfig['train']['optimizer'] = param[0]
                subconfig['train']['optim_args'] = deepcopy(param[1]) # Deepcopy because this is a dictionary
            elif param_name == 'dataset':
                subconfig['dataset']['dataset'] = param
            elif param_name == 'batch_size':
                subconfig['dataset']['batch_size'] = param
            elif param_name == 'epochs':
                subconfig['train']['epochs'] = param
        # Add the config number to the name of the model with a 4 digits format (0000, 0001, etc.)
        subconfig['model']['model_args']['name'] = subconfig['model']['model_args']['name'] + f"_{idx:04d}"
        subconfigs.append(subconfig)
        combinations.append(params)
    return subconfigs, combinations