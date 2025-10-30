from tomlkit.toml_file import TOMLFile
from model_training.builder import Builder
from model_training.trainer import Trainer
from model_training.results.plotter import Plotter
from model_training.config_helper import config_splitter
import argparse
from time import sleep

parser = argparse.ArgumentParser(
    prog="Main",
    description="Run a training following the specified config file")

parser.add_argument('configfile')
parser.add_argument('--gpu',
                    type=str,
                    default='cuda:0',
                    help="the cuda identifier of the gpu to run the config on")
parser.add_argument('--plot_only',
                    type=bool,
                    default=False,
                    help="if True, only plot the results of the training, without re-running it")

try:
    args = parser.parse_args()

    filepath = args.configfile
except:
    filepath = 'config/config.toml'
    
try:
    device = args.gpu
except:
    device = None

try:
    config = TOMLFile(filepath).read()
except:
    print("config file not found, exitting...")
    exit()

config = config.unwrap()
filepath = filepath.split('/')[-1] # name of the config file
if device:
    config['model']['device'] = device # change the selected device

if not args.plot_only:
    subconfigs, _ = config_splitter(config) # split the config into subconfigs
    idx = 0
    for subconfig in subconfigs:
        if idx == 110:
            idx += 1
            continue
        idx += 1
        print(f"""
            #######################################################
            #        Starting training with subconfig {subconfig['model']['model_args']['name'].split('_')[-1]}        #
            #######################################################
            """)
        t = Trainer(subconfig, Builder(subconfig), filepath, device)
        t.train()
        t.save()

p = Plotter(config, filepath)
p.plot()