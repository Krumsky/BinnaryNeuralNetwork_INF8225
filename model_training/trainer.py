import torch
import torch.nn as nn
import torch.optim as optim

# Modified optimizers
from qnn.optimizers.adam import Adam  # Adam with weight-specific learning rates

from utils import accuracy
from qnn.factories import freeze_agent_factory
from qnn.losses import *
from qnn.schedulers import *
from model_training.builder import Builder

from tqdm import tqdm
from csv import DictWriter
from time import time
import os
import json

class Trainer():

    def __init__(self, config, builder: Builder, filepath, device=None):
        self.builder = builder
        self.model = self.builder.model
        self.train_set = self.builder.dataloader['train']
        self.test_set = self.builder.dataloader['test']
        self.valid_set = self.builder.dataloader['valid'] if 'valid' in self.builder.dataloader else None
        
        train_config = config['train']
        
        for attr_name in train_config:
            # set attributes such as epochs, optimizer, device, method
            self.__setattr__(attr_name, train_config[attr_name])

        if device:
            self.device = device
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
        self.epoch = 0

        # Put the model in the correct device
        self.model.to(self.device)

        # Initialize the optimizer
        if self.optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), **self.optim_args)
        elif self.optimizer == "adam":
            self.optimizer = Adam(self.model.parameters(), **self.optim_args)

        # Inititalize the scheduler
        self.lr_scheduler = None
        if self.scheduler == "MultiStepLR":
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.scheduler_args["milestones"], self.scheduler_args["gamma"])
        
        if "fp_pretraining_end" in self.scheduler_args:
            self.lr_scheduler = TwoStepBinaryTraining(self.lr_scheduler, fp_pretraining_end=self.scheduler_args["fp_pretraining_end"], model=self.model)
            
        
        # Initialize the loss function
        if self.loss_fn == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_fn == "cross_entropy_binreg": # cross entropy with regularization forcing weights around -1 or 1
            self.loss_fn = CrossEntropyBinReg(model=self.model, lbda=0.01)
                    
        # Initialize the freeze agent
        freeze_args = self.freeze_args
        freeze_args['trainer'] = self
        freeze_args['model'] = self.model
        self.freeze_agent = freeze_agent_factory(self.freeze_agent, freeze_args)
        if 'GV' in self.freeze_agent.__class__.__name__:
            # The freeze agent works on a weight-specific learning rate, so we need to specify it in the optimizer
            self.optimizer.lr_list = self.freeze_agent.get_scale_list(self.model.parameters())

        self.result_directory = "results_" + filepath.split('.')[0]
        self.result_filepath = "runs/" + self.result_directory
        if self.result_directory not in os.listdir("runs/"):
            os.mkdir(self.result_filepath)
            os.mkdir(self.result_filepath + "/plots")
            os.mkdir(self.result_filepath + "/raw_csv")
            os.mkdir(self.result_filepath + "/weights")
        self.result_file = None

    def train_step(self,epoch):
        self.model.train()
        epoch_loss = 0.
        n = torch.zeros(1, device=self.device)
        self.epoch = epoch
        with tqdm(
            total=len(self.train_set),
            desc="Train Epoch #{}".format(epoch + 1)
        ) as t:
            for idx, (data,target) in enumerate(self.train_set):
                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)

                out = self.model(data)
                loss = self.loss_fn(out, target)
                loss.backward()
                self.freeze_agent.update_mask_step() # Update the freezing mask for the current step
                self.optimizer.step()
                self.freeze_agent.restore_tensors() # Restore the frozen weights

                epoch_loss += loss.item()
                n = idx
                t.set_postfix({
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                t.update()
                
            n += 1
            self.freeze_agent.update_mask_epoch() # Update the freezing mask for the current epoch
            self.freeze_agent.restore_tensors() # Restore the frozen weights
            epoch_loss = epoch_loss/n
            test_loss, acc1, acc5 = self.test()
            if self.lr_scheduler:
                self.lr_scheduler.step()
        return epoch_loss, test_loss, acc1, acc5
    
    def reset_optim(self):
        if isinstance(self.optimizer, optim.SGD):
            # Reset the momentum without reseting the lr_scheduler
            state = self.optimizer.state_dict()
            state['state'] = {}
            self.optimizer.load_state_dict(state)
    
    def train(self):
        self.res_dict = {
                    'epoch': 0,
                    'train_loss': 0,
                    'test_loss': 0,
                    'acc1': 0,
                    'acc5': 0,
                    'epoch_time': 0,
                    'frozen_neurons': 0
                }
        with open(self.result_filepath + f"/raw_csv/{self.builder.model_args['name']}.csv", "w", newline='') as csv:
            self.result_file = DictWriter(csv, fieldnames=['epoch','train_loss','test_loss','acc1','acc5','epoch_time','%freeze'])
            self.result_file.writeheader()
            self.model.train()
            for e in range(self.epochs):
                beg = time()
                loss, test_loss, acc1, acc5 = self.train_step(e)
                end = time()
                self.res_dict = {
                    'epoch': e+1,
                    'train_loss': loss,
                    'test_loss': test_loss,
                    'acc1': acc1.item(),
                    'acc5': acc5.item(),
                    'epoch_time': end-beg,
                    '%freeze': self.freeze_agent.get_pfrozen()
                }
                self.result_file.writerow(self.res_dict)
                print('epoch {}, train loss {}, test loss {}, test acc1 {}, test acc5 {}, %freeze {:.4f}'.format(e+1, loss, test_loss, acc1.item(), acc5.item(), self.freeze_agent.get_pfrozen()))

    def test(self):
        self.model.eval()
        test_loss = 0
        acc1,acc5 = 0,0
        with tqdm(
            total=len(self.test_set),
            desc="Test"
        ) as t:
            with torch.no_grad():
                for data, target in self.test_set:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    haty = self.model(data)

                    test_loss += self.loss_fn(haty, target).item()
                    dacc1,dacc5 = accuracy(haty, target, (1,5))
                    acc1 += dacc1
                    acc5 += dacc5

                    t.update()
                test_loss /= len(self.test_set)
                acc1 /= len(self.test_set)
                acc5 /= len(self.test_set)
        return test_loss, acc1, acc5
                                
    def save(self):
        torch.save(self.model.state_dict(), self.result_filepath + f"/weights/{self.builder.model_args['name']}.pth")
