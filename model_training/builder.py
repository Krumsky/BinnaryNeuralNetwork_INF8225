import torch
from torch.utils.data import random_split,DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision


from qnn.factories import model_factory


class Builder():

    def __init__(self, config):
        
        model_config = config['model']
        dataset_config = config['dataset']
        
        for attr_name in model_config:
            # set model attributes
            self.__setattr__(attr_name, model_config[attr_name])

        for attr_name in dataset_config:
            # set dataset attributes
            self.__setattr__(attr_name, dataset_config[attr_name])
        
        if "thermometer" not in self.model_args:
            self.model_args["thermometer"] = False

        # Set the RNG seed
        torch.manual_seed(self.seed)
        
        
        #Initialize the dataset
        self.dataset_name = self.dataset
        self.input_size = None
        if self.dataset == 'mnist':
            self.input_size = 28
            # Transform
            if self.model_args["thermometer"]:
                train_transform = T.Compose([
                T.ToTensor()
                ])
            else:
                train_transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(0, 1)
                ])
            test_transform = train_transform
            self.dataset = {
                'train': datasets.MNIST("./datasets/MNIST", train=True, transform=train_transform, download=True),
                'test' : datasets.MNIST("./datasets/MNIST", train=False, transform=test_transform, download=True)
                }
            # Set the model arguments
            self.model_args["out_features"] = 10
            self.model_args["in_features"] = 28*28
            self.model_args["in_channels"] = 1
            self.model_args["input_size"] = 28
        # MNIST Fashion
        elif self.dataset == 'mnist_fashion':
            self.input_size = 28
            # Transform
            if self.model_args["thermometer"]:
                train_transform = T.Compose([
                    T.ToTensor()
                ])
            else:
                train_transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(0.5, 0.5)
                ])
            test_transform = train_transform
            self.dataset = {
                'train': datasets.FashionMNIST("./datasets/MNISTF", train=True, transform=train_transform, download=True),
                'test' : datasets.FashionMNIST("./datasets/MNISTF", train=False, transform=test_transform, download=True)
                }
            # Set the model arguments
            self.model_args["out_features"] = 10
            self.model_args["in_features"] = 28*28
            self.model_args["in_channels"] = 1
            self.model_args["input_size"] = 28
        # CIFAR10
        elif self.dataset == 'c10':
            self.input_size = 32
            # Transform
            if self.model_args["thermometer"]:
                train_transform = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, 4), 
                    T.ToTensor()
                ])
                test_transform = T.Compose([
                    T.ToTensor()
                ])
            else:
                train_transform = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, 4), 
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
                test_transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            self.dataset = {
                'train': datasets.CIFAR10("./datasets/CIFAR10", train=True, transform=train_transform, download=True),
                'test': datasets.CIFAR10("./datasets/CIFAR10", train=False, transform=test_transform, download=True)
            }
            # Set the model arguments
            self.model_args["out_features"] = 10
            self.model_args["in_channels"] = 3
            self.model_args["input_size"] = 32
        # CIFAR100
        elif self.dataset == 'c100':
            self.input_size = 32
            # Transform
            if self.model_args["thermometer"]:
                train_transform = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, 4),
                    T.ToTensor()
                ])
                test_transform = T.Compose([
                    T.ToTensor()
                ])
            else:
                train_transform = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, 4),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
                test_transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            self.dataset = {
                'train': datasets.CIFAR100("./datasets/CIFAR100", train=True, transform=train_transform, download=True),
                'test': datasets.CIFAR100("./datasets/CIFAR100", train=False, transform=test_transform, download=True)
            }
            # Set the model arguments
            self.model_args["out_features"] = 100
            self.model_args["in_channels"] = 3
            self.model_args["input_size"] = 32
            
        self.n_classes = len(self.dataset['train'].classes)
        self.classes = self.dataset['train'].classes
            
        self.dataloader = {
            'train': DataLoader(dataset=self.dataset['train'], batch_size=self.batch_size, shuffle=True, num_workers=16),
            'test': DataLoader(dataset=self.dataset['test'], batch_size=self.batch_size, shuffle=False, num_workers=16)
        }
        
        # Initialize the model
        self.model_name = self.model
        self.model = model_factory(self.model_name, self.model_args)
        if self.weights and self.weights != "None":
            # Load the model weights if specified
            print("Weights found")
            self.load_model(self.weights)
        

    def load_model(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path))