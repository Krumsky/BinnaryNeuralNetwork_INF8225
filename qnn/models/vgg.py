import torch
from torch import nn
from torch.nn import functional as F

from qnn.modules import *
from qnn.models.basic_models import ModelBase


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, binarize_weight=True, binarize_activation=True):
        super(VGGBlock, self).__init__()
        # Convolutional layer
        if binarize_weight:
            self.conv = QATConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        # Activation function
        if binarize_activation:
            self.act = Binarize()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class VGG(ModelBase):
    """
    VGG model with binarized weights and activations. Inspired from the VGG-16 architecture A.
    The exact architecture is from the papers :
    - "BinaryConnect: Training Deep Neural Networks with binary weights during propagations",
    Courbariaux, M., Bengio, Y., & David, J. P. (2015). Binaryconnect: Training deep neural networks with binary weights during propagations. 
     
    - "Binarized Neural Networks: Training Neural Networks with Weights and  Activations Constrained to +1 or −1", 
    Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). Binarized neural networks: Training deep neural 
    networks with weights and activations constrained to+ 1 or-1.
    """
    
    def __init__(self, in_channels, out_features, input_size=32, binary_weights=True, binary_activations=True, *args, **kwargs):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(in_channels, 128, kernel_size=3, stride=1, padding=(32-input_size+1), binarize_weight=binary_weights, binarize_activation=binary_activations),
            VGGBlock(128, 128, kernel_size=3, stride=1, padding=1, binarize_weight=binary_weights, binarize_activation=binary_activations),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(128, 256, kernel_size=3, stride=1, padding=1, binarize_weight=binary_weights, binarize_activation=binary_activations),
            VGGBlock(256, 256, kernel_size=3, stride=1, padding=1, binarize_weight=binary_weights, binarize_activation=binary_activations),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(256, 512, kernel_size=3, stride=1, padding=1, binarize_weight=binary_weights, binarize_activation=binary_activations),
            VGGBlock(512, 512, kernel_size=3, stride=1, padding=1, binarize_weight=binary_weights, binarize_activation=binary_activations),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QATLinear(8192, 1024) if binary_weights else nn.Linear(8192, 1024),
            nn.BatchNorm1d(1024),
            Binarize() if binary_activations else nn.ReLU(),
            nn.Identity() if binary_activations or binary_weights else nn.Dropout(0.5), # If we binarize activations, we don't need dropout (as the binarization is already a form of regularization)
            QATLinear(1024, 1024) if binary_weights else nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            Binarize() if binary_activations else nn.ReLU(),
            nn.Identity() if binary_activations or binary_weights else nn.Dropout(0.5), # If we binarize activations, we don't need dropout (as the binarization is already a form of regularization)
            QATLinear(1024, out_features) if binary_weights else nn.Linear(1024, out_features)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class BnVGG(ModelBase):
    
    def __init__(self, in_channels, out_features, input_size=32, *args, **kwargs):
        super(BnVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(3,3), stride=1, padding=(32-input_size+1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, out_features)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class BnVGGNoReg(ModelBase):
    
    def __init__(self, in_channels, out_features, input_size=32, *args):
        super(BnVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(3,3), stride=1, padding=(32-input_size+1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, out_features)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class BVGG(ModelBase):
    
    def __init__(self, in_channels, out_features, input_size=32, *args):
        super(BVGG, self).__init__()
        self.features = nn.Sequential(
            QATConv2d(in_channels=in_channels, out_channels=128, kernel_size=(3,3), stride=1, padding=(32-input_size+1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            QATConv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            QATConv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            QATConv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            QATConv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QATConv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QATLinear(8192, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            QATLinear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            QATLinear(1024, out_features)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_tensor_list(self):
        tensor_list = []
        binarized_list = [0,3,7,10,14,17]
        for idx in binarized_list:
            tensor_list.append(self.features[idx].weight)
        binarized_list = [1,4,7]
        for idx in binarized_list:
            tensor_list.append(self.classifier[idx].weight)
        return tensor_list