import torch
from torch import nn
from torch.nn import functional as F

from qnn.modules import QATLinear,QATConv2d,Binarize,clip
from qnn.models.basic_models import ModelBase

    
class MLP0(ModelBase):
    
    def __init__(self, in_features, out_features, *args):
        super().__init__()
        self.flatten = nn.Flatten()
        self.output = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.output(x)
        return F.softmax(x,dim=1)
    
class MLP2(ModelBase):
    
    def __init__(self, in_features, out_features, *args):
        super().__init__()
        self.flatten = nn.Flatten()
        self.entry = nn.Linear(in_features, 1024)
        self.entry_BN = nn.BatchNorm1d(1024)
        self.hidden1 = QATLinear(1024, 1024)
        self.hidden1_BN = nn.BatchNorm1d(1024)
        self.hidden2 = QATLinear(1024, 1024)
        self.hidden2_BN = nn.BatchNorm1d(1024)
        self.output = nn.Linear(1024, out_features)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.entry(x)
        x = self.entry_BN(x)
        x = F.relu(x)
        x = self.hidden1(x)
        x = self.hidden1_BN(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.hidden2_BN(x)
        x = F.relu(x)
        x = self.output(x)
        return F.log_softmax(x,dim=1)
    
class MLP(ModelBase):
    
    def __init__(self, in_features, out_features, hidden_dim=1024, binary_weights=False, binary_activations=False, *args, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.entry = QATLinear(in_features, hidden_dim) if binary_weights else nn.Linear(in_features, hidden_dim)
        self.entry_BN = nn.BatchNorm1d(hidden_dim)
        self.hidden1 = QATLinear(hidden_dim, hidden_dim) if binary_weights else nn.Linear(hidden_dim, hidden_dim)
        self.hidden1_BN = nn.BatchNorm1d(hidden_dim)
        self.hidden2 = QATLinear(hidden_dim, hidden_dim) if binary_weights else nn.Linear(hidden_dim, hidden_dim)
        self.hidden2_BN = nn.BatchNorm1d(hidden_dim)
        self.output = QATLinear(hidden_dim, out_features) if binary_weights else nn.Linear(hidden_dim, out_features)
        self.activation_func = Binarize() if binary_activations else nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.entry(x)
        x = self.entry_BN(x)
        x = self.activation_func(x)
        x = self.hidden1(x)
        x = self.hidden1_BN(x)
        x = self.activation_func(x)
        x = self.hidden2(x)
        x = self.hidden2_BN(x)
        x = self.activation_func(x)
        x = self.output(x) # Do not binarize the output
        return x
    
    def get_tensor_list(self):
        return [self.entry.weight, self.hidden1.weight, self.hidden2.weight, self.output.weight]