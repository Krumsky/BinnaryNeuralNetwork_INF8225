import torch
from torch import nn
from torch.nn import functional as F

from qnn.modules import QATLinear, QATConv2d, Binarize, clip  
from qnn.models.basic_models import ModelBase 


class BPReLU(nn.Module):
    """
    Biased Parametric ReLU activation function.
    alpha and gamma are learnable real biases that translate horizontally and, 
    respectively, vertically the activation function.
    """
    
    def __init__(self, in_channels):
        super(BPReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(in_channels))
        self.gamma = nn.Parameter(torch.Tensor(in_channels))
        self.prelu = nn.PReLU()
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.alpha, 0.0, 1.0)
        nn.init.uniform_(self.gamma, 0.0, 1.0)
        
    def forward(self, x):
        return (self.prelu(x.transpose(1,-1) + self.alpha) - self.gamma).transpose(1,-1)
    

class FracBNNResidualBlock(nn.Module):
    
    def __init__(self, inp, downsample=False):
        super(FracBNNResidualBlock, self).__init__()
        
        # Parameter initialization
        self.in_channels = inp
        self.out_channels = inp * 2 if downsample else inp
        self.downsample = downsample
        self.stride = 2 if downsample else 1
        
        # 3x3 Convolution block
        self.sign = Binarize()
        self.conv1 = QATConv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1) # Downsampling layer (if downsample there is)
        self.bn1 = nn.BatchNorm2d(inp)
        self.bprelu1 = BPReLU(inp)
        self.bn_out1 = nn.BatchNorm2d(inp) # Supplementary BN layer used after the residual connection
        self.downsample_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2) if downsample else None # Avg pool to downsample the input for residual connection
        
        # 1x1 Expansion convolution block
        self.conv2 = QATConv2d(inp, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.bprelu2 = BPReLU(self.out_channels)
        self.bn_out2 = nn.BatchNorm2d(self.out_channels) # Supplementary BN layer used after the residual connection

    def forward(self, x):
        identity = x
        
        # 3x3 Convolution block
        out = self.sign(x)
        out = self.conv1(out) # Downsample if needed
        out = self.bn1(out)
        out = self.bprelu1(out)
        out = out + (self.downsample_avg_pool(identity) if self.downsample else identity) # Residual connection 1
        out = self.bn_out1(out)
        
        identity = out
        
        # 1x1 Expansion convolution block
        out = self.sign(out)
        out = self.conv2(out) # If there is downsampling, double the number of channels
        out = self.bn2(out)
        out = self.bprelu2(out)
        out = out + (torch.cat([identity, identity], dim=1) if self.downsample else identity) # Residual connection 2
        out = self.bn_out2(out)
        
        return out

    def get_qat_weights(self):
        """ Helper to get weights from QATConv2d layers within this block """
        # Weights are already collected during init
        return [self.conv1.weight, self.conv2.weight]
    
class FracBNNModel(ModelBase):
    """
    Resnet-18 architecture with FracBNN blocks.
    """
    def __init__(self, in_channels, out_features, input_size=32, thermometer_embedding=True, *args, **kwargs):
        super(FracBNNModel, self).__init__()
        
        # Assert input size is divisible by 32, as the model is designed for Imagenet (works with CIFAR-10 and CIFAR-100 too)
        assert input_size % 32 == 0, "Input size must be divisible by 32"
        
        # Resnet-18 architecture (inp, oup, downsample)
        arch = [
            [64, 64, False], # Conv2_1
            [64, 64, False], # Conv2_2
            [64, 128, True], # Conv3_1, downsample
            [128, 128, False], # Conv3_2
            [128, 256, True], # Conv4_1, downsample
            [256, 256, False], # Conv4_2
            [256, 512, True], # Conv5_1, downsample
            [512, 512, False] # Conv5_2
        ]
        self.classifier_in_features = arch[-1][1] * ((input_size // 32) ** 2) # Number of features after the last conv block
        
        # Entry layer (non binarized)
        self.thermometer_embedding = thermometer_embedding
        if thermometer_embedding:
            self.entry = QATConv2d(in_channels*8, 64, kernel_size=7, stride=2, padding=3, bias=True)
        else:
            self.entry = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.entry_BN = nn.BatchNorm2d(64)
        self.entry_act = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for inp, _, downsample in arch:
            self.residual_blocks.append(FracBNNResidualBlock(inp, downsample))
        
        # Classifier
        self.final_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(self.classifier_in_features, out_features, bias=True)
        
    def forward(self, x):
        # Entry layer
        x = self.entry(x)
        x = self.entry_BN(x)
        x = self.entry_act(x)
        x = self.max_pool(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Classifier
        x = self.final_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x