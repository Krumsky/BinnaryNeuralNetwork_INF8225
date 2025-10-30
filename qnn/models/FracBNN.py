import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T

from qnn.modules import QATLinear, QATConv2d, Binarize, clip  
from qnn.models.basic_models import ModelBase 
from qnn.thermometer import ThermometerLayer


class BPReLU(nn.Module):
    """
    Biased Parametric ReLU activation function.
    alpha and gamma are learnable real biases that translate horizontally and, 
    respectively, vertically the activation function.
    """
    
    def __init__(self, in_channels):
        super(BPReLU, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)  # Horizontal bias
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)  # Vertical bias
        self.prelu = nn.PReLU(in_channels)
        
    def forward(self, x):
        return self.prelu(x + self.alpha.expand_as(x)) - self.gamma.expand_as(x)
    
class SelfCatLayer(nn.Module):
    """
    Concatenation layer that concatenates the input with itself along the channel dimension.
    """
    
    def __init__(self, dim=1):
        super(SelfCatLayer, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return torch.cat((x, x), dim=self.dim)

class FracBNNResidualBlock(nn.Module):
    
    def __init__(self, inp, downsample=False):
        super(FracBNNResidualBlock, self).__init__()
        
        # Parameter initialization
        self.in_channels = inp
        self.out_channels = inp * 2 if downsample else inp # Double the number of channels if downsampling
        self.downsample = downsample
        self.stride = 2 if downsample else 1
        
        # Sign activation function
        self.sign = Binarize()
        
        # 3x3 Convolution block
        self.conv1 = QATConv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bprelu1 = BPReLU(self.out_channels)
        self.bn_out1 = nn.BatchNorm2d(self.out_channels) # Supplementary BN layer used after the residual connection
        
        # Shortcut connection for downsampling (avgpool and concatenate if downsampling, otherwise identity)
        self.shortcut1 = nn.Sequential()
        if self.downsample:
            self.shortcut1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2), # Avg pool to downsample the input for residual connection
                SelfCatLayer(dim=1), # Concatenate the input with itself along the channel dimension
            )
        
        # Second 3x3 Expansion convolution block
        self.conv2 = QATConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.bprelu2 = BPReLU(self.out_channels)
        self.bn_out2 = nn.BatchNorm2d(self.out_channels) # Supplementary BN layer used after the residual connection

    def forward(self, x):
        
        # Keep the input
        identity = x
        # 3x3 Convolution block
        out = self.sign(x)
        out = self.conv1(out) # Downsample if needed
        out = self.bn1(out)
        out = self.bprelu1(out)
        out = out + self.shortcut1(identity) # Residual connection 1
        out = self.bn_out1(out)
        
        # Keep the activation
        identity = out
        # 3x3 Expansion convolution block
        out = self.sign(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.bprelu2(out)
        out = out + identity # Residual connection 2
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
    def __init__(self, in_channels, out_features, input_size=32, thermometer=True, *args, **kwargs):
        super(FracBNNModel, self).__init__()
        
        # Assert input size is divisible by 32, as the model is designed for Imagenet (works with CIFAR-10 and CIFAR-100 too)
        assert input_size % 32 == 0, "Input size must be divisible by 32"
        
        # Resnet-18 architecture (inp, oup, downsample)
        arch = [
            [16, 16, False], # Conv2_1
            [16, 16, False], # Conv2_2
            [16, 16, False], # Conv2_3
            [16, 32, True], # Conv3_1, downsample
            [32, 32, False], # Conv3_2
            [32, 32, False], # Conv3_3
            [32, 64, True], # Conv4_1, downsample
            [64, 64, False], # Conv4_2
            [64, 64, False], # Conv4_3
        ]
        self.classifier_in_features = arch[-1][1] # Number of features after the last conv block
        
        # Entry layer (non binarized)
        self.thermometer = thermometer
        if thermometer:
            # Thermometer encoding layer
            self.thermometer_layer = ThermometerLayer(in_channels=in_channels)
            # Binary entry layer with 8 times more channels
            self.entry = QATConv2d(in_channels*32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.entry = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.entry_BN = nn.BatchNorm2d(16)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for inp, _, downsample in arch:
            self.residual_blocks.append(FracBNNResidualBlock(inp, downsample))
        
        # Classifier
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(self.classifier_in_features, out_features, bias=True)
        
    def forward(self, x):
        if self.thermometer:
            # Apply thermometer encoding if enabled
            x = self.thermometer_layer(x)
        # Entry layer
        x = self.entry(x)
        x = self.entry_BN(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final pooling (if needed) and flattening
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.flatten(x)
        
        # Classifier
        x = self.classifier(x)
        
        return x