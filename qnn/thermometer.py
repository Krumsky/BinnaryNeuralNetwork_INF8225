import torch
from torch import nn

class ThermometerLayer(nn.Module):
    def __init__(self, in_channels=3):
        # Thermometer with a resolution of 8
        thermo = []
        for i in range(32):
            thermo.extend([(i+1)*8 - 1]*in_channels) # Repeat the color channels
        thermo = torch.tensor(thermo, dtype=torch.float32)
        thermo = thermo/255 # tensor with values between 0 and 1 as the input is scaled from [0,255] to [0,1]
        self.thermometer_tensor = nn.Parameter(thermo, requires_grad=False)
        
    def forward(self, x):
        # Repeat the color channels of the input and apply thermometer encoding
        x = x.repeat(1, 32, 1, 1)
        # Encoding
        xsize = x.size()
        thermometer_tensor = self.thermometer_tensor.repeat(xsize[0], xsize[3], xsize[2], 1).transpose(1,3)
        mask = (x >= thermometer_tensor) 
        x = torch.full_like(x, -1.0)
        x[mask] = 1 # Create a binary input with the thermometer encoding
        return x