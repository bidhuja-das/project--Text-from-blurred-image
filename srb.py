import torch
import torch.nn as nn

class SpatialRefinementBlock(nn.Module):
    def __init__(self, channels=64):
        super(SpatialRefinementBlock, self).__init__()
        
        # Convolution layers for feature extraction
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Activation function (Parametric ReLU)
        self.prelu = nn.PReLU()
        
        # Pixel Rectification Module (PRM)
        self.prm = nn.Conv2d(channels, channels, kernel_size=1)  # 1x1 convolution for weighting

    def forward(self, x):
        identity = x  # Save input for residual connection

        # First convolution + activation
        out = self.conv1(x)
        out = self.prelu(out)

        # Second convolution
        out = self.conv2(out)

        # Pixel Rectification Module (PRM)
        weights = torch.sigmoid(self.prm(out))  # Compute pixel-wise importance
        out = out * weights  # Apply weighting

        # Add residual connection
        out = out + identity  
        return out
