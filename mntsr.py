import torch
import torch.nn as nn
from feature_extractor import FeatureExtractor
from upsambler import SuperResolutionUpsampler

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, C)

        attn_output, _ = self.mha(x, x, x)  # Self-Attention
        attn_output = self.norm(attn_output + x)  # Add & Normalize

        return attn_output.permute(0, 2, 1).view(b, c, h, w)

class MntsrModel(nn.Module):
    def __init__(self, in_channels=3, channels=64, num_srb=3, upscale_factor=2):
        super(MntsrModel, self).__init__()
        
        # Initial Convolution
        self.initial_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        # Feature Extraction with Self-Attention
        self.feature_extractor = FeatureExtractor(channels, num_srb)

        # Super-Resolution Upsampling & Reconstruction
        self.upsampler = SuperResolutionUpsampler(channels, upscale_factor)

    def forward(self, x):
        x = self.initial_conv(x)   # Convert input to feature maps
        features = self.feature_extractor(x)  # Extract features with Self-Attention
        hr_image = self.upsampler(features)   # Upscale & reconstruct
        return hr_image
