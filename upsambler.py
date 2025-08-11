import torch
import torch.nn as nn

class SuperResolutionUpsampler(nn.Module):
    def __init__(self, channels=64, upscale_factor=2):
        """
        Upsampling and Reconstruction Module
        Args:
            channels (int): Number of feature map channels.
            upscale_factor (int): Factor by which to upscale the image.
        """
        super(SuperResolutionUpsampler, self).__init__()

        # Sub-Pixel Convolution for Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),  # Rearrange pixels to upscale
            nn.PReLU()  # Activation
        )

        # Final convolution to refine the upscaled image
        self.reconstruction = nn.Conv2d(channels, 3, kernel_size=3, padding=1)  # 3 output channels (RGB)

    def forward(self, x):
        """
        Args:
            x: Feature maps from the Feature Extractor (batch, channels, height, width)
        Returns:
            High-Resolution output image
        """
        out = self.upsample(x)  # Perform upsampling
        out = self.reconstruction(out)  # Final reconstruction step
        return out
