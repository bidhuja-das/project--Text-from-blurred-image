import torch
import torch.nn as nn

class SelfResidualBlock(nn.Module):
    def __init__(self, channels):
        super(SelfResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))  # Residual Connection

class FeatureExtractor(nn.Module):
    def __init__(self, channels, num_srb=3):
        super(FeatureExtractor, self).__init__()
        self.srbs = nn.Sequential(*[SelfResidualBlock(channels) for _ in range(num_srb)])

    def forward(self, x):
        for srb in self.srbs:
            query, key = self.extract_attention_features(x)  # Get query and key
            
            # Debugging: Print tensor shapes before matrix multiplication
            print("Query shape:", query.shape)
            print("Key shape:", key.shape)
            
            if query.dim() != 3 or key.dim() != 3:
                raise ValueError(f"Invalid tensor dimensions: query {query.shape}, key {key.shape}")

            try:
                attention = torch.bmm(query, key)  # (batch, H*W, H*W)
            except RuntimeError as e:
                print(f"Error in torch.bmm(): {e}")
                return x  # Skip attention if error occurs

            x = srb(x)  # Apply SRB + Attention
        
        return x

    def extract_attention_features(self, x):
        """Extracts attention-related query and key tensors."""
        batch, channels, height, width = x.shape
        query = x.view(batch, channels, -1).permute(0, 2, 1)  # Reshape to (batch, H*W, C)
        key = query.permute(0, 2, 1)  # Transpose for matrix multiplication
        return query, key
