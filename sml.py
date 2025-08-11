import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfSupervisedMemory(nn.Module):
    def __init__(self, channels=64, memory_size=100):
        """
        Self-Supervised Memory Learning Module
        Args:
            channels (int): Number of feature map channels
            memory_size (int): Number of memory slots
        """
        super(SelfSupervisedMemory, self).__init__()

        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, channels), requires_grad=True)  # Memory Bank

        # Feature transformation layers
        self.key_transform = nn.Linear(channels, channels)
        self.value_transform = nn.Linear(channels, channels)

    def forward(self, x):
        """
        Args:
            x: Input feature maps of shape (batch, channels, height, width)
        Returns:
            Refined features after memory retrieval
        """
        batch, channels, height, width = x.shape
        x_reshaped = x.view(batch, channels, -1).permute(0, 2, 1)  # Reshape to (batch, HW, channels)

        # Compute keys & values
        keys = self.key_transform(x_reshaped)  # (batch, HW, channels)
        values = self.value_transform(x_reshaped)

        # Compute similarity with memory
        similarity = torch.matmul(keys, self.memory.T)  # (batch, HW, memory_size)
        attention = F.softmax(similarity, dim=-1)  # Normalize similarity scores

        # Retrieve memory-enhanced features
        memory_out = torch.matmul(attention, self.memory)  # (batch, HW, channels)
        memory_out = memory_out.permute(0, 2, 1).view(batch, channels, height, width)  # Reshape back

        return memory_out + x  # Add residual connection for refinement
