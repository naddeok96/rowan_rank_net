# Imports
import torch.nn as nn

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) with one hidden layer.

    Args:
        input_size (int): Size of the input tensor.
        hidden_size (int, optional): Number of neurons in the hidden layer (default: 10).
        output_size (int, optional): Size of the output tensor (default: 1).
    """
    def __init__(self, input_size=10, hidden_size=10, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out) * 100.0 # Bound output between 0 and 100
        return out
