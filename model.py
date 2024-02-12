import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    """
    A simple feedforward neural network for classification.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initializes the neural network with one input layer, two hidden layers, and one output layer.
        
        Parameters:
        - input_size (int): The size of the input feature vector.
        - hidden_size (int): The size of the hidden layers.
        - num_classes (int): The number of output classes (size of the output layer).
        """
        super(NeuralNet, self).__init__()
        # Define the network architecture using nn.Sequential
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input layer to first hidden layer
            nn.ReLU(),  # Activation function between the layers
            nn.Linear(hidden_size, hidden_size),  # First hidden layer to second hidden layer
            nn.ReLU(),  # Activation function for the second hidden layer
            nn.Linear(hidden_size, num_classes)  # Second hidden layer to output layer
        )
    
    def forward(self, x):
        """
        Defines the forward pass through the network.
        
        Parameters:
        - x (Tensor): The input tensor containing the feature data.
        
        Returns:
        - Tensor: The output of the network after passing through the layers.
        """
        return self.network(x)
