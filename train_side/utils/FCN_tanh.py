import torch
import torch.nn as nn

class MultiStepPredictor(nn.Module):
    def __init__(self, horizon, hidden_size=16, num_hidden_layers=2):
        """
        A network that predicts a full trajectory in one shot.
        
        Parameters:
          horizon (int): The prediction horizon (number of time steps).
          hidden_size (int): Number of neurons in the hidden layers.
          num_hidden_layers (int): Number of hidden layers.
        """
        super(MultiStepPredictor, self).__init__()
        self.horizon = horizon
        self.state_dim = 3   # surge, sway, yaw
        self.control_dim = 2 # two thrusters

        # Total input: initial state + control inputs over horizon.
        self.input_size = self.state_dim + horizon * self.control_dim
        
        # Total output: states over horizon.
        self.output_size = horizon * self.state_dim

        layers = []
        layers.append(nn.Linear(self.input_size, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, self.output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
          x (torch.Tensor): Input tensor of shape [batch_size, input_size].
          
        Returns:
          torch.Tensor: Predicted trajectory of shape [batch_size, output_size].
        """
        out = self.network(x)
        # Optionally, you can reshape the output to [batch_size, horizon, state_dim]
        return out.view(-1, self.horizon, self.state_dim)