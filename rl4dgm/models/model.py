import copy

import torch
from torch import nn, vmap
from torch.func import stack_module_state, functional_call
import torch.nn.functional as F

import numpy as np

# from torchensemble import VotingRegressor

class CNNModel(nn.Module):
    def __init__(
            self,
            channels=4,
            size=64,
            device=None,
            model_initialization_weight:float=None,
            model_initialization_seed=None,
        ):
        """
        Series of convolutional layers with ReLU activations        
        """
        super(CNNModel, self).__init__()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.size = size
        self.channels = channels
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=3, stride=1, padding=1).to(device)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=channels*2, out_channels=channels*4, kernel_size=3, stride=1, padding=1).to(device)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=channels*4, out_channels=channels*8, kernel_size=3, stride=1, padding=1).to(device)
        
        # Fully connected layer
        self.fc1 = nn.Linear(size * size * channels // 8, 2048).to(device)

        if model_initialization_seed is not None:
            torch.manual_seed(model_initialization_seed)


        # initialize model weights
        if model_initialization_weight is not None:
            weight_scale = model_initialization_weight
            for layer in self.model.children():
                if isinstance(layer, nn.Linear):
                    layer.weight.data = torch.clip(weight_scale * torch.ones_like(layer.weight), -0.3, 0.3)
                    weight_scale *= -1.1

        self.float()

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer followed by ReLU activation and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer followed by ReLU activation and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, self.size * self.size * self.channels // 8)
        
        # Apply the fully connected layer
        x = F.relu(self.fc1(x))
        
        return x


class LinearModel(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims=[16384]*6,
            output_dim=2048,
            device=None,
            model_initialization_weight:float=None,
            model_initialization_seed=None,
        ):
        """
        Series of linear layers with ReLU activations        
        """
        super(LinearModel, self).__init__()

        if model_initialization_seed is not None:
            torch.manual_seed(model_initialization_seed)
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # input layer
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        ] 
        
        # hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers).to(device)

        # initialize model weights
        if model_initialization_weight is not None:
            weight_scale = model_initialization_weight
            for layer in self.model.children():
                if isinstance(layer, nn.Linear):
                    layer.weight.data = torch.clip(weight_scale * torch.ones_like(layer.weight), -0.3, 0.3)
                    weight_scale *= -1.1

        self.float()

    def forward(self, x):
        latents = self.model(x)
        return latents