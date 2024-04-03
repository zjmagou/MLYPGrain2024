#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 22:47:46 2023

@author: maeko
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Define the CNN Model
class SpatialTemporalCNN(nn.Module):
    def __init__(self):
        super(SpatialTemporalCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # ConvLSTM layer
        self.conv_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        # Dense layers for value branch
        self.fc1_value = nn.Linear(in_features=128, out_features=64)
        self.fc2_value = nn.Linear(in_features=64, out_features=1)  # Adjust output features based on your time series

        # Dense layers for deviation branch
        self.fc1_deviation = nn.Linear(in_features=128, out_features=64)
        self.fc2_deviation = nn.Linear(in_features=64, out_features=1)  # Adjust output features based on your time series

    def forward(self, x):
        # Apply convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for LSTM
        x = x.view(x.size(0), -1)  # Flatten the tensor for LSTM layer

        # Apply ConvLSTM
        x, (h_n, c_n) = self.conv_lstm(x)

        # Split into two branches
        # Value branch
        value = F.relu(self.fc1_value(x))
        value = self.fc2_value(value)

        # Deviation branch
        deviation = F.relu(self.fc1_deviation(x))
        deviation = self.fc2_deviation(deviation)

        return value, deviation

# Instantiate the model
model = SpatialTemporalCNN()

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Use Mean Squared Error Loss for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example input (adjust the size according to your data)
example_input = torch.randn(1, 1, 64, 64)  # Batch size, Channels, Height, Width
example_input = Variable(example_input)

# Forward pass
value_output, deviation_output = model(example_input)
print(value_output, deviation_output)
