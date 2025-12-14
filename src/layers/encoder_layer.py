import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Sigmoid()  # The article uses a logistic unit.

    def forward(self, x):
        return self.activation(self.linear(x))
