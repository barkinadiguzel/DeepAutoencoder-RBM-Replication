import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DecoderLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Sigmoid()  # Logistic output

    def forward(self, x):
        return self.activation(self.linear(x))
