import torch
import torch.nn as nn

class CodeLayer(nn.Module):
    def __init__(self, input_size, code_size):
        super(CodeLayer, self).__init__()
        self.linear = nn.Linear(input_size, code_size)
        self.activation = nn.Identity() # I put this here because there is no activation here

    def forward(self, x):
        return self.activation(self.linear(x))
