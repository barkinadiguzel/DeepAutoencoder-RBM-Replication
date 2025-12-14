import torch.nn as nn
from src.layers.encoder_layer import EncoderLayer
from src.layers.code_layer import CodeLayer

class EncoderNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(EncoderNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(EncoderLayer(layer_sizes[i], layer_sizes[i+1]))
        self.encoder = nn.Sequential(*layers)
        self.code_layer = CodeLayer(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        x = self.encoder(x)
        x = self.code_layer(x)
        return x
