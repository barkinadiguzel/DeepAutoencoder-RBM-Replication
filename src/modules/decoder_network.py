import torch.nn as nn
from src.layers.decoder_layer import DecoderLayer
from src.layers.code_layer import CodeLayer

class DecoderNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(DecoderNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1, 1, -1):
            layers.append(DecoderLayer(layer_sizes[i], layer_sizes[i-1]))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder(x)
        return x
