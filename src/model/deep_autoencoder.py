import torch
import torch.nn as nn
from src.layers.encoder_layer import EncoderLayer
from src.layers.decoder_layer import DecoderLayer
from src.layers.code_layer import CodeLayer
from src.modules.rbm_stack import RBMStack

class DeepAutoencoder(nn.Module):
    def __init__(self, layer_sizes, code_size, k=1):
        super(DeepAutoencoder, self).__init__()

        # --- Encoder ---
        self.enc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.enc_layers.append(EncoderLayer(layer_sizes[i], layer_sizes[i+1]))

        # --- Code Layer ---
        self.code_layer = CodeLayer(layer_sizes[-1], code_size)

        # --- Decoder ---
        self.dec_layers = nn.ModuleList()
        reversed_layers = layer_sizes[::-1]
        for i in range(len(reversed_layers) - 1):
            self.dec_layers.append(DecoderLayer(reversed_layers[i], reversed_layers[i+1]))

        # --- Optional RBM stack for pretraining ---
        self.rbm_stack = RBMStack(layer_sizes + [code_size], k=k)

    def forward(self, x):
        # Encoder forward
        for enc in self.enc_layers:
            x = enc(x)

        # Code layer
        x = self.code_layer(x)

        # Decoder forward
        for dec in self.dec_layers:
            x = dec(x)

        return x

    def pretrain(self, data, lr=0.01):
        self.rbm_stack.pretrain(data, lr)
