from src.layers.rbm_layer import RBMLayer

class RBMStack:
    def __init__(self, layer_sizes, k=1):
        self.rbms = []
        for i in range(len(layer_sizes) - 1):
            self.rbms.append(RBMLayer(layer_sizes[i], layer_sizes[i+1], k))

    def pretrain(self, data, lr=0.01):
        input_data = data
        for rbm in self.rbms:
            rbm.contrastive_divergence(input_data, lr)
            input_data = rbm.forward(input_data)
        return self.rbms
