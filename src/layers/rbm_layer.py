import torch
import torch.nn as nn
import torch.nn.functional as F

class RBMLayer(nn.Module):
    def __init__(self, visible_units, hidden_units, k=1):
        super(RBMLayer, self).__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k  

        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))

    def sample_h(self, v):
        # P(h=1 | v)
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        h_sample = torch.bernoulli(p_h)
        return p_h, h_sample

    def sample_v(self, h):
        # P(v=1 | h)
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        v_sample = torch.bernoulli(p_v)
        return p_v, v_sample

    def forward(self, v):
        v_sample = v
        for _ in range(self.k):
            _, h_sample = self.sample_h(v_sample)
            _, v_sample = self.sample_v(h_sample)
        return v_sample

    def contrastive_divergence(self, v, lr=0.01):
        # Hidden sample from data
        p_h0, h0 = self.sample_h(v)

        # Gibbs sampling
        v_k = v
        for _ in range(self.k):
            p_hk, h_k = self.sample_h(v_k)
            p_vk, v_k = self.sample_v(h_k)

        # Weights and biases update
        self.W.data += lr * (torch.matmul(v.t(), p_h0) - torch.matmul(v_k.t(), p_hk)) / v.size(0)
        self.v_bias.data += lr * torch.mean(v - v_k, dim=0)
        self.h_bias.data += lr * torch.mean(p_h0 - p_hk, dim=0)

        return v_k
