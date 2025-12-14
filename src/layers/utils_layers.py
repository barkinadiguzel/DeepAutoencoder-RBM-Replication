import torch

def flatten(x):
    return x.view(x.size(0), -1)

def reshape(x, shape):
    return x.view(shape)
