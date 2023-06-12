import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters_requiring_grad(layer):
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)

def count_parameters(layer):
    return sum(p.numel() for p in layer.parameters())

def BatchNorm1D_controlled_bias(num_features, bias=True):
    bn = torch.nn.BatchNorm1d(num_features)  # Initialize BatchNorm layer
    if not bias:
        bn.bias.data.zero_()  # Zero out the bias
        bn.bias.requires_grad = False  # Set bias to not require gradients
    return bn