import torch.nn as nn

def relu_activation():
  return nn.ReLU(inplace=True)

def silu_activation():
  return nn.SiLU(inplace=True)