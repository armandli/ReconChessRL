import torch
from torch import nn

from senseis.torch_modules.activation import relu_activation
from senseis.torch_modules.residual_layer import ResidualLayer1DV5, ResidualLayer2DV3

# Sense Policy Model
class RCSenseModel1(nn.Module):
  def __init__(self, csz, row_sz, col_sz, a_sz):
    super(RCSenseModel1, self).__init__()
    self.clayers = nn.Sequential(
        ResidualLayer2DV3(csz, 24,  3, relu_activation, nn.BatchNorm2d),
        ResidualLayer2DV3(24,  48,  3, relu_activation, nn.BatchNorm2d),
    )
    self.flayers = nn.Sequential(
        ResidualLayer1DV5(48 * row_sz * col_sz, 2048, relu_activation, nn.LayerNorm),
        ResidualLayer1DV5(2048, 512, relu_activation, nn.LayerNorm),
        nn.Linear(512, a_sz),
        nn.Softmax(dim=1),
    )

  def forward(self, x):
    x = self.clayers(x)
    x = torch.flatten(x, start_dim=1)
    x = self.flayers(x)
    return x
