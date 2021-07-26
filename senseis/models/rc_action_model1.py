import torch
from torch import nn

from senseis.torch_modules.activation import relu_activation
from senseis.torch_modules.residual_layer import ResidualLayer1DV5, ResidualLayer2DV3

# Dueling Q Model
class RCActionModel1(nn.Module):
  def __init__(self, csz, row_sz, col_sz, a_sz):
    super(RCActionModel1, self).__init__()
    self.clayers = nn.Sequential(
        ResidualLayer2DV3(csz, 24, 5, relu_activation, nn.BatchNorm2d),
        ResidualLayer2DV3(24, 48, 3, relu_activation, nn.BatchNorm2d),
        ResidualLayer2DV3(48, 96, 3, relu_activation, nn.BatchNorm2d),
        ResidualLayer2DV3(96, 128, 3, relu_activation, nn.BatchNorm2d),
    )
    self.alayers = nn.Sequential(
        ResidualLayer1DV5(128 * row_sz * col_sz, 8192, relu_activation, nn.LayerNorm),
        nn.Linear(8192, a_sz),
    )
    self.vlayers = nn.Sequential(
        ResidualLayer1DV5(128 * row_sz * col_sz, 4096, relu_activation, nn.LayerNorm),
        nn.Linear(4096, 1),
    )

  def forward(self, x):
    print("gothere80")
    x = self.clayers(x)
    print("gothere81")
    x = self.clayers(x)
    print("gothere82")
    x = torch.flatten(x, start_dim=1)
    print("gothere83")
    v = x
    v = self.vlayers(v)
    print("gothere84")
    a = x
    a = self.alayers(a)
    print("gothere85")
    mean_a = torch.mean(a, dim=1, keepdim=True)
    print("gothere86")
    q = v + (a - mean_a)
    print("gothere87")
    return q
