from senseis.torch_modules.residual_layer import downsampling2DV2
import torch
from torch import nn
from senseis.torch_modules.residual_layer import downsampling2DV2
from senseis.torch_modules.torus_conv import TorusConv2DV1

class TorusResidualLayer2DV1(nn.Module):
  def __init__(self, in_c, out_c, ksz, act_layer, norm_layer, stride=1):
    super(TorusResidualLayer2DV1, self).__init__()
    self.c1 = TorusConv2DV1(in_c, out_c, ksz, stride=stride)
    self.c2 = TorusConv2DV1(out_c, out_c, ksz, stride=stride)
    self.a1 = act_layer()
    self.a2 = act_layer()
    self.b1 = norm_layer(in_c)
    self.b2 = norm_layer(out_c)
    self.downsample = None
    if in_c != out_c or stride > 1:
      self.downsample = downsampling2DV2(in_c, out_c, stride, norm_layer)
    self.esz = ksz // 2

  def forward(self, x):
    s = x
    x = self.b1(x)
    x = self.a1(x)
    x = self.c1(x)
    x = self.b2(x)
    x = self.a2(x)
    x = self.c2(x)
    if self.downsample is not None:
      s = self.downsample(s)
    x = x + s
    return x