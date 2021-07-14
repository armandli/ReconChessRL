import torch
from torch import nn

class TorusConv2DV1(nn.Module):
  def __init__(self, in_c, out_c, ksz, stride=1):
    super(TorusConv2DV1, self).__init__()
    self.conv = nn.Conv2d(in_c, out_c, ksz, stride=stride, padding=0, bias=False)
    self.esz = ksz // 2

  def forward(self, x):
    x = torch.cat([x[:,:,:,-self.esz:], x, x[:,:,:,:self.esz]], dim=3)
    x = torch.cat([x[:,:,-self.esz:,:], x, x[:,:,:self.esz,:]], dim=2)
    x = self.conv(x)
    return x