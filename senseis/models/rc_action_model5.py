import torch

from torch import nn

class RCActionModel5(nn.Module):
  def __init__(self, isz, osz, hsz, ssz):
    super(RCActionModel5, self).__init__()
    self.hidden_size = hsz
    self.gru = nn.GRU(input_size=isz, hidden_size=hsz, num_layers=ssz, batch_first=True)
    self.flayers = nn.Sequential(
        nn.Linear(hsz, osz),
        nn.Softmax(dim=2),
    )

  def forward(self, x, h):
    o, h = self.gru(x, h)
    o = self.flayers(o)
    return o, h

  def init(self, batch_size):
    return torch.zeros(1, batch_size, self.hidden_size)
