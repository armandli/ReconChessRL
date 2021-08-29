import torch
from torch import nn

# RNN Action Policy Model V1
class RCActionModel4(nn.Module):
  def __init__(self, isz, osz, hsz, ssz):
    super(RCActionModel4, self).__init__()
    self.hidden_size = hsz
    self.rnn = nn.RNN(isz, hsz, ssz, batch_first=True)
    self.flayers = nn.Sequential(
        nn.Linear(hsz, osz),
        nn.Softmax(dim=2),
    )

  def forward(self, x, h):
    o, h = self.rnn(x, h)
    o = self.flayers(o)
    return o, h

  def init(self, batch_size):
    return torch.zeros(1, batch_size, self.hidden_size)
