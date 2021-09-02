import torch
from torch import nn

# RNN Sense Policy Model V1
# input is in the shape (batchsize, sequence_size, features)
class RCSenseModel2(nn.Module):
  def __init__(self, isz, osz, hsz, ssz):
    super(RCSenseModel2, self).__init__()
    self.hidden_size = hsz
    self.rnn = nn.RNN(input_size=isz, hidden_size=hsz, num_layers=ssz, batch_first=True)
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
