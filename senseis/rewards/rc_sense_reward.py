import torch

def rc_sense_reward1(oma, omb):
  v = torch.sum(torch.abs(oma - omb))
  return v
