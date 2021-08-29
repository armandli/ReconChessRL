import torch
from torch.nn.utils import rnn
from torch.utils import data

# RC Action Experience Buffer for Immitation Learning with event sequence
# data

class RCActionEpisode3:
  def __init__(self):
    self.csts = []
    self.acts = []

  def append_st(self, cst, act):
    cst = cst.unsqueeze(0)
    self.csts.append(cst)
    self.acts.append(act)

class RCActionEC3:
  def __init__(self):
    self.csts = []
    self.acts = []

  def append_episode(self, episode):
    csts = torch.cat(episode.csts, dim=0)
    acts = torch.tensor(episode.acts, dtype=torch.long)
    acts = acts.unsqueeze(1)
    self.csts.append(csts)
    self.acts.append(acts)

  def size(self):
    return len(self.acts)

  def to_dataset(self):
    csts = rnn.pad_sequence(self.csts, batch_first=True)
    acts = rnn.pad_sequence(self.acts, batch_first=True)
    return RCActionEB3(csts, acts)

class RCActionEB3(data.Dataset):
  def __init__(self, cst, act):
    self.cur_states = cst
    self.actions    = act

  def __getitem__(self, index):
    cur_state = self.cur_states[index]
    action    = self.actions[index]
    return (cur_state, action)

  def __len__(self):
    return self.actions.shape[0]
