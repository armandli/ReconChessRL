import torch
from torch.nn.utils import rnn
from torch.utils import data

class RCSenseEpisode2:
  def __init__(self):
    self.csts = []
    self.acts = []
    self.rewards = []

  def append_st(self, cst, act):
    cst = cst.unsqueeze(0)
    self.csts.append(cst)
    self.acts.append(act)

  def append_post(self, reward):
    self.rewards.append(reward)

class RCSenseEC2:
  def __init__(self):
    self.csts = []
    self.acts = []
    self.rewards = []

  def append_episode(self, episode):
    csts = torch.cat(episode.csts, dim=0)
    acts = torch.tensor(episode.acts, dtype=torch.long)
    rewards = torch.tensor(episode.rewards, dtype=torch.float32)
    acts = acts.unsqueeze(1)
    rewards = rewards.unsqueeze(1)
    self.csts.append(csts)
    self.acts.append(acts)
    self.rewards.append(rewards)

  def size(self):
    # only count number of episodes
    return len(self.acts)

  def to_dataset(self):
    # dim = (batch, sequence, x)
    csts = rnn.pad_sequence(self.csts, batch_first=True)
    # dim = (batch, sequence, 1)
    acts = rnn.pad_sequence(self.acts, batch_first=True)
    rewards = rnn.pad_sequence(self.rewards, batch_first=True)
    return RCSenseEB2(csts, acts, rewards)

class RCSenseEB2(data.Dataset):
  def __init__(self, csts, acts, rewards):
    self.cur_states = csts
    self.actions = acts
    self.rewards = rewards

  def __getitem__(self, index):
    cur_state = self.cur_states[index]
    action    = self.actions[index]
    reward    = self.rewards[index]
    return (cur_state, action, reward)

  def __len__(self):
    return self.actions.shape[0]
