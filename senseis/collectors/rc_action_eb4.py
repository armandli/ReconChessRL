import torch
from torch.nn.utils import rnn
from torch.utils import data

# RC Action Experience Buffer based on event sequence and RL

class RCActionEpisode4:
  def __init__(self):
    self.csts = []
    self.acts = []
    self.reward = []

  def append_st(self, cst, act):
    cst = cst.unsqueeze(0)
    self.csts.append(act)
    self.acts.append(act)

  def append_post(self, reward):
    self.rewards.append(reward)

  def append_terminal(self, final_reward):
    #TODO: reguarlize all rewards based on final reward of the game
    pass

class RCActionEC4:
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
    return len(self.acts)

  def to_dataset(self):
    csts = rnn.pad_sequence(self.csts, batch_first=True)
    acts = rnn.pad_sequence(self.acts, batch_first=True)
    rewards = rnn.pad_sequence(self.rewards, batch_first=True)
    return RCActionEB4(csts, acts, rewards)

class RCActionEB4(data.Dataset):
  def __init__(self, cst, act, rewards):
    self.cur_states = cst
    self.actions    = act
    self.rewards    = rewards

  def __getitem__(self, index):
    cur_state = self.cur_states[index]
    action    = self.actions[index]
    reward    = self.rewards[index]
    return (cur_state, action, reward)

  def __len__(self):
    return self.actions.shape[0]