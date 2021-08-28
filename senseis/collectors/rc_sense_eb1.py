import torch
from torch.utils import data

class RCSenseEC1:
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

  def last_cst(self):
    return self.csts[-1]

  def size(self):
    return len(self.rewards)

  def to_dataset(self):
    assert(len(self.csts) == len(self.acts) == len(self.rewards))
    csts = torch.cat(self.csts, dim=0)
    acts = torch.tensor(self.acts, dtype=torch.long)
    rewards = torch.tensor(self.rewards, dtype=torch.float32)
    return RCSenseEB1(csts, acts, rewards)

def combine_sense_ec1(ecs):
  if not ecs:
    return RCSenseEC1()
  new_ec = RCSenseEC1()
  csts = ecs[0].csts
  acts = ecs[0].acts
  rewards = ecs[0].rewards
  for i in range(1, len(ecs)):
    csts.extend(ecs[i].csts)
    acts.extend(ecs[i].acts)
    rewards.extend(ecs[i].rewards)
  new_ec.csts = csts
  new_ec.acts = acts
  new_ec.rewards = rewards
  return new_ec

class RCSenseEB1(data.Dataset):
  def __init__(self, cst, act, reward):
    self.cur_states = cst
    self.actions    = act
    self.rewards    = reward

  def __getitem__(self, index):
    cur_state = self.cur_states[index]
    action    = self.actions[index]
    reward    = self.rewards[index]
    return (cur_state, action, reward)

  def __len__(self):
    return self.actions.shape[0]
