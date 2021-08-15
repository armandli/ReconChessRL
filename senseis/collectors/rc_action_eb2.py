import torch
from torch.utils import data

# RC Action Experience Buffer for Immitation Learning

class RCActionEC2:
  def __init__(self):
    self.csts = []
    self.actions = []

  def append_st(self, cst, action):
    cst = cst.unsqueeze(0)
    self.csts.append(cst)
    self.actions.append(action)

  def size(self):
    return len(self.actions)

  def to_dataset(self):
    assert(len(self.csts) == len(self.actions))
    csts = torch.cat(self.csts, dim=0)
    actions = torch.tensor(self.actions, dtype=torch.long)
    return RCActionEB2(csts, actions)

def combine_action_ec2(ecs):
  if not ecs:
    return RCActionEC2()
  new_ec = RCActionEC2()
  csts = ecs[0].csts
  actions = ecs[0].actions
  for i in range(1, len(ecs)):
    csts.extend(ecs[i].csts)
    actions.extend(ecs[i].actions)
  new_ec.csts = csts
  new_ec.actions = actions
  return new_ec

class RCActionEB2(data.Dataset):
  def __init__(self, cst, act):
    self.cur_states = cst
    self.actions    = act

  def __getitem__(self, index):
    cur_state = self.cur_states[index]
    action    = self.actions[index]
    return (cur_state, action)

  def __len__(self):
    return self.actions.shape[0]
