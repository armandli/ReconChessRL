import torch
from torch.utils import data

class RCActionEC1:
  def __init__(self):
    self.csts = []
    self.nsts = []
    self.actions = []
    self.rewards = []
    self.terminals = []

  def append_st(self, cst, action):
    cst = cst.unsqueeze(0)
    self.csts.append(cst)
    self.actions.append(action)

  def append_post(self, nst, reward):
    nst = nst.unsqueeze(0)
    self.nsts.append(nst)
    self.rewards.append(reward)
    self.terminals.append(False)

  def append_terminal(self, reward):
    self.rewards[-1] = reward
    self.terminals[-1] = True

  def size(self):
    return len(self.terminals)

  def to_dataset(self):
    assert(len(self.csts) == len(self.nsts) == len(self.actions) == len(self.rewards) == len(self.terminals))
    csts = torch.cat(self.csts, dim=0)
    nsts = torch.cat(self.nsts, dim=0)
    actions = torch.tensor(self.actions, dtype=torch.long)
    rewards = torch.tensor(self.rewards, dtype=torch.float32)
    terminals = torch.tensor(self.terminals, dtype=torch.bool)
    return RCActionEB1(csts, nsts, actions, rewards, terminals)

def combine_action_ec1(ecs):
  if not ecs:
    return RCActionEC1()
  new_ec = RCActionEC1()
  csts = ecs[0].csts
  nsts = ecs[0].nsts
  actions = ecs[0].actions
  rewards = ecs[0].rewards
  terminals = ecs[0].terminals
  for i in range(1, len(ecs)):
    csts.extend(ecs[i].csts)
    nsts.extend(ecs[i].nsts)
    actions.extend(ecs[i].actions)
    rewards.extend(ecs[i].rewards)
    terminals.extend(ecs[i].terminals)
  new_ec.csts = csts
  new_ec.nsts = nsts
  new_ec.actions = actions
  new_ec.rewards = rewards
  new_ec.terminals = terminals
  return new_ec

class RCActionEB1(data.Dataset):
  def __init__(self, cst, nst, act, reward, terminal):
    self.cur_states = cst
    self.nxt_states = nst
    self.actions    = act
    self.rewards    = reward
    self.terminals  = terminal

  def __getitem__(self, index):
    cur_state = self.cur_states[index]
    nxt_state = self.nxt_states[index]
    action    = self.actions[index]
    reward    = self.rewards[index]
    terminal  = self.terminals[index]
    return (cur_state, nxt_state, action, reward, terminal)

  def __len__(self):
    return self.actions.shape[0]
