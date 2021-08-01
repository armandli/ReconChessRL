from os import path
import torch
from torch import nn
from torch.utils import data
from torch.optim import Adam

from .rc_trainer import RCSelfTrainer

from senseis.collectors.rc_sense_eb import RCSenseEC, RCSenseEB, combine_sense_ec
from senseis.collectors.rc_action_eb import RCActionEC, RCActionEB, combine_action_ec
from senseis.encoders.rc_encoder2 import RCStateEncoder2, RCActionEncoder2, RCSenseEncoder2
from senseis.models.rc_action_model2 import RCActionModel2
from senseis.models.rc_sense_model1 import RCSenseModel1
from senseis.rewards.rc_sense_reward import rc_sense_reward1
from senseis.rewards.rc_action_reward import rc_action_reward1
from senseis.agents.rc_qagent2 import RCQAgent2
from senseis.torch_modules.loss import PGError
from senseis.learning.rc_qconfig import EGParams, epsilon_decay, QConfig

class RCQTrainer2(RCSelfTrainer):
  def __init__(self, config: QConfig, reporter):
    self.configuration = config
    self.action_ecs = []
    self.sense_ecs  = []
    self.action_model = None
    self.sense_model = None
    self.snapshot_count = None
    self.reporter = reporter

  def episodes(self):
    return self.configuration.episodes

  def create_agent(self):
    action_ec = RCActionEC()
    sense_ec = RCSenseEC()
    if self.action_model is None:
      if path.exists(self.configuration.action_model_filename):
        self.action_model = torch.load(self.configuration.action_model_filename, map_location=self.configuration.device)
      else:
        self.action_model = RCActionModel2(*RCStateEncoder2.dimension(), RCActionEncoder2.dimension())
    if self.sense_model is None:
      if path.exists(self.configuration.sense_model_filename):
        self.sense_model = torch.load(self.configuration.sense_model_filename, map_location=self.configuration.device)
      else:
        self.sense_model = RCSenseModel1(*RCStateEncoder2.dimension(), RCSenseEncoder2.dimension())
    if self.configuration.action_model_filename:
      agent = RCQAgent2(
        RCStateEncoder2(),
        RCActionEncoder2(),
        RCSenseEncoder2(),
        self.action_model,
        self.sense_model,
        self.configuration.device,
        action_ec,
        sense_ec,
        rc_action_reward1,
        rc_sense_reward1,
        self.configuration.epl_param.epsilon
    )
    self.action_ecs.append(action_ec)
    self.sense_ecs.append(sense_ec)
    return agent

  def should_learn(self, episode):
    if episode == self.configuration.episodes - 1:
      return True
    action_size = sum([ec.size() for ec in self.action_ecs])
    sense_size = sum([ec.size() for ec in self.sense_ecs])
    if action_size >= self.configuration.eb_size and sense_size >= self.configuration.eb_size:
      return True
    else:
      return False

  def learn(self, episode):
    self.learn_sense(episode)
    self.learn_action(episode)
    if self.configuration.snapshot_frequency > 0 and episode / self.configuration.snapshot_frequency > self.snapshot_count:
      action_model_snapshot_filename = "{}_{}.pt".format(self.configuration.action_model_snapshot_prefix, self.snapshot_count)
      sense_model_snapshot_filename = "{}_{}.pt".format(self.configuration.sense_model_snapshot_prefix, self.snapshot_count)
      torch.save(self.action_model, action_model_snapshot_filename)
      torch.save(self.sense_model, sense_model_snapshot_filename)
    if episode == self.configuration.episodes - 1:
      torch.save(self.action_model, self.configuration.action_model_filename)
      torch.save(self.sense_model, self.configuration.sense_model_filename)
    # clean up the experience buffer
    self.action_ecs = []
    self.sense_ecs = []

  def learn_sense(self, episode):
    sense_ec = combine_sense_ec(self.sense_ecs)
    sense_eb = sense_ec.to_dataset()
    sense_loader = data.DataLoader(sense_eb, batch_size=self.configuration.batchsize, shuffle=True, pin_memory=True, num_workers=0)
    optimizer = self.sense_optimizer()
    loss = self.sense_loss()
    self.sense_model.train()
    for e in range(self.configuration.iterations):
      for i, (cs, a, r) in enumerate(sense_loader):
        optimizer.zero_grad()
        cs, a, r = cs.to(self.configuration.device), a.to(self.configuration.device), r.to(self.configuration.device)
        pi = self.sense_model(cs)
        pi = torch.index_select(pi, 1, a).diagonal()
        l = loss(pi, r, self.configuration.pg_epsilon)
        l.backward()
        optimizer.step()
        self.reporter.train_sense_gather(episode, i, len(sense_eb), l.item())

  def sense_optimizer(self):
    return Adam(
        self.sense_model.parameters(),
        lr=self.configuration.learning_rate,
        weight_decay=self.configuration.weight_decay
    )

  def sense_loss(self):
    return PGError()

  def learn_action(self, episode):
    action_ec = combine_action_ec(self.action_ecs)
    action_eb = action_ec.to_dataset()
    action_loader = data.DataLoader(action_eb, batch_size=self.configuration.batchsize, shuffle=True, pin_memory=True, num_workers=0)
    optimizer = self.action_optimizer()
    loss = self.action_loss()
    tmodel = RCActionModel2(*RCStateEncoder2.dimension(), RCActionEncoder2.dimension())
    tmodel.load_state_dict(self.action_model.state_dict())
    tmodel.eval()
    tc_step_count = 0
    for e in range(self.configuration.iterations):
      for i, (cs, ns, a, r, t) in enumerate(action_loader):
        if tc_step_count >= self.configuration.tc_steps:
          tmodel = RCActionModel2(*RCStateEncoder2.dimension(), RCActionEncoder2.dimension())
          tmodel.load_state_dict(self.action_model.state_dict())
          tmodel.eval()
          tc_step_count = 0
        optimizer.zero_grad()
        cs, ns, a, r, t = cs.to(self.configuration.device), ns.to(self.configuration.device), a.to(self.configuration.device), r.to(self.configuration.device), t.to(self.configuration.device)

        with torch.no_grad():
          self.action_model.eval()
          nqval = tmodel(ns)
          nqsel = self.action_model(ns)
          nqsel = torch.argmax(nqsel, dim=1)
          nqval = torch.index_select(nqval, 1, nqsel).diagonal()
          t = t.logical_not()
          target = r + self.configuration.gamma * t * nqval

        self.action_model.train()
        oqval = self.action_model(cs)
        oqval = torch.index_select(oqval, 1, a).diagonal()

        l = loss(oqval, target)
        l.backward()
        optimizer.step()
        self.reporter.train_action_gather(episode, i, len(action_eb), l.item())
        tc_step_count += 1
    self.configuration.epl_param = epsilon_decay(self.configuration.epl_param)

  def action_optimizer(self):
    return Adam(
        self.action_model.parameters(recurse=True),
        lr=self.configuration.learning_rate,
        weight_decay=self.configuration.weight_decay
    )

  def action_loss(self):
    return nn.MSELoss()
