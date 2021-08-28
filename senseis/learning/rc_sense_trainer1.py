from os import path
import torch
from torch import nn
from torch.utils import data
from torch.optim import Adam

from .rc_trainer import RCSelfTrainer

from senseis.collectors.rc_sense_eb1 import RCSenseEC1, RCSenseEB1, combine_sense_ec1
from senseis.encoders.rc_encoder2 import RCSenseEncoder2
from senseis.encoders.rc_encoder4 import RCStateEncoder4
from senseis.models.rc_sense_model1 import RCSenseModel1
from senseis.rewards.rc_sense_reward import rc_sense_reward1
from senseis.agents.rc_troute_agent1 import RCTrouteAgent1
from senseis.torch_modules.loss import PGError
from senseis.learning.rc_qconfig import SenseConfig1


class RCSenseTrainer1(RCSelfTrainer):
  def __init__(self, config: SenseConfig1, reporter):
    self.configuration = config
    self.sense_ecs  = []
    self.sense_model = None
    self.snapshot_count = None
    self.reporter = reporter

  def episodes(self):
    return self.configuration.episodes

  def create_agent(self):
    sense_ec = RCSenseEC1()
    if self.sense_model is None:
      if path.exists(self.configuration.sense_model_filename):
        self.sense_model = torch.load(self.configuration.sense_model_filename, map_location=self.configuration.device)
      else:
        self.sense_model = RCSenseModel1(*RCStateEncoder4.dimension(), RCSenseEncoder2.dimension())

    agent = RCTrouteAgent1(
        RCStateEncoder4(),
        RCSenseEncoder2(),
        self.sense_model,
        self.configuration.device,
        sense_ec,
        rc_sense_reward1
    )
    self.sense_ecs.append(sense_ec)
    return agent

  def should_learn(self, episode):
    if episode == self.configuration.episodes - 1:
      return True
    sense_size = sum([ec.size() for ec in self.sense_ecs])
    if sense_size >= self.configuration.eb_size:
      return True
    else:
      return False

  def learn(self, episode):
    self.learn_sense(episode)
    if self.configuration.snapshot_frequency > 0 and episode / self.configuration.snapshot_frequency > self.snapshot_count:
      sense_model_snapshot_filename = "{}_{}.pt".format(self.configuration.sense_model_snapshot_prefix, self.snapshot_count)
      torch.save(self.sense_model, sense_model_snapshot_filename)
    if episode == self.configuration.episodes - 1:
      torch.save(self.sense_model, self.configuration.sense_model_filename)
    # clean up the experience buffer
    self.sense_ecs = []

  def learn_sense(self, episode):
    sense_ec = combine_sense_ec1(self.sense_ecs)
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
