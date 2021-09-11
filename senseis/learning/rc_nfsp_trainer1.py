from os import path
import random
import torch
from torch import nn
from torch.utils import data
from torch.optim import Adam

from .rc_trainer import RCSelfTrainer

from senseis.collectors.rc_sense_eb2 import RCSenseEpisode2, RCSenseEC2, RCSenseEB2
from senseis.collectors.rc_action_eb4 import RCActionEpisode4, RCActionEC4, RCActionEB4
from senseis.collectors.rc_action_eb5 import RCActionEpisode5, RCActionEC5, RCActionEB5
from senseis.encoders.rc_encoder5 import RCStateEncoder5, RCActionEncoder4, RCSenseEncoder3
from senseis.models.rc_sense_model2 import RCSenseModel2
from senseis.models.rc_action_model4 import RCActionModel4
from senseis.rewards.rc_sense_reward import rc_sense_reward2
from senseis.rewards.rc_action_reward import rc_action_reward2
from senseis.agents.rc_nfsp_agent1 import RCNFSPAgent1
from senseis.torch_modules.loss import PGError
from senseis.learning.rc_qconfig import NFSPConfig

class RCNFSPTrainer1(RCSelfTrainer):
  def __init__(self, config: NFSPConfig, reporter):
    self.config = config
    self.sense_ec = RCSenseEC2()
    self.sense_episode_exp = []
    self.action_alpha_ec = [RCActionEC4(), RCActionEC4()]
    self.action_beta_ec  = [RCActionEC5(), RCActionEC5()]
    self.action_alpha_exp = [None, None]
    self.action_beta_exp  = [None, None]
    self.sense_model = None
    self.action_alpha_model = [None, None]
    self.action_beta_model  = [None, None]
    self.snapshot_count = None
    self.reporter = reporter
    self.is_agent1 = True

  def episodes(self):
    return self.config.episodes

  def initialize(self):
    # do nothing
    pass

  def teardown(self):
    for i in range(2):
      self.action_alpha_ec[i].append_episode(self.action_alpha_exp[i])
      self.action_beta_ec[i].append_episode(self.action_beta_exp[i])
    self.action_alpha_exp = [None, None]
    self.action_beta_exp  = [None, None]
    for exp in self.sense_episode_exp:
      self.sense_ec.append_episode(exp)
    self.sense_episode_exp = []

  def create_agent(self):
    if self.sense_model is None:
      if path.exists(self.config.sense_model_filename):
        self.sense_model = torch.load("{}.pt".format(self.config.sense_model_filename), map_location=self.config.device)
      else:
        self.sense_model = RCSenseModel2(RCStateEncoder5.sense_dimension(), RCSenseEncoder3.dimension(), self.config.sense_hidden_size, 1)
    for i in range(2):
      if self.action_alpha_model[i] is None:
        if path.exists(self.config.action_alpha_model_filename + str(i)):
          self.action_alpha_model[i] = torch.load("{}_{}.pt".format(self.config_action_alpha_model_filename, i), map_location=self.config.device)
        else:
          self.action_alpha_model[i] = RCActionModel4(RCStateEncoder5.action_dimension(), RCActionEncoder4.dimension(), self.config.action_alpha_hidden_size, 1)
      if self.action_beta_model[i] is None:
        if path.exists(self.config.action_beta_model_filename + str(i)):
          self.action_beta_model[i] = torch.load("{}_{}.pt".format(self.config_action_beta_model_filename, i), map_location=self.config.device)
        else:
          self.action_beta_model[i] = RCActionModel4(RCStateEncoder5.action_dimension(), RCActionEncoder4.dimension(), self.config.action_beta_hidden_size, 1)
    if self.is_agent1:
      action_alpha_model = self.action_alpha_model[0]
      action_beta_model = self.action_beta_model[0]
      self.action_alpha_exp[0] = RCActionEpisode4()
      self.action_beta_exp[0] = RCActionEpisode5()
      action_alpha_exp = self.action_alpha_exp[0]
      action_beta_exp = self.action_beta_exp[0]
    else:
      action_alpha_model = self.action_alpha_model[1]
      action_beta_model = self.action_beta_model[1]
      self.action_alpha_exp[1] = RCActionEpisode4()
      self.action_beta_exp[1] = RCActionEpisode5()
      action_alpha_exp = self.action_alpha_exp[1]
      action_beta_exp = self.action_beta_exp[1]
    sense_exp = RCSenseEpisode2()
    self.sense_episode_exp.append(sense_exp)
    is_best_response = random.random() < self.config.mu
    agent = RCNFSPAgent1(
        RCStateEncoder5(),
        RCActionEncoder4(),
        RCSenseEncoder3(),
        action_alpha_model,
        action_beta_model,
        self.sense_model,
        self.config.device,
        is_best_response,
        action_alpha_exp,
        action_beta_exp,
        sense_exp,
        rc_action_reward2,
        rc_sense_reward2,
    )
    self.is_agent1 = not self.is_agent1
    return agent

  def should_learn(self, episode):
    min_episodes = min(min([ec.size() for ec in self.action_alpha_ec]), min([ec.size() for ec in self.action_beta_ec]), self.sense_ec.size())
    if min_episodes == 0:
      return False
    if episode == self.config.episodes - 1:
      return True
    if min_episodes >= self.config.eb_size:
      return True
    return False

  def learn(self, episode):
    self.learn_sense(episode)
    self.learn_action_alpha(episode)
    self.learn_action_beta(episode)
    self.sense_ec = RCSenseEC2()
    self.action_alpha_ec = [RCActionEC4(), RCActionEC4()]
    self.action_beta_ec = [RCActionEC5(), RCActionEC5()]
    if self.config.snapshot_frequency > 0 and episode / self.config.snapshot_frequency > self.snapshot_count:
      for i in range(2):
        action_alpha_model_snapshot_filename = "{}_{}.pt".format(self.config.action_alpha_model_snapshot_prefix + str(i), self.snapshot_count)
        torch.save(self.action_alpha_model[i], action_alpha_model_snapshot_filename)
        action_beta_model_snapshot_filename = "{}_{}.pt".format(self.config.action_beta_model_snapshot_prefix + str(i), self.snapshot_count)
        torch.save((self.action_beta_model[i], action_beta_model_snapshot_filename))
      sense_model_snapshot_filename = "{}_{}.pt".format(self.config.sense_model_snapshot_prefix, self.snapshot_count)
      torch.save(self.sense_model, sense_model_snapshot_filename)
    if episode == self.config.episodes - 1:
      for i in range(2):
        torch.save(self.action_alpha_model[i], "{}_{}.pt".format(self.config.action_alpha_model_filename, i))
        torch.save(self.action_beta_model[i], "{}_{}.pt".format(self.config.action_beta_model_filename, i))
      torch.save(self.sense_model, "{}.pt".format(self.config.sense_model_filename))

  def learn_sense(self, episode):
    sense_eb = self.sense_ec.to_dataset()
    sense_loader = data.DataLoader(sense_eb, batch_size=self.config.batchsize, shuffle=True, pin_memory=True, num_workers=0)
    optimizer = Adam(self.sense_model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
    loss = PGError()
    self.sense_model.train()
    for e in range(self.config.iterations):
      for i, (cs, a, r) in enumerate(sense_loader):
        optimizer.zero_grad()
        batchsize = a.shape[0]
        # cs in (b, S, F), a in (b, S, 1), r in (b, S, 1)
        cs, a, r = cs.to(self.config.device), a.to(self.config.device), r.to(self.config.device)
        h = self.sense_model.init(batchsize).to(self.config.device)
        pi, _ = self.sense_model(cs, h)
        pi = pi.reshape(pi.shape[0] * pi.shape[1], pi.shape[2])
        a = a.reshape(a.shape[0] * a.shape[1])
        r = r.reshape(r.shape[0] * r.shape[1])
        pi = torch.index_select(pi, 1, a).diagonal()
        l = loss(pi, r, self.config.pg_epsilon)
        l.backward()
        optimizer.step()
        self.reporter.train_sense_gather(episode, i, len(sense_eb), l.item())

  def learn_action_alpha(self, episode):
    for i in range(2):
      action_eb = self.action_alpha_ec[i].to_dataset()
      action_loader = data.DataLoader(action_eb, batch_size=self.config.batchsize, shuffle=True, pin_memory=True, num_workers=0)
      optimizer = Adam(self.action_alpha_model[i].parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
      loss = PGError()
      self.action_alpha_model[i].train()
      for e in range(self.config.iterations):
        for i, (cs, a, r) in enumerate(action_loader):
          optimizer.zero_grad()
          batchsize = a.shape[0]
          cs, a, r = cs.to(self.config.device), a.to(self.config.device), r.to(self.config.device)
          h = self.action_alpha_model[i].init(batchsize).to(self.config.device)
          pi, _ = self.action_alpha_model[i](cs, h)
          pi = pi.reshape(pi.shape[0] * pi.shape[1], pi.shape[2])
          a = a.reshape(a.shape[0] * a.shape[1])
          r = r.reshape(r.shape[0] * r.shape[1])
          pi = torch.index_select(pi, 1, a).diagonal()
          l = loss(pi, r, self.config.pg_epsilon)
          l.backward()
          optimizer.step()
          self.reporter.train_action_gather(episode, i, len(action_eb), l.item())

  def learn_action_beta(self, episode):
    for i in range(2):
      action_eb = self.action_beta_ec[i].to_dataset()
      action_loader = data.DataLoader(action_eb, batch_size=self.config.batchsize, shuffle=True, pin_memory=True, num_workers=0)
      optimizer = Adam(self.action_beta_model[i].parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
      loss = nn.CrossEntropyLoss()
      self.action_beta_model[i].train()
      for e in range(self.config.iterations):
        for i, (cs, a) in enumerate(action_loader):
          optimizer.zero_grad()
          batchsize = a.shape[0]
          cs, a = cs.to(self.config.device), a.to(self.config.device)
          h = self.action_beta_model[i].init(batchsize).to(self.config.device)
          pi, _ = self.action_beta_model[i](cs, h)
          pi = pi.reshape(pi.shape[0] * pi.shape[1], pi.shape[2])
          a = a.reshape(a.shape[0] * a.shape[1])
          l = loss(pi, a)
          l.backward()
          optimizer.step()
          self.reporter.train_action_gather(episode, 1, len(action_eb), l.item())
