import torch
from senseis.learning.rc_qconfig import NFSPConfig
from senseis.learning.rc_nfsp_trainer1 import RCNFSPTrainer1
from senseis.reporters.rc_reporter import RCEpisodicReporter

device = torch.device("cpu")

config = NFSPConfig(
    device=device,
    action_alpha_model_filename='../models/rc_action_alpha_model_v6',
    action_beta_model_filename='../models/rc_action_beta_model_v6',
    sense_model_filename='../models/rc_sense_model_v6',
    episodes=12,
    iterations=2,
    eb_size=12,
    batchsize=12,
    learning_rate=0.0001,
    weight_decay=0.0000001,
    pg_epsilon=0.000000001,
    mu=0.2,
    action_alpha_hidden_size=32,
    action_beta_hidden_size=32,
    sense_hidden_size=32,
)

reporter = RCEpisodicReporter(config.batchsize, 1, 1)

trainer = RCNFSPTrainer1(config, reporter)
trainer.train()
