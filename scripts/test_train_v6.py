import torch
from senseis.learning.rc_qconfig import PGConfig
from senseis.learning.rc_pg_trainer1 import RCPGTrainer1
from senseis.reporters.rc_reporter import RCEpisodicReporter

device = torch.device("cpu")

config = PGConfig(
    device=device,
    action_model_filename='../models/rc_action_model_v5.pt',
    sense_model_filename='../models/rc_sense_model_v5.pt',
    episodes=4,
    iterations=2,
    eb_size=2,
    batchsize=2,
    learning_rate=0.0001,
    weight_decay=0.0000001,
    pg_epsilon=0.000000001,
    action_hidden_size=32,
    sense_hidden_size=32,
)

reporter = RCEpisodicReporter(config.batchsize, 1, 1)

trainer = RCPGTrainer1(config, reporter)
trainer.train()
