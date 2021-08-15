import torch
from senseis.learning.rc_qconfig import SenseConfig
from senseis.learning.rc_sense_trainer1 import RCSenseTrainer1
from senseis.reporters.rc_reporter import RCEpisodicReporter

device = torch.device("cpu")

config = SenseConfig(
    device=device,
    sense_model_filename='../models/rc_sense_model_v3.pt',
    episodes=20,
    iterations=4,
    eb_size=128,
    batchsize=64,
    learning_rate=0.0001,
    weight_decay=0.000001,
    pg_epsilon=0.0001,
)
reporter = RCEpisodicReporter(config.batchsize, 1, 1)

trainer = RCSenseTrainer1(config, reporter)
trainer.train()
