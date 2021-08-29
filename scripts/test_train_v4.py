import torch
from senseis.learning.rc_qconfig import DaggerConfig
from senseis.learning.rc_dagger_trainer1 import RCDaggerTrainer1
from senseis.reporters.rc_reporter import RCEpisodicReporter

device = torch.device("cpu")

config = DaggerConfig(
    device=device,
    action_model_filename='../models/rc_action_model_v4.pt',
    sense_model_filename='../models/rc_sense_model_v4.pt',
    episodes=4,
    iterations=2,
    eb_size=2,
    batchsize=2,
    learning_rate=0.0001,
    weight_decay=0.000001,
    pg_epsilon=0.00000001,
    sense_hidden_size=32,
    action_hidden_size=32,
)

reporter = RCEpisodicReporter(config.batchsize, 1, 1)

trainer = RCDaggerTrainer1(config, reporter)
trainer.train()
