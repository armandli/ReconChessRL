import torch
from senseis.learning.rc_qconfig import SenseConfig2
from senseis.learning.rc_sense_trainer2 import RCSenseTrainer2
from senseis.reporters.rc_reporter import RCEpisodicReporter

device = torch.device("cpu")

config = SenseConfig2(
    device=device,
    sense_model_filename='../models/rc_sense_model_v4.pt',
    episodes=4,
    iterations=2,
    eb_size=2,
    batchsize=2,
    learning_rate=0.0001,
    weight_decay=0.000001,
    pg_epsilon=0.00000001,
    sense_hidden_size=32
)

reporter = RCEpisodicReporter(config.batchsize, 1, 1)

trainer = RCSenseTrainer2(config, reporter)
trainer.train()
