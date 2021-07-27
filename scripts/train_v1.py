import torch
from senseis.learning.rc_qtrainer import EGParams, QConfig, RCQTrainer
from senseis.reporters.rc_reporter import RCEpisodicReporter

device = torch.device("cpu")

config = QConfig(
    device=device,
    action_model_filename='../models/rc_action_model_v1.pt',
    sense_model_filename='../models/rc_sense_model_v1.pt',
    episodes=20,
    iterations=4,
    eb_size=128,
    batchsize=64,
    tc_steps=2,
    learning_rate=0.0001,
    weight_decay=0.000001,
    gamma=0.999,
    pg_epsilon=0.0001,
    epl_param=EGParams(
        epsilon_step=0,
        epsilon_scale=200,
        epsilon_max=0.5,
        epsilon_min=0.02,
        epsilon=0.5
    )
)
reporter = RCEpisodicReporter(config.batchsize, 1, 1)

trainer = RCQTrainer(config, reporter)
trainer.train()
