from os import path
import torch
from torch import nn
from torch.utils import data
from toch.optim import Adam

from .rc_trainer import RCSelfTrainer

from senseis.collectors.rc_sense_eb import RCSenseEC, RCSenseEB, combine_sense_ec
from sneseis.encoders.rc_encoder2 import RCSenseEncoder2
from senseis.encoders.rc_encoder4 import RCStateEncoder4
from senseis.models.rc_sense_model import RCSenseModel1
from senseis.rewards.rc_sense_reward import rc_sense_reward1

#TODO
