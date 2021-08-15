from dataclasses import dataclass
import torch

@dataclass
class EGParams:
  epsilon_step: float
  epsilon_scale: float
  epsilon_max: float
  epsilon_min: float
  epsilon: float

def epsilon_decay(param: EGParams):
  param.epsilon = min(
    param.epsilon_max,
    1. / (param.epsilon_step / param.epsilon_scale + 1. / param.epsilon_max) + param.epsilon_min
  )
  param.epsilon_step += 1
  return param

@dataclass
class QConfig:
  device: torch.device
  action_model_filename: str
  sense_model_filename: str
  episodes: int
  iterations: int
  eb_size: int
  batchsize: int
  tc_steps: int
  learning_rate: float
  weight_decay: float
  gamma: float
  pg_epsilon: float
  epl_param: EGParams
  action_model_snapshot_prefix: str = None
  sense_model_snapshot_prefix: str = None
  snapshot_frequency: int = 0

@dataclass
class SenseConfig:
  device: torch.device
  sense_model_filename: str
  episodes: int
  iterations: int
  eb_size: int
  batchsize: int
  learning_rate: float
  weight_decay: float
  pg_epsilon: float
  sense_model_snapshot_prefix: str = None
  snapshot_frequency: int = 0
