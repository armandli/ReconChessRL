from abc import ABC, abstractmethod

class RLReporter(ABC):
  @abstractmethod
  def train_gather(self, episode, step, loss):
    pass
  def train_reset(self):
    pass

class ConsoleRLReporter(RLReporter):
  def __init__(self, batch_size, frequency, report_frequency=0):
    self.batch_size = batch_size
    self.frequency = frequency
    self.report_freq = report_frequency
    self.losses = []

  def train_gather(self, episode, step, loss):
    self.losses.append((episode, step, loss))
    if self.report_freq > 0 and step % self.report_freq == 0:
      print('Episode {} Step {} Loss: {}'.format(episode, step, loss))

  def offline_train_gather(self, episode, batch_idx, dataset_size, loss):
    if episode % self.frequency == 0 and (batch_idx + 1) * self.batch_size >= dataset_size:
      self.losses.append((episode, loss))
    if self.report_freq > 0 and episode % self.report_freq == 0 and (batch_idx + 1) * self.batch_size >= dataset_size:
      print("Episode {}, {}/{}: {}".format(episode, batch_idx * self.batch_size, dataset_size, loss))