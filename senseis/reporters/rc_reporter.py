class RCEpisodicReporter:
  def __init__(self, batch_size, frequency, report_frequency=0):
    self.batch_size = batch_size
    self.frequency = frequency
    self.report_freq = report_frequency
    self.sense_losses = []
    self.action_losses = []

  def train_sense_gather(self, episode, batch_idx, dataset_size, loss):
    if episode % self.frequency == 0 and (batch_idx + 1) * self.batch_size >= dataset_size:
      self.sense_losses.append((episode, loss))
    if self.report_freq > 0 and episode % self.report_freq == 0 and (batch_idx + 1) * self.batch_size >= dataset_size:
      print("Sense Episode {}, {}/{}: {}".format(episode, batch_idx * self.batch_size, dataset_size, loss))

  def train_action_gather(self, episode, batch_idx, dataset_size, loss):
    if episode % self.frequency == 0 and (batch_idx + 1) * self.batch_size >= dataset_size:
      self.action_losses.append((episode, loss))
    if self.report_freq > 0 and episode % self.report_freq == 0 and (batch_idx + 1) * self.batch_size >= dataset_size:
      print("Action Episode {}, {}/{}: {}".format(episode, batch_idx * self.batch_size, dataset_size, loss))
