from abc import ABC, abstractmethod
from reconchess import play_local_game, LocalGame


class RCSelfTrainer(ABC):
  @abstractmethod
  def initialize(self):
    pass

  @abstractmethod
  def teardown(self):
    pass

  @abstractmethod
  def episodes(self):
    pass

  @abstractmethod
  def create_agent(self):
    pass

  @abstractmethod
  def should_learn(self):
    pass

  @abstractmethod
  def learn(self):
    pass

  def train(self):
    for e in range(self.episodes()):
      self.initialize()
      agent1 = self.create_agent()
      agent2 = self.create_agent()
      game =LocalGame(900)
      try:
        winner, win_reason, history = play_local_game(agent1, agent2, game=game)
        print("winner {} win reason {}".format(winner, win_reason))
      except Exception as e:
        print("Exception {}".format(e))
        game.end()
        return
      self.teardown()
      if self.should_learn(e):
        self.learn(e)
