from abc import ABC, abstractmethod
from reconchess import play_local_game, LocalGame


class RCSelfTrainer(ABC):
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
      agent1 = self.create_agent()
      agent2 = self.create_agent()
      game =LocalGame(900)
      try:
        _, _, _ = play_local_game(agent1, agent2, game=game)
      except:
        game.end()
        return
      if self.should_learn():
        self.learn(e)
