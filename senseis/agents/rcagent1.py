from typing import Optional, List, Tuple
import random
from reconchess import Color, Player, Square, WinReason, GameHistory
from chess import Board, Piece, Move
import torch

from .rc_qagent1 import RCQAgent1
from senseis.encoders.rc_encoder1 import RCStateEncoder1, RCActionEncoder1, RCSenseEncoder1
from senseis.models.rc_sense_model1 import RCSenseModel1
from senseis.models.rc_action_model1 import RCActionModel1

class RCAgent1(Player):
  def __init__(self):
    device = torch.device('cpu')
    action_model = torch.load('models/rc_action_model1.pt', map_location=device)
    sense_model = torch.load('models/rc_sense_model1.pt', map_location=device)
    self.agent = RCQAgent1(
      RCStateEncoder1(),
      RCActionEncoder1(),
      RCSenseEncoder1(),
      action_model,
      sense_model,
      device
    )

  def handle_game_start(self, color: Color, board: Board, opponent_name: str):
    self.agent.handle_game_start(color, board, opponent_name)

  def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
    self.agent.handle_opponent_move_result(captured_my_oiece, capture_square)

  def choose_sense(self, sense_action: List[Square], move_action: List[Move], seconds_left: float) -> Optional[Square]:
    return self.agent.choose_sense(sense_action, move_action, seconds_left)

  def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    self.agent.handle_sense_result(sense_result)

  def choose_move(self, move_action: List[Move], seconds_left: float) -> Optional[Move]:
    return self.agent.choose_move(move_action, seconds_left)

  def handle_move_result(self, requested_move: Optional[Move], taken_move: Optional[Move], captured_opponent_piece: bool, capture_square: Optional[Square]):
    self.agent.handle_move_result(requested_move, taken_move, captured_opponent_piece, capture_square)

  def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
    self.agent.handle_game_end(winner_color, win_reason, game_history)
