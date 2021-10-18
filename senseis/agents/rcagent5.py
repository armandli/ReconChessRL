from typing import Optional, List, Tuple
import random
from reconchess import Color, Player, Square, WinReason, GameHistory
from chess import Board, Piece, Move
import torch

# NFSP Best Action Agent Agent

from senseis.encoders.rc_encoder5 import RCStateEncoder5, RCActionEncoder4, RCSenseEncoder3
from senseis.models.rc_sense_model2 import RCSenseModel2
from senseis.models.rc_action_model4 import RCActionModel4

import senseis.agents.rc_nfsp_agent1 as agent

class RCAgent5(Player):
  def __init__(self):
    device = torch.device("cpu")
    action_alpha_model = torch.load('models/rc_action_alpha_model_v7_1.pt', map_location=device)
    action_beta_model = torch.load('models/rc_action_beta_model_v7_1.pt', map_location=device)
    sense_model = torch.load('models/rc_sense_model_v7.pt', map_location=device)
    self.agent = agent.RCNFSPAgent1(
        RCStateEncoder5(),
        RCActionEncoder4(),
        RCSenseEncoder3(),
        action_alpha_model,
        action_beta_model,
        sense_model,
        device,
        True
    )

  def handle_game_start(self, color: Color, board: Board, opponent_name: str):
    self.agent.handle_game_start(color, board, opponent_name)

  def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
    self.agent.handle_opponent_move_result(captured_my_piece, capture_square)

  def choose_sense(self, sense_action: List[Square], move_action: List[Move], seconds_left: float) -> Optional[Square]:
    move = self.agent.choose_sense(sense_action, move_action, seconds_left)
    return move

  def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    self.agent.handle_sense_result(sense_result)

  def choose_move(self, move_action: List[Move], seconds_left: float) -> Optional[Move]:
    move = self.agent.choose_move(move_action, seconds_left)
    return move

  def handle_move_result(self, requested_move: Optional[Move], taken_move: Optional[Move], captured_opponent_piece: bool, capture_square: Optional[Square]):
    self.agent.handle_move_result(requested_move, taken_move, captured_opponent_piece, capture_square)

  def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
    self.agent.handle_game_end(winner_color, win_reason, game_history)
