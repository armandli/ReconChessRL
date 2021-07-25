from typing import Optional, List, Tuple
import random
from reconchess import Color, Player, Square, WinReason, GameHistory
from chess import Board, Piece, Move
import torch

class RCQAgent1(Player):
  def __init__(self, state_encoder, action_encoder, sense_encoder, action_model, sense_model, action_exp, sense_exp, action_reward, sense_reward, device, epsilon):
    self.state_encoder = state_encoder
    self.action_encoder = action_encoder
    self.sense_encoder = sense_encoder
    self.action_model = action_model
    self.sense_model = sense_model
    self.sense_exp = sense_exp
    self.action_exp = action_exp
    self.action_reward = action_reward
    self.sense_reward = sense_reward
    self.device = device
    self.epsilon = epsilon
    self.self_capture_count = None
    self.oppo_capture_count = None
    self.color = None
    self.sense_cst = None # for sense reward calculation

  def handle_game_start(self, color: Color, board: Board, opponent_name: str):
    self.state_encoder.init(color, board)
    self.color = color
    self.self_capture_count = 0
    self.oppo_capture_count = 0

  def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
    self.state_encoder.op_move_update(capture_square)
    if captured_my_piece:
      self.oppo_capture_count += 1

  def choose_sense(self, sense_action: List[Square], move_action: List[Move], seconds_left: float) -> Optional[Square]:
    cst = self.state_encoder.encode()
    self.sense_cst = cst
    cst_dev = cst.unsqueeze(0).to(self.device)
    act = self.sense_model(cst_dev)
    action = self.sense_decoder.decode(act)[0]
    self.sense_exp.append_st(cst, action)
    return action

  def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    self.state_encoder.sense_update(sense_result)
    nst = self.state_encoder.encode()
    reward = self.sense_reward(self.sense_cst, nst)
    self.sense_exp.append_post(reward)

  def choose_move(self, move_action: List[Move], seconds_left: float) -> Optional[Move]:
    cst = self.state_encoder.encode()
    if random.random() > self.epsilon:
      with torch.no_grad():
        cst_dev = cst.unsqueeze(0).to(self.device)
        act = self.action_model(cst_dev)
        action = self.action_encoder.decode(act)[0]
        if action not in move_action:
          action = None
    else:
      action = random.choice(move_action)
    self.action_exp.append_st(cst, action)
    return action

  def handle_move_result(self, requested_move: Optional[Move], taken_move: Optional[Move], captured_opponent_piece: bool, capture_square: Optional[Square]):
    if captured_opponent_piece:
      self.self_capture_count += 1
    self.state_encoder.move_update(taken_move, capture_square)
    nst = self.state_encoder.encode()
    reward = self.action_reward(self.self_capture_count, self.oppo_capture_count, False)
    self.action_exp.append_post(nst, reward)

  def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
    win = True if winner_color == self.color else False
    reward = self.action_reward(self.self_capture_count, self.oppo_capture_count, True, win)
    self.action_exp.append_terminal(reward)
