from typing import Optional, List, Tuple
from copy import copy
import random
from reconchess import Color, Player, Square, WinReason, GameHistory
from chess import Board, Piece, Move
import torch
from senseis.encoders.rc_encoder_util import update_board_sense_result1, update_board_self_move1, update_board_oppo_move1

# Neural Ficticious Self Play Agent, with event sequence RNN model
# compatible with RCStateEncoder5, RCActionModel4, RCSenseModel2

class RCNFSPAgent1(Player):
  def __init__(
    self,
    state_encoder,
    action_encoder,
    sense_encoder,
    action_alpha_model,
    action_beta_model,
    sense_model,
    device,
    is_best_response,
    action_alpha_exp=None,
    action_beta_exp=None,
    sense_exp=None,
    action_reward=None,
    sense_reward=None,
  ):
    self.state_encoder = state_encoder
    self.action_encoder = action_encoder
    self.sense_encoder = sense_encoder
    self.action_alpha_model = action_alpha_model
    self.action_beta_model = action_beta_model
    self.sense_model = sense_model
    self.action_alpha_exp = action_alpha_exp
    self.action_beta_exp = action_beta_exp
    self.sense_exp = sense_exp
    self.action_reward = action_reward
    self.sense_reward = sense_reward
    self.device = device
    self.is_best_response = is_best_response
    self.color = None
    self.sense_board = None
    self.action_prev_board = None
    self.action_board = None
    self.action_hvec = None
    self.sense_hvec = None

  def handle_game_start(self, color: Color, board: Board, opponent_name: str):
    self.state_encoder.init(color, board)
    self.action_encoder.init(color, board)
    self.sense_encoder.init(color, board)
    self.color = color
    self.sense_board = copy(board)
    self.action_board = copy(board)
    if self.is_best_response:
      self.action_hvec = self.action_alpha_model.init(1)
    else:
      self.action_hvec = self.action_beta_model.init(1)
    self.action_hvec = self.action_hvec.to(self.device)
    self.sense_hvec = self.sense_model.init(1)
    self.sense_hvec = self.sense_hvec.to(self.device)

  def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
    self.state_encoder.op_move_update(capture_square)
    update_board_oppo_move1(self.sense_board, capture_square)
    self.action_prev_board = copy(self.action_board)
    update_board_oppo_move1(self.action_board, capture_square)

  def choose_sense(self, sense_action: List[Square], move_action: List[Move], seconds_left: float) -> Optional[Square]:
    with torch.no_grad():
      cst = self.state_encoder.encode_sense()
      cst_dev = cst.unsqueeze(0).unsqueeze(0).to(self.device)
      act, h = self.sense_model(cst_dev, self.sense_hvec)
      self.sense_hvec = h
      actions = self.sense_encoder.decode(act)
      action = actions[0][0]
      action_idxes = self.sense_encoder.action_index(actions[0])
      action_index = action_idxes[0]
      if self.sense_exp is not None:
        self.sense_exp.append_st(cst, action_index)
      return action

  def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    prev_board = copy(self.sense_board)
    self.state_encoder.sense_update(sense_result)
    update_board_sense_result1(self.sense_board, sense_result, self.color)
    update_board_sense_result1(self.action_board, sense_result, self.color)
    if self.sense_exp is not None:
      reward = self.sense_reward(prev_board, self.sense_board, self.color)
      self.sense_exp.append_post(reward)

  def choose_move(self, move_action: List[Move], seconds_left: float) -> Optional[Move]:
    with torch.no_grad():
      cst = self.state_encoder.encode_action()
      cst_dev = cst.unsqueeze(0).unsqueeze(0).to(self.device)
      if self.is_best_response:
        act, h = self.action_alpha_model(cst_dev, self.action_hvec)
      else:
        act, h = self.action_beta_model(cst_dev, self.action_hvec)
      self.action_hvec = h
      action = self.action_encoder.decode(act)[0][0]
      if self.action_alpha_exp is not None or self.action_beta_exp is not None:
        self.action_alpha_exp.append_st(cst)
        if self.is_best_response:
          self.action_beta_exp.append_st(cst)
      if action in move_action:
        return action
      # to deal with stalemate where no agent is making a valid move for a long time
      return random.choice(move_action)

  def handle_move_result(self, requested_move: Optional[Move], taken_move: Optional[Move], captured_opponent_piece: bool, capture_square: Optional[Square]):
    update_board_self_move1(self.sense_board, taken_move, capture_square)
    if self.action_prev_board is None:
      slf.action_prev_board = copy(self.action_board)
    update_board_self_move1(self.action_board, taken_move, capture_square)
    self.state_encoder.move_update(taken_move, capture_square)
    if self.action_alpha_exp is not None or self.action_beta_exp is not None:
      reward = self.action_reward(False, prev_board=self.action_prev_board, next_board=self.action_board, my_color=self.color)
      action_idx = self.action_encoder.action_index(taken_move)
      self.action_alpha_exp.append_post(action_idx, reward)
      if self.is_best_response:
        self.action_beta_exp.append_post(action_idx)

  def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
    if self.action_alpha_exp is not None:
      reward = self.action_reward(True, winner=winner_color, my_color=self.color)
      self.action_alpha_exp.append_terminal(reward)
