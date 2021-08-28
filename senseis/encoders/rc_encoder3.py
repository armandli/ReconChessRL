from typing import Optional, List, Tuple
from reconchess import Color
from chess import Board, Piece, Move, Square
import numpy as np
import torch
from .rc_encoder_util import encode_move_type_dim3, decode_move_dim3, move_to_action_index3, encode_initial_board3, update_state_oppo2, update_state_self2, update_sense2, MOVE_MAP_SIZE_TOTAL

# adding in history as part of the state encoding
class RCStateEncoder3:
  # add in one history state frame from both sides = 6 * 2 * 2, add 1 dim for color, 8 by 8 board
  dim = (25, 8, 8)

  def __init__(self):
    # previous board tensor
    self.omp = None
    self.mmp = None
    self.om = None
    self.mm = None
    self.color = None
    self.counts = None

  def init(self, my_color: Color, board: Board):
    self.om, self.mm = encode_initial_board3(my_color, board)
    self.color = my_color
    # 0 opponent, 1 myself
    self.counts = [16, 16]

  def sense_update(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    self.om = update_sense2(self.om, sense_result, self.color)

  def move_update(self, taken_move: Optional[Move], captured_square: Optional[Square]):
    if captured_square is not None:
      self.counts[0] -= 1
    if not self.color: # black move second
      #TODO: make sure this is hard copy, not soft copy
      self.omp, self.mmp = self.om, self.mm
    self.om, self.mm = update_state_self2(self.om, self.mm, taken_move, captured_square, self.color)

  def op_move_update(self, capture_square: Optional[Square]):
    if capture_square is not None:
      self.counts[1] -= 1
    if self.color: # white move first
      #TODO: make sure this is hard copy, not soft copy
      self.omp, self.mmp = self.om, self.mm
    self.om, self.mm = update_state_oppo2(self.om, self.mm, capture_square, self.counts[0], self.color)

  def encode(self):
    if self.color:
      cm = torch.ones(1, 8, 8)
    else:
      cm = torch.zeros(1, 8, 8)
    if self.omp is None:
      omp = self.om
    else:
      omp = self.omp
    if self.mmp is None:
      mmp = self.mm
    else:
      mmp = self.mmp
    m = torch.cat([omp, mmp, self.om, self.mm, cm], dim=0)
    return m

  @staticmethod
  def dimension():
    return RCStateEncoder3.dim

# Policy Gradient Model action encoder/decoder using 1792 move dim
class RCActionEncoder3:
  dim = MOVE_MAP_SIZE_TOTAL

  def __init__(self):
    self.color = None

  def init(self, my_color: Color, _):
    self.color = my_color

  def encode(self, move: Move):
    m = torch.zeros(self.dim)
    d = encode_move_type_dim3(move, self.color)
    m[d] = 1.
    return m

  def decode(self, m):
    actions = []
    for i in range(m.shape[0]):
      action_idx = np.random.choice(self.dim, p=m[i].numpy())
      move = decode_move_dim3(action_idx, self.color)
      actions.append(move)
    return actions

  def action_index(self, move: Move):
    return move_to_action_index3(move, self.color)

  @staticmethod
  def dimension():
    return RCActionEncoder3.dim
