from typing import Optional, List, Tuple
from reconchess import Color
from chess import Board, Piece, Move, Square
import numpy as np
import torch
from .rc_encoder_util import encode_move_type_dim3, decode_move_dim3, move_to_action_index3, encode_initial_board3, update_state_oppo2, update_state_self2, update_sense2, MOVE_MAP_SIZE_TOTAL

class RCStateEncoder2:
  # 6 types of pieces per player, 8 by 8 board, 1 dim for color
  dim = (13, 8, 8)

  def __init__(self):
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
    self.om, self.mm = update_state_self2(self.om, self.mm, taken_move, captured_square, self.color)

  def op_move_update(self, capture_square: Optional[Square]):
    if capture_square is not None:
      self.counts[1] -= 1
    self.om, self.mm = update_state_oppo2(self.om, self.mm, capture_square, self.counts[0], self.color)

  def encode(self):
    if self.color:
      cm = torch.ones(1, 8, 8)
    else:
      cm = torch.zeros(1, 8, 8)
    m = torch.cat([self.om, self.mm, cm], dim=0)
    return m

  @staticmethod
  def dimension():
    return RCStateEncoder2.dim

class RCSenseEncoder2:
  dim = 64

  def __init__(self):
    self.color = None

  def init(self, my_color: Color, _):
    self.color = my_color

  def encode(self, action: Square):
    if not self.color: # black
      action = 63 - action
    m = torch.zeros(self.dim)
    m[action] = 1.
    return m

  def decode(self, m):
    actions = []
    if not self.color: # black
      for i in range(m.shape[0]):
        action_idx = np.random.choice(self.dim, p=m[i].numpy())
        action_idx = 63 - action_idx
        actions.append(action_idx)
    else:
      for i in range(m.shape[0]):
        action_idx = np.random.choice(self.dim, p=m[i].numpy())
        actions.append(action_idx)
    return actions

  @staticmethod
  def dimension():
    return RCSenseEncoder2.dim

# 1792 moves, eliminated all moves that would go to invalid square
# this means the number of moves per square is different, and the move
# set per square is also different for each square; there are still
# invalid moves however, as the move piece can still limit what moves
# are valid
class RCActionEncoder2:
  dim = MOVE_MAP_SIZE_TOTAL

  def __init__(self):
    self.color = None

  def init(self, my_color: Color, _):
    self.color = my_color

  def encode(self, move: Move):
    m = torch.zeros(self.dim)
    d= encode_move_type_dim3(move, self.color)
    m[d] = 1.
    return m

  def decode(self, m):
    max_idx = torch.argmax(m, dim=1)
    max_idx = max_idx.numpy().tolist()
    actions = []
    for idx in max_idx:
      move = decode_move_dim3(idx, self.color)
      actions.append(move)
    return actions

  def action_index(self, move: Move):
    return move_to_action_index3(move, self.color)

  @staticmethod
  def dimension():
    return RCActionEncoder2.dim
