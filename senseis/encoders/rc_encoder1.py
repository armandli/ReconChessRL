from typing import Optional, List, Tuple
from reconchess import Color
from chess import Board, Piece, Move, Square
import numpy as np
import torch
from .rc_encoder_util import encode_move_type_dim1, decode_move_dim1, move_to_action_index1, encode_initial_board2, update_state_oppo1, update_state_self1, update_sense1

class RCStateEncoder1:
  # 6 types of pieces per player, 8 by 8 board
  dim = (12, 8, 8)

  def __init__(self):
    self.om = None
    self.mm = None
    self.color = None
    self.counts = None

  def init(self, my_color: Color, board: Board):
    self.mm, self.om = encode_initial_board2(my_color, board)
    self.color = my_color
    # 0 is opponent, 1 is myself
    self.counts = [16, 16]

  def sense_update(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    self.om = update_sense1(self.om, sense_result)

  def move_update(self, taken_move: Optional[Move], captured_square: Optional[Square]):
    if captured_square is not None:
      self.counts[0] -= 1
    self.om, self.mm = update_state_self1(self.om, self.mm, taken_move, captured_square)

  def op_move_update(self, capture_square: Optional[Square]):
    if capture_square is not None:
      self.counts[1] -= 1
    self.om, self.mm = update_state_oppo1(self.om, self.mm, capture_square, self.counts[0])

  def encode(self):
    m = torch.cat([self.om, self.mm], dim=0)
    return m

  @staticmethod
  def dimension():
    return RCStateEncoder1.dim

class RCSenseEncoder1:
  # 64 sense actions, one per each square
  dim = 64

  def __init__(self):
    pass

  def encode(self, action: Square):
    m = torch.zeros(self.dim)
    m[action] = 1.
    return m

  def decode(self, m):
    actions = []
    for i in range(m.shape[0]):
      action_idx = np.random.choice(self.dim, p=m[i].numpy())
      actions.append(action_idx)
    return actions

  @staticmethod
  def dimension():
    return RCSenseEncoder1.dim

# 4096 moves, in 8 x 8 x 64 = 4096 dimensions
# 8 x 8 describes the location of the piece
# 56 planes encode possible queen moves, a number [1..7] in which the piece
# moves in directions {N, NE, E, SE, S, SW, W, NW}
# 8 planes encode knight moves
# we ignore underpromotions and promotions
class RCActionEncoder1:
  dim = 4096

  def __init__(self):
    pass

  def encode(self, move: Move):
    m = torch.zeros(self.dim)
    d = encode_move_type_dim1(move)
    m[move.from_square * d] = 1.
    return m

  def decode(self, m):
    max_idx = torch.argmax(m, dim=1)
    max_idx = max_idx.numpy().tolist()
    actions = []
    for idx in max_idx:
      move_square = idx // 64
      move_type = idx % 64
      move = decode_move_dim1(move_square, move_type)
      actions.append(move)
    return actions

  def action_index(self, move: Move):
    return move_to_action_index1(move)

  @staticmethod
  def dimension():
    return RCActionEncoder1.dim
