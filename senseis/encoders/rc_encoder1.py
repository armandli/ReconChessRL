from typing import Optional, List, Tuple
from reconchess import Color
from reconchess.chess import Board, Piece, Move
import numpy as np
import torch
from .rc_encoder_util import encode_move_type_dim1, decode_move_dim1, encode_initial_board2, update_state_oppo1, update_state_self1, update_sense1

class RCStateEncoder1:
  def __init__(self):
    self.om = None
    self.mm = None
    self.color = None
    self.counts = None
    # 6 types of pieces per player, 8 by 8 board
    self.dim = (12, 8, 8)

  def init(self, my_color: Color, board: Board):
    self.mm, self.om = encode_initial_board2(my_color, board)
    self.color = color
    # 0 is opponent, 1 is myself
    self.counts = [16, 16]

  def sense_update(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    self.om = update_sense1(self.om, sense_result)

  def move_update(self, taken_move: Optional[Move], captured_square: Optional[Square]):
    if capture_square is not None:
      self.counts[0] -= 1
    self.om, self.mm = update_state_self1(self.om, self.mm, taken_move, captured_square)

  def op_move_update(self, capture_square: Optional[Square]):
    if capture_square is not None:
      self.counts[1] -= 1
    self.om, self.mm = update_state_oppo1(self.om, self.mm, capture_square, self.counts[0])

  def encode(self):
    m = torch.cat([self.om, self.mm], dim=0)
    return m

  def dimension(self):
    return self.dim

class RCSenseEncoder1:
  def __init__(self):
    # 64 sense actions, one per each square
    self.dim = 64

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

  def dimension(self):
    return self.dim

# 4096 moves, in 8 x 8 x 64 = 4096 dimensions
# 8 x 8 describes the location of the piece
# 56 planes encode possible queen moves, a number [1..7] in which the piece
# moves in directions {N, NE, E, SE, S, SW, W, NW}
# 8 planes encode knight moves
# we ignore underpromotions and promotions
class RCActionEncoder1:
  def __init__(self):
    self.dim = 4096

  def encode(self, move: Move):
    m = torch.zeros(self.dim)
    dim = encode_move_type_dim(move)
    m[move.from_square * dim] = 1.
    return m

  def decode(self, m):
    max_idx = torch.argmax(m, dim=1)
    max_idx = max_idx.numpy().to_list()
    actions = []
    for idx in max_idx:
      move_square = idx // 64
      move_type = idx % 64
      move = decode_move_dim1(move_square, move_type)
      actions.append(move)
    return actions

  def dimension(self):
    return self.dim
