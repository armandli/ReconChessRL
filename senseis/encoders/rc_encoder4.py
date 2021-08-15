from typing import Optional, List, Tuple
from reconchess import Color
from chess import Board, Piece, Move, Square
import numpy as np
import torch
from .rc_encoder_util import encode_initial_board3, update_state_oppo3, update_state_self2, update_sense2

# more primitive version for the sense model, we don't include any
# board decay or speculative modifications
class RCStateEncoder4:
  dim=(13, 8, 8)

  def __init__(self):
    self.om = None
    self.mm = None
    self.color = None

  def init(self, my_color: Color, board: Board):
    self.om, self.mm = encode_initial_board3(my_color, board)
    self.color = my_color

  def sense_update(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    self.om = update_sense2(self.om, sense_result, self.color)

  def move_update(self, taken_move: Optional[Move], captured_square: Optional[Square]):
    self.om, self.mm = update_state_self2(self.om, self.mm, taken_move, captured_square, self.color)

  def op_move_update(self, capture_square: Optional[Square]):
    self.om, self.mm = update_state_oppo3(self.om, self.mm, capture_square, self.color)

  def encode(self):
    if self.color:
      cm = torch.ones(1, 8, 8)
    else:
      cm = torch.zeros(1, 8, 8)
    m = torch.cat([self.om, self.mm, cm], dim=0)
    return m

  @staticmethod
  def dimension():
    return RCStateEncoder4.dim
