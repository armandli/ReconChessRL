from typing import Optional, List, Tuple
from copy import copy
from reconchess import Color
from chess import Board, Piece, Move, Square
import numpy as np
import torch
from .rc_encoder_util import encode_self_action1, encode_oppo_result1, encode_sense_result1, encode_null_self_move1, encode_null_oppo_move1, encode_null_sense1, encode_move_type_dim3, decode_move_dim3, move_to_action_index3, SELF_ACTION_EVENT_DIM1, OPPO_ACTION_EVENT_DIM1, SENSE_EVENT_DIM1, MOVE_MAP_SIZE_TOTAL, is_valid_square_for_sense_idx, square_to_sense_idx, sense_idx_to_square

# instead of encoding the board, we'll focus encoding different types of events that has been detected by the agent
# types of events:
#   * initialization
#   * player move (both capture and no capture)
#   * opponent move (both capture and no capture)
#   * sense result

# different encoding will be used for sense model and action model, for sense model, the encoding includes
#   player color (1) + player move result ((64 + 64) * 6 + 1 + 1) + opponent move result (64 + 1 + 1)
# for action model,
#   player color (1) + opponent move result (64 + 1 + 1) + sense result (9 * 64 * (6 + 1))
# all results need a state representing none
# player move result is encoding the move from 3 dim into 1 dim, also a flag indicating capture
# opponent move result is encoding whether there is opponent capture, if capture the square location of capture in 2 dim encoded as 1 dim
# sense result is encoding the 9 squares with information discovered, which is 9 * 2 dim into 1 dim, also including the piece type
# encoded event dimension for sense model and action model will be different
class RCStateEncoder5:
  sense_dim  = 1 + SELF_ACTION_EVENT_DIM1 + OPPO_ACTION_EVENT_DIM1
  action_dim = 1 + OPPO_ACTION_EVENT_DIM1 + SENSE_EVENT_DIM1

  def __init__(self):
    self.board     = None
    self.color     = None
    self.color_vec = None
    self.self_move = encode_null_self_move1()
    self.oppo_move = encode_null_oppo_move1()
    self.sense     = encode_null_sense1()

  # generate the init hidden vector for the model instead
  def init(self, my_color: Color, board: Board):
    self.board = copy(board)
    self.color = my_color
    if my_color:
      self.color_vec = torch.ones(1)
    else:
      self.color_vec = torch.zeros(1)

  def sense_update(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    self.sense = encode_sense_result1(sense_result, self.color)
    for square, piece in sense_result:
      self.board.set_piece_at(square, piece)

  def move_update(self, taken_move: Optional[Move], captured_square: Optional[Square]):
    self.self_move = encode_self_action1(self.board, taken_move, captured_square, self.color)
    if taken_move is not None:
      piece = self.board.piece_at(taken_move.from_square)
      self.board.remove_piece_at(taken_move.from_square)
      self.board.set_piece_at(taken_move.to_square, piece)

  def op_move_update(self, captured_square: Optional[Square]):
    self.oppo_move = encode_oppo_result1(captured_square, self.color)
    if captured_square is not None:
      self.board.remove_piece_at(captured_square)

  def encode_sense(self):
    m = torch.cat([self.color_vec, self.self_move, self.oppo_move], dim=0)
    return m

  def encode_action(self):
    m = torch.cat([self.color_vec, self.oppo_move, self.sense], dim=0)
    return m

  @staticmethod
  def sense_dimension():
    return RCStateEncoder5.sense_dim

  @staticmethod
  def action_dimension():
    return RCStateEncoder5.action_dim

# revert square for black player, shrink number of actions to 64 - 28, not using boundary indicies
class RCSenseEncoder3:
  dim = 36

  def __init__(self):
    self.color = None

  def init(self, my_color: Color, _):
    self.color = my_color

  def encode(self, action: Square):
    if not self.color: # black
      action = 63 - action
    m = torch.zeros(self.dim)
    if is_valid_square_for_sense_idx(action):
      sense_idx = square_to_sense_idx(action)
      m[sense_idx] = 1.
    return m

  def decode(self, m):
    actions = []
    if not self.color: # black
      for i in range(m.shape[0]):
        sequence = []
        for j in range(m.shape[1]):
          midx = np.random.choice(self.dim, p=m[i,j].numpy())
          action_idx = sense_idx_to_square(midx)
          action_idx = 63 - action_idx
          sequence.append(action_idx)
        actions.append(sequence)
    else:
      for i in range(m.shape[0]):
        sequence = []
        for j in range(m.shape[1]):
          midx = np.random.choice(self.dim, p=m[i,j].numpy())
          action_idx = sense_idx_to_square(midx)
          sequence.append(action_idx)
        actions.append(sequence)
    return actions

  def action_index(self, actions: List[Square]):
    encoded = []
    for action in actions:
      if not self.color: # black
        action = 63 - action
      if is_valid_square_for_sense_idx(action):
        sense_idx = square_to_sense_idx(action)
        encoded.append(sense_idx)
    return encoded

  @staticmethod
  def dimension():
    return RCSenseEncoder3.dim

# Policy Gradient Model action encoder/decoder using 1792 move dim, expecting sequence of actions in dim (b, S, A)
class RCActionEncoder4:
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
      sequence = []
      for j in range(m.shape[1]):
        action_idx = np.random.choice(self.dim, p=m[i,j].numpy())
        move = decode_move_dim3(action_idx, self.color)
        sequence.append(move)
      actions.append(sequence)
    return actions

  def action_index(self, move: Move):
    return move_to_action_index3(move, self.color)

  @staticmethod
  def dimension():
    return RCActionEncoder4.dim
