from typing import Optional, List, Tuple
from chess import Board, Piece, Move, Square
from reconchess import Color

def rc_action_reward1(self_capture_count, oppo_capture_count, terminal, win=False):
  r = 0. + self_capture_count - oppo_capture_count
  if terminal:
    r += 40. if win else -40
  return r

def piece_count(board: Board, my_color: Color):
  my_count = 0
  op_count = 0
  for i in range(64):
    p = board.piece_at(i)
    if p is not None:
      if p.color == my_color:
        my_count += 1
      else:
        op_count += 1
  return (my_count, op_count)

def rc_action_reward2(is_terminal: bool, prev_board: Board = None, next_board: Board = None, winner: Color = None, my_color: Color = None):
  if is_terminal:
    if winner == my_color:
      return 1.0
    else:
      return -1.0
  else:
    # opponent piece count is unreliable, but my piece count is, so intermediate reward is best to be negative
    my_prev_count, op_prev_count = piece_count(prev_board, my_color)
    my_next_count, op_next_count = piece_count(next_board, my_color)
    value = (my_next_count - my_prev_count) * 0.0625
    return value
