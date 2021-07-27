from typing import List, Tuple, Optional
from reconchess import Color
from chess import Board, Piece, Move, Square
import torch

# board positions displayed
# 63 62 .. 58 57
# .     .     .
# .      .    .
# 7  6  .. 1  0

def encode_initial_board1(my_color: Color, board: Board):
  m = torch.zeros(12, 8, 8)
  piece_map = board.piece_map()
  for (square, piece) in piece_map.items():
    # P = 1 N = 2 B = 3 R = 4 K = 5 Q = 6
    dim = (piece.piece_type - 1) + int(piece.color == my_color) * 6
    m[dim, square // 8, square % 8] = 1.
  return m

def encode_initial_board2(my_color: Color, board: Board):
  om = torch.zeros(6, 8, 8)
  mm = torch.zeros(6, 8, 8)
  piece_map = board.piece_map()
  for (square, piece) in piece_map.items():
    # P = 1 N = 2 B = 3 R = 4 K = 5 Q = 6
    dim = (piece.piece_type - 1)
    if piece.color == my_color:
      mm[dim, square // 8, square % 8] = 1.
    else:
      om[dim, square // 8, square % 8] = 1.
  return (om, mm)

# simply decay the prob. of opponent pieces, not following any game
# rules
def board_decay1(om, mm, epsilon):
  om = torch.where(mm < 1., om + epsilon, om * 0.)
  om = om * (1. - epsilon)
  return om

def update_state_oppo1(om, mm, capture: Optional[Square], oppo_piece_count):
  if capture is not None:
    cx, cy = capture // 8, capture % 8
    mm[:,cx,cy] = 0.
    om[1:,cx,cy] = 1.
    om[0,cx,cy] = 0.5 # pawn special
    return (om, mm)
  else:
    om = board_decay1(om, mm, 1. / oppo_piece_count / 64.)
    return (om, mm)

# determine the row diff and col diff per path square update
def move_step(rowd, cold):
  # knight move
  if (abs(rowd) == 1 and abs(cold) == 2) or (abs(rowd) == 2 and abs(cold) == 1):
    return rowd, cold
  if rowd > 0:
    rowd = 1
  elif rowd < 0:
    rowd = -1
  if cold > 0:
    cold = 1
  elif cold < 0:
    cold = -1
  return rowd, cold

# determine the number of squares that can be updated between the move
def move_squares(rowd, cold):
  # knight move
  if (abs(rowd) == 1 and abs(cold) == 2) or (abs(rowd) == 2 and abs(cold) == 1):
    return 1
  if rowd == cold or cold == 0:
    return rowd
  else:
    return cold

def clear_oppo_prob(om, move: Move):
  frow, fcol = move.from_square // 8, move.from_square % 8
  trow, tcol = move.to_square // 8, move.to_square % 8
  rowd, cold = trow - frow, tcol - fcol
  srow, scol = move_step(rowd, cold)
  n = move_squares(rowd, cold)
  for i in range(n):
    x, y = frow + i * srow, fcol + i * scol
    om[0:6, x, y] = 0.
  return om

def update_state_self1(om, mm, move: Move, capture: Optional[Square]):
  if move is None:
    return (om, mm)
  if capture is not None:
    om = clear_oppo_prob(om, move)
  fx, fy = move.from_square // 8, move.from_square % 8
  tx, ty = move.to_square // 8, move.to_square % 8
  mm[6:12, tx, ty] = mm[6:12, fx, fy]
  mm[6:12, fx, fy] = 0.
  # if the piece is a pawn and pawn gets promoted
  if move.promotion is not None:
    mm[6, tx, ty] = 0.
    mm[5 + move.promotion, tx, ty] = 1.
  return (om, mm)

# if a piece is detected, its prob on om is set to 1, otherwise 0 for the
# sensed squares
def update_sense1(om, sense_result: List[Tuple[Square, Optional[Piece]]]):
  for square, piece in sense_result:
    x, y = square // 8, square % 8
    if piece is None:
      om[:,x,y] = 0.
    else:
        om[:,x,y] = 0.
        om[piece.piece_type-1,x,y] = 1.
  return om

MOVE_MAP = {
  # N
  (1, 0): 0,
  (2, 0): 1,
  (3, 0): 2,
  (4, 0): 3,
  (5, 0): 4,
  (6, 0): 5,
  (7, 0): 6,
  # NE
  (1, -1): 7,
  (2, -2): 8,
  (3, -3): 9,
  (4, -4): 10,
  (5, -5): 11,
  (6, -6): 12,
  (7, -7): 13,
  # E
  (0, -1): 14,
  (0, -2): 15,
  (0, -3): 16,
  (0, -4): 17,
  (0, -5): 18,
  (0, -6): 19,
  (0, -7): 20,
  # SE
  (-1, -1): 21,
  (-2, -2): 22,
  (-3, -3): 23,
  (-4, -4): 24,
  (-5, -5): 25,
  (-6, -6): 26,
  (-7, -7): 27,
  # S
  (-1, 0): 28,
  (-2, 0): 29,
  (-3, 0): 30,
  (-4, 0): 31,
  (-5, 0): 32,
  (-6, 0): 33,
  (-7, 0): 34,
  # SW
  (-1, 1): 35,
  (-2, 2): 36,
  (-3, 3): 37,
  (-4, 4): 38,
  (-5, 5): 39,
  (-6, 6): 40,
  (-7, 7): 41,
  # W
  (0, 1): 42,
  (0, 2): 43,
  (0, 3): 44,
  (0, 4): 45,
  (0, 5): 46,
  (0, 6): 47,
  (0, 7): 48,
  # NW
  (1, 1): 49,
  (2, 2): 50,
  (3, 3): 51,
  (4, 4): 52,
  (5, 5): 53,
  (6, 6): 54,
  (7, 7): 55,
  # Knight Moves
  ( 2, -1): 56,
  ( 1, -2): 57,
  (-1, -2): 58,
  (-2, -1): 59,
  (-2,  1): 60,
  (-1,  2): 61,
  ( 1,  2): 62,
  ( 2,  1): 63,
}

INV_MOVE_MAP = {
  # N
  0:  (1, 0),
  1:  (2, 0),
  2:  (3, 0),
  3:  (4, 0),
  4:  (5, 0),
  5:  (6, 0),
  6:  (7, 0),
  # NE
  7:  (1, -1),
  8:  (2, -2),
  9:  (3, -3),
  10: (4, -4),
  11: (5, -5),
  12: (6, -6),
  13: (7, -7),
  # E
  14: (0, -1),
  15: (0, -2),
  16: (0, -3),
  17: (0, -4),
  18: (0, -5),
  19: (0, -6),
  20: (0, -7),
  # SE
  21: (-1, -1),
  22: (-2, -2),
  23: (-3, -3),
  24: (-4, -4),
  25: (-5, -5),
  26: (-6, -6),
  27: (-7, -7),
  # S
  28: (-1, 0),
  29: (-2, 0),
  30: (-3, 0),
  31: (-4, 0),
  32: (-5, 0),
  33: (-6, 0),
  34: (-7, 0),
  # SW
  35: (-1, 1),
  36: (-2, 2),
  37: (-3, 3),
  38: (-4, 4),
  39: (-5, 5),
  40: (-6, 6),
  41: (-7, 7),
  # W
  42: (0, 1),
  43: (0, 2),
  44: (0, 3),
  45: (0, 4),
  46: (0, 5),
  47: (0, 6),
  48: (0, 7),
  # NW
  49: (1, 1),
  50: (2, 2),
  51: (3, 3),
  52: (4, 4),
  53: (5, 5),
  54: (6, 6),
  55: (7, 7),
  # Knight Moves
  56: ( 2, -1),
  57: ( 1, -2),
  58: (-1, -2),
  59: (-2, -1),
  60: (-2,  1),
  61: (-1,  2),
  62: ( 1,  2),
  63: ( 2,  1),
}

def encode_move_type_dim1(move: Move):
  frow, fcol = move.from_square // 8, move.from_square % 8
  trow, tcol = move.to_square // 8, move.to_square % 8
  rowd, cold = trow - frow, tcol - fcol
  move_dim = MOVE_MAP[(rowd, cold)]
  return move_dim

def decode_move_dim1(from_square, move_type):
  x,  y  = from_square // 8, from_square % 8
  dx, dy = INV_MOVE_MAP[move_type]
  to_square = (x + dx) * 8 + y + dy
  # invalid move, return None
  if to_square <= 0 or to_square >= 64:
    return None
  #ignore underpromotion and promotion
  return Move(from_square, to_square)

def move_to_action_index1(move: Move):
  frow, fcol = move.from_square // 8, move.from_square % 8
  trow, tcol = move.to_square // 8, move.to_square % 8
  rowd, cold = trow - frow, tcol - fcol
  move_dim = MOVE_MAP[(rowd, cold)]
  return move.from_square * 64 + move_dim

##TODO: what if they are invalid moves
#def pawn_init_move_t(pc):
#  m = torch.zeros(5, 3)
#  m[2,1] = 1. - 1. / pc
#  m[3,0] = m[3,2] = m[3,1] = m[4,1] = 1. / pc / 4.
#  return m
#
#def pawn_move_t(pc):
#  m = torch.zeros(3, 3)
#  m[1,1] = 1. - 1. / pc
#  m[2,0] = m[2,1] = m[2,2] = 1. / pc / 3.
#  return m
#
#def knight_move_t(pc):
#  m = torch.zeros(5, 5)
#  m[2,2] = 1. - 1. / pc
#  m[0,1] = m[0,3] = m[1,0] = m[1,4] = m[3,0] = m[3,4] = m[4,1] = m[4,3] = 1. / pc / 8.
#  return m
#
#def bishop_move_t(pc):
#  m = torch.zeros(15, 15)
#  m[7,7] = 1. - 1. / pc
#  for i in range(7):
#    m[i,i] = m[i+8,i+8] = m[14-i,i] = m[6-i,i+8] = 1. / pc / 28.
#  return m
#
#def rook_move_t(pc):
#  m = torch.zeros(15, 15)
#  m[7,7] = 1. - 1. / pc
#  for i in range(7):
#    m[7,i] = m[7,i+8] = m[i,7] = m[i+8,7] = 1. / pc / 28.
#  return m
#
#def queen_move_t(pc):
#  m = torch.zeros(15, 15)
#  m[7,7] = 1. - 1. / pc
#  for i in range(7):
#    m[7,i] = m[7,i+8] = m[i,7] = m[i+8,7] = m[i,i] = m[i+8,i+8] = m[14-i,i] = m[6-i,i+8] = 1. / pc / 56.
#  return m
#
#def king_move_t(pc):
#  m = torch.zeros(3, 3)
#  for i in range(3):
#    for j in range(3):
#      m[i,j] = 1. / pc / 8.
#  m[1,1] = 1. - 1. / pc
#  return m
#
#def move_t(pc):
#  #7 layers, 2 for pawn, 1 for each other piece
#  m = torch.zeros(7, 15, 15)
#  m[:,7,7] = 1. - 1. / pc
#  #pawn init
#  m[0,8,6] = m[0,8,7] = m[0,8,8] = m[0,9,7] = 1. / pc / 4.
#  #pawn
#  m[1,8,6] = m[1,8,7] = m[1,8,8] = 1. / pc / 3.
#  #knight
#  m[2,5,6] = m[2,5,8] = m[2,6,5] = m[2,6,9] = m[2,8,5] = m[2,8,9] = m[2,9,6] = m[2,9,8] = 1. / pc / 8.
#  for i in range(7):
#    #bishop
#    m[3,i,i] = m[3,i+8,i+8] = m[3,14-i,i] = m[3,6-i,i+8] = 1. / pc / 28.
#    #rook
#    m[4,7,i] = m[4,7,i+8] = m[4,i,7] = m[4,i+8,7] = 1. / pc / 28.
#    #queen
#    m[6,i,i] = m[6,i+8,i+8] = m[6,14-i,i] = m[6,6-i,i+8] = m[6,7,i] = m[6,7,i+8] = m[6,i,7] = m[6,i+8,7] = 1. / pc / 56.
#  #king
#  for i in range(3):
#    for j in range(3):
#      m[5,i+6,j+6] = 1. / pc / 8.
#  m[5,7,7] = 1. - 1 / pc
#  return m
#
#def six_to_seven_dim(m):
#  n = torch.zeros(7, 8, 8)
#  n[1:,:,:] = m[0:,:,:]
#  n[0,1,:] = m[0,1,:]
#  n[1,1,:] = 0.
#  return n
#
#def seven_to_six_dim(m):
#  n = torch.zeros(6, 8, 8)
#  n[1:,:,:] = m[2:,:,:]
#  n[0,:,:] = m[0,:,:] + m[1,:,:]
#  return n
#
## update state after opponent makes a move
## oppo. piece prob. update:
##   if my piece is captured,
##     gather prob. of oppo. piece to 1, but not knowing what type of the piece, note on pawn capture
##   if no piece is captured,
##     do one step prob. difusion based on game rule: prob. of choosing a piece at a square * prob. of piece on square * prob. of taking the specific move
#def update_state_oppo(mm, om, capture: Optional[Square], piece_count):
#  # remove my piece if it is captured
#  if capture is not None:
#    cx, cy = capture // 8, capture % 8
#    mm[6:12, cx, cy] = 0.
#    #TODO: if there is a capture, then there is definitely an opponent piece there
#  else:
#    #TODO: is this correct? or we should do inverse conv2d
#    s = six_to_seven_dim(om)
#    step_mtx = move_t(piece_count)
#    u = F.conv2d(s, step_mtx, None, 1, 7, 1, 1)
#    return (mm, u)
#
## determine the row diff and col diff per path square update
#def move_step(rowd, cold):
#  # knight move
#  if (abs(rowd) == 1 and abs(cold) == 2) or (abs(rowd) == 2 and abs(cold) == 1):
#    return rowd, cold
#  if rowd > 0:
#    rowd = 1
#  elif rowd < 0:
#    rowd = -1
#  if cold > 0:
#    cold = 1
#  elif cold < 0:
#    cold = -1
#  return rowd, cold
#
## determine the number of squares that can be updated between the move
#def move_squares(rowd, cold):
#  # knight move
#  if (abs(rowd) == 1 and abs(cold) == 2) or (abs(rowd) == 2 and abs(cold) == 1):
#    return 1
#  if rowd == cold or cold == 0:
#    return rowd
#  else:
#    return cold
#
#def clear_oppo_prob(m, move: Move):
#  frow, fcol = move.from_square // 8, move.from_square % 8
#  trow, tcol = move.to_square // 8, move.to_square % 8
#  rowd, cold = trow - frow, tcol - fcol
#  srow, scol = move_step(rowd, cold)
#  n = move_squares(rowd, cold)
#  for i in range(n):
#    x, y = frow + i * srow, fcol + i * scol
#    m[0:6, x, y] = 0.
#
## uniformly add some prob. to all types based on sm into om where om is > 0. and < 1.
#def redistribute_oppo_square_prob_all_types(om, sm):
#  #TODO: 1) count the number of cells in om > 0. and < 1. for each dim
#  #TODO: 2) add a prob. porportional to each cell where it is > 0. and < 1.
#  pass
#
## transfer the probability of the piece to all existing locations where the probability is > 0 evenly,
## also avoid transfer to where it's already 1
#def redistribute_oppo_prob(om, move: Move):
#  frow, fcol = move.from_square // 8, move.from_square % 8
#  trow, tcol = move.to_square // 8, move.to_square % 8
#  rowd, cold = trow - frow, tcol - fcol
#  srow, scol = move_step(rowd, cold)
#  n = move_squares(rowd, cold)
#  sm = torch.zeros(6, 1, 1)
#  for i in range(n):
#    x, y = frow + i * srow, fcol + i * scol
#    sm += m[0:6, x, y]
#    om[0:6, x, y] = 0.
#  redistribute_oppo_square_prob_all_types(om, sm)
#  return om
#
#  for i in range(n):
#    x, y = frow + i * srow, fcol + i * scol
#    sm += m[0:6, x, y]
#    m[0:6, x, y] = 0.
#  #TODO: redistribute sm back to the rest of the tensor in m
#
## update state after agent makes a move
## oppo. prob. dist. update:
## if capture,
##   1) reduce the capture square oppo. piece prob. to 0, for all piece types
##   2) intermediate squares are also all reduced to 0, transfer the prob. to other squares uniformly
## else do nothing
#def update_state_self(mm, om, move: Move, capture: Optional[Square]):
#  if capture is not None:
#    om = redistribute_oppo_prob(om, move)
#  fx, fy = move.from_square // 8, move.from_square % 8
#  tx, ty = move.to_square // 8, move.to_square % 8
#  mm[6:12, tx, ty] = mm[6:12, fx, fy]
#  mm[6:12, fx, fy] = 0.
#  # if the piece is a pawn and pawn gets promoted
#  if move.promotion is not None:
#    mm[6, tx, ty] = 0.
#    mm[5 + move.promotion, tx, ty] = 1.
#  return (mm, om)
#
## update state after sense
## oppo. prob. dist. update:
##   1) gather prob. of oppo. piece to 1 for the specific piece type, reduce the difference prob. of oppo. piece type in other squares uniformly
##   2) reduce the prob. of oppo. piece to 0 for non-detected squres, transfer the prob. to other squares of the same piece type uniformly
#def update_state_sense(m, sense_result: List[Tuple[Square, Optional[Piece]]]):
#  #TODO: are we also getting my own pieces from the sense result?
#  pass
