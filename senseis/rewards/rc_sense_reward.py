from typing import Optional, List, Tuple
from reconchess import Color
from chess import Board, Piece, Move, Square
import torch

def rc_sense_reward1(oma, omb):
  v = torch.sum(torch.abs(oma - omb))
  return v

def rc_sense_reward2(a: Board, b: Board, my_color: Color):
  s = 0.
  for i in range(64):
    pa = a.piece_at(i)
    pb = b.piece_at(i)
    if pa is not None and pb is None:
      if pa.color != my_color:
        s += 1.
    elif pa is None and pb is not None:
      if pb.color != my_color:
        s += 1.
  return s
