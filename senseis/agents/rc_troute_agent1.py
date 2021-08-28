from typing import Optional, List, Tuple
import os
from copy import copy
from reconchess import Color, Player, Square, WinReason, GameHistory
from chess import Board, Piece, Move
from chess import engine
import torch

# Runs Stockfish agent to learn the best sensing action

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'

class RCTrouteAgent1(Player):
  def __init__(
    self,
    state_encoder,
    sense_encoder,
    sense_model,
    device,
    sense_exp=None,
    sense_reward=None
  ):
    if STOCKFISH_ENV_VAR not in os.environ:
      raise KeyError("Missing stockfish executable definition in environement for variable {}".format(STOCKFISH_ENV_VAR))
    stockfish_path = os.environ[STOCKFISH_ENV_VAR]
    if not os.path.exists(stockfish_path):
      raise ValueError("No stockfish found at {}".format(stockfish_path))
    self.action_engine = engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)
    self.state_encoder = state_encoder
    self.sense_encoder = sense_encoder
    self.sense_model = sense_model
    self.sense_exp = sense_exp
    self.sense_reward = sense_reward
    self.device = device
    self.self_capture_count = None
    self.oppo_capture_count = None
    self.color = None
    self.sense_cst = None
    self.board = None

  def handle_game_start(self, color: Color, board: Board, opponent_name: str):
    self.state_encoder.init(color, board)
    self.sense_encoder.init(color, board)
    self.color = color
    self.self_capture_count = 0
    self.oppo_capture_count = 0
    self.board = copy(board)

  def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
    self.state_encoder.op_move_update(capture_square)
    if captured_my_piece:
      self.oppo_capture_count += 1
      self.board.remove_piece_at(capture_square)

  def choose_sense(self, sense_action: List[Square], move_action: List[Move], seconds_left: float) -> Optional[Square]:
    with torch.no_grad():
      cst = self.state_encoder.encode()
      self.sense_cst = cst
      cst_dev = cst.unsqueeze(0).to(self.device)
      act = self.sense_model(cst_dev)
      action = self.sense_encoder.decode(act)[0]
      if self.sense_exp is not None:
        self.sense_exp.append_st(cst, action)
      return action

  def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    self.state_encoder.sense_update(sense_result)
    for square, piece in sense_result:
      self.board.set_piece_at(square, piece)
    if self.sense_exp is not None:
      nst = self.state_encoder.encode()
      reward = self.sense_reward(self.sense_cst, nst)
      self.sense_exp.append_post(reward)

  def choose_move(self, move_action: List[Move], seconds_left: float) -> Optional[Move]:
    enemy_king_square = self.board.king(not self.color)
    if enemy_king_square:
      enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
      if enemy_king_attackers:
        attacker_square = enemy_king_attackers.pop()
        action = Move(attacker_square, enemy_king_square)
        if action in move_action:
          return action
    try:
      self.board.turn = self.color
      self.board.clear_stack()
      result = self.action_engine.play(self.board, engine.Limit(time=0.5)) #TODO: use seconds left ?
      action = result.move
      if action in move_action:
        return action
    except engine.EngineTerminatedError as e:
      print("stockfish died: {}".format(e))
    except engine.EngineError as e:
      print("stockfish engine bad state at {}".format(self.board.fen()))
    return None

  def handle_move_result(self, requested_move: Optional[Move], taken_move: Optional[Move], captured_opponent_piece: bool, capture_square: Optional[Square]):
    if captured_opponent_piece:
      self.self_capture_count += 1
    self.state_encoder.move_update(taken_move, capture_square)
    if taken_move is not None:
      piece = self.board.piece_at(taken_move.from_square)
      self.board.remove_piece_at(taken_move.from_square)
      self.board.set_piece_at(taken_move.to_square, piece)

  def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
    try:
      self.action_engine.quit()
    except engine.EngineTerminatedError:
      pass
