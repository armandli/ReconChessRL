from typing import Optional, List, Tuple
import os
from copy import copy
import random
from reconchess import Color, Player, Square, WinReason, GameHistory
from chess import Board, Piece, Move
from chess import engine
import torch

# Stockfish immitation agent using Dagger, changing encoder and model to RNN
# compatible with RCStateEncoder5, RCActionModel4, RCSenseModel2

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'

class RCDaggerAgent2(Player):
  def __init__(
    self,
    state_encoder,
    action_encoder,
    sense_encoder,
    action_model,
    sense_model,
    device,
    choose_engine=True,
    action_exp=None,
    sense_exp=None,
    sense_reward=None,
  ):
    if STOCKFISH_ENV_VAR not in os.environ:
      raise KeyError("Missing stockfish executable definition in environement for variable {}".format(STOCKFISH_ENV_VAR))
    stockfish_path = os.environ[STOCKFISH_ENV_VAR]
    if not os.path.exists(stockfish_path):
      raise ValueError("No stockfish found at {}".format(stockfish_path))
    self.action_engine = engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)
    self.state_encoder = state_encoder
    self.action_encoder = action_encoder
    self.sense_encoder = sense_encoder
    self.action_model = action_model
    self.sense_model = sense_model
    self.action_exp = action_exp
    self.sense_exp = sense_exp
    self.sense_reward = sense_reward
    self.device = device
    self.color = None
    self.board = None
    self.sense_hvec = None
    self.action_hvec = None
    self.choose_engine = choose_engine

  def handle_game_start(self, color: Color, board: Board, opponent_name: str):
    self.state_encoder.init(color, board)
    self.action_encoder.init(color, board)
    self.sense_encoder.init(color, board)
    self.color = color
    self.board = copy(board)
    self.sense_hvec = self.sense_model.init(1)
    self.sense_hvec = self.sense_hvec.to(self.device)
    self.action_hvec = self.action_model.init(1)
    self.action_hvec = self.action_hvec.to(self.device)

  def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
    self.state_encoder.op_move_update(capture_square)
    if captured_my_piece:
      self.board.remove_piece_at(capture_square)

  def choose_sense(self, sense_action: List[Square], move_action: List[Move], seconds_left: float) -> Optional[Square]:
    with torch.no_grad():
      cst = self.state_encoder.encode_sense()
      cst_dev = cst.unsqueeze(0).unsqueeze(0).to(self.device)
      act, h = self.sense_model(cst_dev, self.sense_hvec)
      self.sense_hvec = h
      actions = self.sense_encoder.decode(act)
      action = actions[0][0]
      action_idxes = self.sense_encoder.action_index(actions[0])
      action_index = action_idxes[0]
      if self.sense_exp is not None:
        self.sense_exp.append_st(cst, action_index)
      return action

  def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[Piece]]]):
    prev_board = copy(self.state_encoder.board)
    self.state_encoder.sense_update(sense_result)
    next_board = self.state_encoder.board
    for square, piece in sense_result:
      self.board.set_piece_at(square, piece)
    if self.sense_exp is not None:
      reward = self.sense_reward(prev_board, next_board, self.color)
      self.sense_exp.append_post(reward)

  def choose_move(self, move_action: List[Move], seconds_left: float) -> Optional[Move]:
    engine_move = self._choose_engine_move(move_action, seconds_left)
    model_move = self._choose_model_move(seconds_left)
    # we assume engine_move is always non-none, could be a bad assumption
    if self.action_exp is not None:
      cst = self.state_encoder.encode_action()
      #TODO: what if engine action is none?
      action_idx = self.action_encoder.action_index(engine_move)
      self.action_exp.append_st(cst, action_idx)
    if self.choose_engine:
      action = engine_move
    else:
      action = model_move
    if action in move_action:
      return action
    action = random.choice(move_action)
    return action

  def _choose_engine_move(self, move_action: List[Move], seconds_left: float) -> Optional[Move]:
    enemy_king_square = self.board.king(not self.color)
    if enemy_king_square:
      enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
      if enemy_king_attackers:
        attacker_square = enemy_king_attackers.pop()
        action = Move(attacker_square, enemy_king_square)
        return action
    try:
      self.board.turn = self.color
      self.board.clear_stack()
      result = self.action_engine.play(self.board, engine.Limit(time=0.5)) #TODO: use seconds left ?
      action = result.move
      return action
    except engine.EngineTerminatedError as e:
      print("stockfish died: {}".format(e))
      print("board\n{}".format(self.board))
      stockfish_path = os.environ[STOCKFISH_ENV_VAR]
      self.action_engine = engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)
    except engine.EngineError as e:
      print("stockfish engine bad state at {}".format(self.board.fen()))
      print("board\n{}".format(self.board))
      stockfish_path = os.environ[STOCKFISH_ENV_VAR]
      self.action_engine = engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)
    action = random.choice(move_action)
    return action

  def _choose_model_move(self, seconds_left: float) -> Optional[Move]:
    with torch.no_grad():
      cst = self.state_encoder.encode_action()
      cst_dev = cst.unsqueeze(0).unsqueeze(0).to(self.device)
      act, h = self.action_model(cst_dev, self.action_hvec)
      self.action_hvec = h
      action = self.action_encoder.decode(act)[0][0]
      return action

  def handle_move_result(self, requested_move: Optional[Move], taken_move: Optional[Move], captured_opponent_piece: bool, capture_square: Optional[Square]):
    self.state_encoder.move_update(taken_move, capture_square)
    if taken_move is not None:
      piece = self.board.piece_at(taken_move.from_square)
      if piece is not None:
        self.board.remove_piece_at(taken_move.from_square)
        self.board.set_piece_at(taken_move.to_square, piece)

  def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
    try:
      self.action_engine.quit()
    except engine.EngineTerminatedError:
      pass
