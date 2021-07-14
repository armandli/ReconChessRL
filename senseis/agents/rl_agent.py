from reconchess import Player, Color, Square, Optional, WinReason, GameHistory, List, Tuple
from reconchess import chess



#TODO: stub
class RLAgent(Player):
  def __init__(self):
    pass

  def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
    pass

  def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
    pass

  def choose_sense(self, sense_action: List[Square], move_action: List[chess.Move], seconds_left: float) -> Optional[Square]:
    pass

  def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
    pass

  def choose_move(self, move_action: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
    pass

  def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move], captured_opponent_piece: bool, capture_square: Optional[Square]):
    pass

  def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
    pass
