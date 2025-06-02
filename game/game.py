# game/game.py

import chess
from engines import greedy, minimax, negamax, mcts  # Đây là ví dụ import, có thể chọn động runtime
from game.constants import WHITE, BLACK


class Game:
    def __init__(self, choose_engine_white=None, choose_engine_black=None):
        # create a python-chess Board (khởi đầu chuẩn)
        self.board = chess.Board()

        # các engine (hàm) cho white và black, nếu None => human
        self.engine_white = choose_engine_white
        self.engine_black = choose_engine_black

    def reset(self):
        self.board.reset()

    def is_game_over(self):
        return self.board.is_game_over()

    def get_result(self):
        return self.board.result()  # Trả về "1-0", "0-1", "1/2-1/2"

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def apply_move(self, move):
        """
        Áp dụng move đã chắc chắn hợp lệ (move: kiểu chess.Move)
        """
        self.board.push(move)

    def engine_move(self):
        """
        Nếu đến lượt 1 trong 2 người chơi là AI, trả về nước đi do AI tính
        """
        if self.board.turn == chess.WHITE and self.engine_white:
            return self.engine_white(self.board.copy())  # Pass bản sao để bảo toàn trạng thái
        elif self.board.turn == chess.BLACK and self.engine_black:
            return self.engine_black(self.board.copy())
        else:
            return None

    def human_move(self, source_sq, target_sq):
        """
        source_sq, target_sq có thể là index 0-63 (hoặc 'e2', 'e4')
        Giả sử bạn dùng GUI để translate pixel -> sq.
        """
        move = chess.Move(source_sq, target_sq)
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False
