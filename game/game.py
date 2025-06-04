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

    def human_move(self, source_sq, target_sq, promotion_piece=None):
        """
        source_sq, target_sq: int (0-63)
        promotion_piece: None hoặc một ký tự 'q','r','b','n' (lowercase) tương ứng Queen, Rook, Bishop, Knight
        Trả về:
          - (True, None) nếu đã push thành công (nếu không cần promotion hoặc promotion_piece đã có)
          - (False, 'PROMOTION') nếu cần phải chọn phong cấp (pawn vừa chạm hàng cuối nhưng promotion_piece=None)
          - (False, None) nếu nước không hợp lệ
        """
        # Xác định xem có phải quân Pawn đang di chuyển tới hàng cuối không
        piece = self.board.piece_at(source_sq)
        if piece and piece.piece_type == chess.PAWN:
            # Tìm rank (0..7) của target_sq
            rank = chess.square_rank(target_sq)
            # White: nếu đến rank 7 → promotion; Black: nếu đến rank 0 → promotion
            if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
                # Nếu chưa có promotion_piece, báo cần chọn loại
                if promotion_piece is None:
                    return False, 'PROMOTION'
                else:
                    # Tạo move với promotion
                    promo_map = {
                        'q': chess.QUEEN,
                        'r': chess.ROOK,
                        'b': chess.BISHOP,
                        'n': chess.KNIGHT
                    }
                    if promotion_piece.lower() in promo_map:
                        move = chess.Move(source_sq, target_sq, promotion=promo_map[promotion_piece.lower()])
                    else:
                        # Nếu promotion_piece không hợp lệ, trả False
                        return False, None

                    if move in self.board.legal_moves:
                        self.board.push(move)
                        return True, None
                    else:
                        return False, None

        # Nếu không phải promotion case, tạo move bình thường
        move = chess.Move(source_sq, target_sq)
        if move in self.board.legal_moves:
            self.board.push(move)
            return True, None
        else:
            return False, None
