# engines/negamax.py

import chess
from engines.helpers import evaluate_board

INFINITY = 10_000

def negamax_ab(board: chess.Board, depth: int, alpha: int, beta: int, color: int) -> int:
    """
    Negamax với cắt tỉa alpha-beta.
    - board: trạng thái bàn cờ.
    - depth: độ sâu còn lại.
    - alpha, beta: tham số cắt nhánh.
    - color: = +1 nếu đang tính cho White (maximize), = -1 nếu cho Black.
    Trả về giá trị kleur * evaluation (đã chứa color).
    """
    # Nếu dừng điều kiện cơ bản
    if depth == 0 or board.is_game_over():
        return color * evaluate_board(board)

    max_eval = -INFINITY
    for move in board.legal_moves:
        board.push(move)
        # Chú ý hoán đổi vai trò alpha, beta và color
        val = -negamax_ab(board, depth - 1, -beta, -alpha, -color)
        board.pop()

        if val > max_eval:
            max_eval = val
        if max_eval > alpha:
            alpha = max_eval
        # Nếu alpha ≥ beta, cắt nhánh
        if alpha >= beta:
            break

    return max_eval

def get_best_move(board: chess.Board, depth: int = 3) -> chess.Move:
    """
    Wrapper trả về best move theo Negamax alpha-beta.
    - Nếu tới lượt White, color = +1; nếu đến lượt Black, color = -1.
    """
    best_move = None
    alpha = -INFINITY
    beta = INFINITY
    color = 1 if board.turn == chess.WHITE else -1
    best_value = -INFINITY

    for move in board.legal_moves:
        board.push(move)
        # Kết quả đánh giá = -negamax ở depth-1, hoán đổi alpha, beta, color
        value = -negamax_ab(board, depth - 1, -beta, -alpha, -color)
        board.pop()

        if value > best_value:
            best_value = value
            best_move = move
        if best_value > alpha:
            alpha = best_value
        # Cắt nhánh tạm nếu muốn (không bắt buộc ở node gốc)
        # if alpha >= beta: break

    return best_move
