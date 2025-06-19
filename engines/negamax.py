# engines/negamax.py

import chess
from engines.helpers import evaluate_board

INFINITY = 1_000_000


def negamax_ab(board: chess.Board, depth: int, alpha: int, beta: int, color: int) -> int:
    """
    Negamax với cắt tỉa alpha-beta, sử dụng evaluate_board từ engines/helpers.py.

    - board: trạng thái bàn cờ (python-chess).
    - depth: số bước (plies) còn lại.
    - alpha, beta: giới hạn cắt tỉa (int, centipawn).
    - color: +1 nếu tính cho White, -1 nếu tính cho Black.

    Trả về: color * eval_centipawn (int).
    """
    # Điều kiện dừng: đã tới độ sâu 0 hoặc game kết thúc
    if depth == 0 or board.is_game_over():
        return color * evaluate_board(board)

    max_eval = -INFINITY

    # === Move Ordering: xếp các nước bắt quân lên trước ===
    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda mv: (board.is_capture(mv), board.gives_check(mv)), reverse=True)

    for move in legal_moves:
        board.push(move)
        # Hoán đổi vai trò alpha, beta và color
        score = -negamax_ab(board, depth - 1, -beta, -alpha, -color)
        board.pop()

        if score > max_eval:
            max_eval = score
        if max_eval > alpha:
            alpha = max_eval
        # Cắt nhánh nếu alpha >= beta
        if alpha >= beta:
            break

    return max_eval


def get_best_move(board: chess.Board, depth: int = 3) -> chess.Move:
    best_move = None
    alpha = -INFINITY
    beta = INFINITY
    color = 1 if board.turn == chess.WHITE else -1
    best_value = -INFINITY

    # Sắp xếp trước các nước bắt quân để cắt nhánh sớm hơn
    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda mv: (board.is_capture(mv), board.gives_check(mv)), reverse=True)

    for move in legal_moves:
        board.push(move)
        value = -negamax_ab(board, depth - 1, -beta, -alpha, -color)
        board.pop()

        if value > best_value:
            best_value = value
            best_move = move
        if best_value > alpha:
            alpha = best_value

        if alpha >= beta:
            break

    return best_move
