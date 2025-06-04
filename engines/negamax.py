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
    legal_moves.sort(key=lambda mv: board.is_capture(mv), reverse=True)

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
    """
    Wrapper để lấy nước đi tốt nhất với Negamax + Alpha-Beta + Move Ordering.

    - Nếu đến lượt White: color = +1
    - Nếu đến lượt Black: color = -1

    Tham số:
    - board: python-chess.Board đang ở vị thế cần chọn nước.
    - depth: số plies (cộng độ sâu) để search.

    Trả về: đối tượng chess.Move hoặc None nếu không có nước.
    """
    best_move = None
    alpha = -INFINITY
    beta = INFINITY
    color = 1 if board.turn == chess.WHITE else -1
    best_value = -INFINITY

    # Sắp xếp trước các nước bắt quân để cắt nhánh sớm hơn
    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda mv: board.is_capture(mv), reverse=True)

    for move in legal_moves:
        board.push(move)
        value = -negamax_ab(board, depth - 1, -beta, -alpha, -color)
        board.pop()

        if value > best_value:
            best_value = value
            best_move = move
        if best_value > alpha:
            alpha = best_value
        # Có thể mở cắt nhánh tại root nếu muốn:
        # if alpha >= beta:
        #     break

    return best_move
