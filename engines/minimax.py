# engines/minimax.py

import chess
from engines.helpers import evaluate_board

INFINITY = 10_000

def minimax_ab(board: chess.Board, depth: int, alpha: int, beta: int, maximizing_player: bool) -> int:
    """
    Minimax với cắt tỉa alpha-beta.
    - board: trạng thái bàn cờ hiện tại.
    - depth: độ sâu còn lại.
    - alpha: best giá trị trắng (maximizer) đã tìm được ở các node anh em bên trái.
    - beta: best giá trị đen (minimizer) đã tìm được ở các node anh em bên trái.
    - maximizing_player: True nếu đang xét node maximize (trắng), False nếu node minimize (đen).
    Trả về giá trị đánh giá của node đó.
    """
    # Nếu dừng điều kiện cơ bản (độ sâu = 0 hoặc game over), trả về giá trị đánh giá
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = -INFINITY
        for move in board.legal_moves:
            board.push(move)
            eval = minimax_ab(board, depth - 1, alpha, beta, False)
            board.pop()

            if eval > max_eval:
                max_eval = eval
            if max_eval > alpha:
                alpha = max_eval
            # Cắt nhánh: nếu alpha ≥ beta thì không cần xét tiếp
            if alpha >= beta:
                break
        return max_eval
    else:
        min_eval = INFINITY
        for move in board.legal_moves:
            board.push(move)
            eval = minimax_ab(board, depth - 1, alpha, beta, True)
            board.pop()

            if eval < min_eval:
                min_eval = eval
            if min_eval < beta:
                beta = min_eval
            # Cắt nhánh: nếu alpha ≥ beta thì không cần xét tiếp
            if alpha >= beta:
                break
        return min_eval

def get_best_move(board: chess.Board, depth: int = 3) -> chess.Move:
    """
    Wrapper để trả về nước đi tốt nhất theo Minimax có cắt tỉa alpha-beta.
    Mặc định depth = 3 (bạn có thể điều chỉnh tuỳ máy).
    """
    best_move = None
    alpha = -INFINITY
    beta = INFINITY

    if board.turn == chess.WHITE:
        max_eval = -INFINITY
        for move in board.legal_moves:
            board.push(move)
            eval = minimax_ab(board, depth - 1, alpha, beta, False)
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move
            if max_eval > alpha:
                alpha = max_eval
            # Bạn có thể cắt tiếp (không cần) ở đây, nhưng để wrapper đúng logic
            # thì chỉ cắt ở node gốc nếu chắc chắn:
            # if alpha >= beta: break
        return best_move

    else:  # đến lượt Đen (minimizer)
        min_eval = INFINITY
        for move in board.legal_moves:
            board.push(move)
            eval = minimax_ab(board, depth - 1, alpha, beta, True)
            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move = move
            if min_eval < beta:
                beta = min_eval
            # Tương tự, có thể cắt nhánh tại node gốc:
            if alpha >= beta:
                break
        return best_move
