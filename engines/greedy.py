# engines/greedy.py

import chess
from engines.helpers import evaluate_board  # import hàm đánh giá tổng hợp

def greedy_move(board: chess.Board) -> chess.Move:
    """
    Trả về nước đi greedy dựa trên hàm đánh giá evaluate_board:
    - Duyệt qua tất cả nước đi hợp lệ, thử đẩy lên board tạm
    - Tính giá trị đánh giá của board mới (từ White perspective)
    - Chọn nước đi tốt nhất: nếu đến lượt White thì maximize, nếu đến lượt Black thì minimize
    """
    best_move = None

    # Giá trị khởi tạo tùy vào bên đang đi (White maximize, Black minimize)
    if board.turn == chess.WHITE:
        best_value = -float('inf')
    else:
        best_value = float('inf')

    for move in board.legal_moves:
        board.push(move)
        value = evaluate_board(board)
        board.pop()

        # Nếu đến lượt White (maximize)
        if board.turn == chess.WHITE:
            if value > best_value:
                best_value = value
                best_move = move
        else:
            # Nếu đến lượt Black (minimize)
            if value < best_value:
                best_value = value
                best_move = move

    return best_move
