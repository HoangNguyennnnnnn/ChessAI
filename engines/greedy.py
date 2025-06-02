# engines/greedy.py

import chess

# giá trị tương đối cho từng piece type
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # giá trị vô hạn nhưng trong Greedy ta không đổi nước vua
}


def get_material_value(board: chess.Board) -> int:
    """
    Tính tổng giá trị vật chất (material) hiện tại của board: white - black
    """
    value = 0
    for piece_type in PIECE_VALUES:
        value += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        value -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    return value


def greedy_move(board: chess.Board) -> chess.Move:
    """
    Trả về nước đi mà greedy cho là tốt nhất:
    - Duyệt qua toàn bộ nước đi hợp lệ, giả sử push tạm lên board
    - Tính toán (hiện tại) giá trị vật chất sau khi di chuyển
    - Chọn nước đưa giá trị tốt nhất (board.turn = True => maximize, False => minimize)
    """
    best_move = None
    best_value = -9999 if board.turn == chess.WHITE else 9999

    for move in board.legal_moves:
        board.push(move)
        value = get_material_value(board)
        board.pop()

        # Nếu đến lượt white (maximize), chọn value lớn nhất
        if board.turn == chess.WHITE:
            if value > best_value:
                best_value = value
                best_move = move
        else:
            # đến lượt black (minimize vật chất => value càng nhỏ càng tốt)
            if value < best_value:
                best_value = value
                best_move = move

    return best_move
