# engines/helpers.py

import chess
import numpy as np

# === 1. Giá trị vật chất (material value) của từng loại quân ===
PIECE_VALUES = {
    chess.PAWN: 100,  # dùng đơn vị centipawn (100 = 1 pawn)
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0  # Giá trị King có thể đặt 20000, nhưng trong đánh giá chủ yếu để tránh mất King
}

# === 2. Piece‑Square Tables (PST) cho WHITE; BLACK sẽ lấy giá trị "đảo ngược" ===
# (Giá trị tham khảo, bạn có thể tinh chỉnh cho phù hợp)
PAWN_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

KNIGHT_PST = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

BISHOP_PST = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

ROOK_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0]
]

QUEEN_PST = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]
]

KING_MID_PST = [
    [20, 30, 10, 0, 0, 10, 30, 20],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30]
]

KING_END_PST = [
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50]
]


# === 3. Hàm lấy giá bàng PST cho một quân ở một ô cụ thể ===
def get_pst_value(piece: chess.Piece, square: int) -> int:
    """
    piece: đối tượng chess.Piece (chứa piece_type, color)
    square: int 0..63 (a8=0 ... h1=63)
    Trả về điểm PST (positive = tốt cho White, negative = tốt cho Black).
    """
    piece_type = piece.piece_type
    color = piece.color  # True = WHITE, False = BLACK

    # Tính file, rank (0-based) theo python-chess: rank 0 = a1, rank 7 = a8
    rank = chess.square_rank(square)  # 0..7, 0 là h1, 7 là h8
    file = chess.square_file(square)  # 0..7, 0 là a, 7 là h

    # Đổi ý rank/file cho dễ mapping vào bảng (bảng định nghĩa từ White perspective,
    # hàng đầu tiên index=0 ứng với rank 8; index=7 ứng với rank 1).
    # Tức white: idx_rank = 7 - rank, black: idx_rank = rank
    if piece_type == chess.PAWN:
        table = PAWN_PST
    elif piece_type == chess.KNIGHT:
        table = KNIGHT_PST
    elif piece_type == chess.BISHOP:
        table = BISHOP_PST
    elif piece_type == chess.ROOK:
        table = ROOK_PST
    elif piece_type == chess.QUEEN:
        table = QUEEN_PST
    elif piece_type == chess.KING:
        # Có thể chuyển giữa giai đoạn giữa ván (midgame) và endgame nếu muốn.
        # Ở đây tạm dùng luôn PST giai đoạn giữa ván:
        table = KING_MID_PST
    else:
        return 0

    if color == chess.WHITE:
        idx_rank = 7 - rank
        idx_file = file
        return table[idx_rank][idx_file]
    else:
        # Nếu là Black, “lật ngược” bảng: dòng 0 (rank8) cho White → thành dòng 7 (rank1) cho Black
        idx_rank = rank
        idx_file = file
        # Nên trả giá trị *(-1) để áp dụng theo White perspective.
        # Ví dụ nếu White Pawn đứng ở e4 được +20, thì Black Pawn ở e5 (mirror) = -20.
        return -table[idx_rank][idx_file]


# === 4. Hàm đánh giá tổng hợp: material + PST ===
def evaluate_board(board: chess.Board) -> int:
    """
    Trả về giá trị đánh giá của board từ White perspective (đơn vị centipawn).
    Giá trị dương → ưu thế White, âm → ưu thế Black.
    """
    if board.is_checkmate():
        # Nếu checkmate, chắc chắn bên đang bị chiếu tướng / chuẩn bị chịu thua
        # Thường gán một giá cực lớn (so với depth) để ai không lựa chọn hướng dẫn đến checkmate.
        return -999_999 if board.turn == chess.WHITE else 999_999
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0  # hoà

    value = 0
    # 1) Điểm vật chất
    for piece_type in PIECE_VALUES:
        value += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        value -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]

    # 2) Điểm vị trí (PST) cho từng quân
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value += get_pst_value(piece, square)

    return value

def board_to_tensor(board: chess.Board):
    """
    Chuyển chess.Board sang tensor shape (12,8,8), dtype float32.
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        planes[idx][7 - rank][file] = 1.0
    return planes