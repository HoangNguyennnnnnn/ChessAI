# gui/gui_utils.py

from game.constants import SQUARE_SIZE
import chess

def pixel_to_square(x, y):
    """
    Nhận tọa độ pixel từ sự kiện chuột, trả về ô chess.SQUARE (int 0-63).
    Giả sử window size = 8 * SQUARE_SIZE và (0,0) là góc trên bên trái (a8).
    """
    file_idx = x // SQUARE_SIZE
    rank_idx = 7 - (y // SQUARE_SIZE)
    if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
        return chess.square(file_idx, rank_idx)
    return None

def square_to_pixel(square):
    """
    Ngược lại: từ ô (0-63) sang tọa độ pixel (x, y) để vẽ highlight hay tương tác.
    """
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    x = file_idx * SQUARE_SIZE
    y = (7 - rank_idx) * SQUARE_SIZE
    return x, y
