# gui/board_renderer.py

import pygame
import chess
from game.constants import WHITE, BLACK, BOARD_SIZE, SQUARE_SIZE


# Pre-load ảnh quân cờ
def load_piece_images():
    pieces = {}
    names = ['p', 'r', 'n', 'b', 'q', 'k']  # pawn, rook, knight, bishop, queen, king
    colors = ['w', 'b']
    for color in colors:
        for name in names:
            filename = f"assets/pieces/{color}{name}.png"
            image = pygame.image.load(filename)
            image = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
            pieces[f"{color}{name}"] = image
    return pieces


class BoardRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.piece_images = load_piece_images()

    def draw_board(self):
        for rank in range(BOARD_SIZE):
            for file in range(BOARD_SIZE):
                color = WHITE if (rank + file) % 2 == 0 else BLACK
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )

    def draw_pieces(self, board: chess.Board):
        """
        board.piece_at(square) trả về Piece hoặc None.
        square: số nguyên 0-63, tuần tự từ a8 (0) -> h1 (63)
        Vị trí pixel: file * SQUARE_SIZE, rank * SQUARE_SIZE
        Lưu ý: chess module mặc định rank 0 là a8 (trên cùng), nên ta cần dịch tọa độ.
        """
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Ví dụ piece.symbol() trả về 'P','n','k',... in uppercase là white, lowercase là black
                symbol = piece.symbol()
                color = 'w' if symbol.isupper() else 'b'
                name = symbol.lower()
                image = self.piece_images[f"{color}{name}"]

                # Tính file, rank: file = file index (0-7), rank = 7 - rank_index để GUI hiển thị rank 8 ở trên
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                x = file_idx * SQUARE_SIZE
                y = (7 - rank_idx) * SQUARE_SIZE
                self.screen.blit(image, (x, y))
