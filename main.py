import sys
import pygame
import chess
from game.game import Game
from gui.board_renderer import BoardRenderer
from gui.gui_utils import pixel_to_square, square_to_pixel
from engines import greedy, minimax, negamax, mcts, mcts_supervised
from game.constants import WIDTH, HEIGHT, SQUARE_SIZE, WHITE as COLOR_LIGHT, BLACK as COLOR_DARK

# Thiết lập thông số cho menu
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 10
FONT_SIZE = 24

# Danh sách tên các engine
ENGINE_NAMES = [
    "Human",
    "Greedy",
    "Minimax",
    "Negamax",
    "MCTS",
    "MCTS_supervised"
]

# Tạo hàm ánh xạ lựa chọn sang hàm tương ứng
def get_engine_function(choice, color):
    if choice == "Human":
        return None
    elif choice == "Greedy":
        return lambda board: greedy.greedy_move(board)
    elif choice == "Minimax":
        return lambda board: minimax.get_best_move(board, depth=3)
    elif choice == "Negamax":
        return lambda board: negamax.get_best_move(board, depth=3)
    elif choice == "MCTS":
        return lambda board: mcts.run_mcts(board, time_limit=1.0)
    elif choice == "MCTS_supervised":
            return lambda board: mcts_supervised.run_mcts_supervised(board, n_simulations=10, c_puct=2.0)[0]
    else:
        return None

# Lớp Button đơn giản để hiển thị và kiểm tra click
class Button:
    def __init__(self, rect, text, font, callback=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback
        self.selected = False

    def draw(self, surface):
        # Màu nền nếu được chọn hay không
        bg_color = (100, 200, 100) if self.selected else (200, 200, 200)
        pygame.draw.rect(surface, bg_color, self.rect)
        # Vẽ viền
        pygame.draw.rect(surface, (50, 50, 50), self.rect, 2)
        # Render text
        text_surf = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback(self.text)

# Hàm hiển thị menu và trả về lựa chọn của người dùng
def show_menu(screen):
    pygame.display.set_caption("Chess AI - Chọn Engine")
    font = pygame.font.SysFont(None, FONT_SIZE)

    # Thiết lập vị trí các button cho White và Black
    buttons_white = []
    buttons_black = []
    start_button = None

    # Tạo button cho engine của White
    for i, name in enumerate(ENGINE_NAMES):
        x = (WIDTH // 4) - (BUTTON_WIDTH // 2)
        y = 100 + i * (BUTTON_HEIGHT + BUTTON_MARGIN)
        btn = Button((x, y, BUTTON_WIDTH, BUTTON_HEIGHT), name, font)
        buttons_white.append(btn)

    # Tạo button cho engine của Black
    for i, name in enumerate(ENGINE_NAMES):
        x = (3 * WIDTH // 4) - (BUTTON_WIDTH // 2)
        y = 100 + i * (BUTTON_HEIGHT + BUTTON_MARGIN)
        btn = Button((x, y, BUTTON_WIDTH, BUTTON_HEIGHT), name, font)
        buttons_black.append(btn)

    # Tạo button Start
    start_rect = ((WIDTH // 2) - (BUTTON_WIDTH // 2), HEIGHT - 100, BUTTON_WIDTH, BUTTON_HEIGHT)
    start_button = Button(start_rect, "Start Game", font)

    # Biến lưu lựa chọn
    choice_white = "Human"
    choice_black = "Human"

    # Callback khi click button engine
    def select_white(name):
        nonlocal choice_white
        choice_white = name
        for btn in buttons_white:
            btn.selected = (btn.text == name)

    def select_black(name):
        nonlocal choice_black
        choice_black = name
        for btn in buttons_black:
            btn.selected = (btn.text == name)

    # Gán callback cho từng button white/black
    for btn in buttons_white:
        btn.callback = select_white
    for btn in buttons_black:
        btn.callback = select_black
    start_button.callback = lambda x: None  # xử lý riêng

    # Khởi đặt ban đầu: chọn ô Human
    buttons_white[0].selected = True
    buttons_black[0].selected = True

    clock = pygame.time.Clock()
    running = True
    start_clicked = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Xử lý click cho từng button
            for btn in buttons_white + buttons_black + [start_button]:
                btn.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.rect.collidepoint(event.pos):
                    start_clicked = True
                    running = False

        # Vẽ background
        screen.fill((180, 180, 180))

        # Vẽ tiêu đề
        title_font = pygame.font.SysFont(None, 36)
        text_w = title_font.render("Chọn Engine cho White", True, (0, 0, 0))
        screen.blit(text_w, (WIDTH // 4 - text_w.get_width() // 2, 50))
        text_b = title_font.render("Chọn Engine cho Black", True, (0, 0, 0))
        screen.blit(text_b, (3 * WIDTH // 4 - text_b.get_width() // 2, 50))

        # Vẽ các button
        for btn in buttons_white + buttons_black + [start_button]:
            btn.draw(screen)

        pygame.display.flip()
        clock.tick(30)

    # Sau khi nhấn Start: trả về 2 lựa chọn
    return choice_white, choice_black


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Hiển thị menu chọn engine
    choice_w, choice_b = show_menu(screen)
    engine_w = get_engine_function(choice_w, chess.WHITE)
    engine_b = get_engine_function(choice_b, chess.BLACK)

    # Khởi tạo game
    game = Game(choose_engine_white=engine_w, choose_engine_black=engine_b)
    renderer = BoardRenderer(screen)
    selected_square = None
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                sq = pixel_to_square(pos[0], pos[1])
                if sq is not None:
                    # Nếu người chơi nhấn khi đến lượt của chính họ
                    if ((game.board.turn == chess.WHITE and engine_w is None) or
                        (game.board.turn == chess.BLACK and engine_b is None)):
                        if selected_square is None:
                            if game.board.piece_at(sq) and \
                               ((game.board.turn == chess.WHITE and game.board.piece_at(sq).color == chess.WHITE) or \
                                (game.board.turn == chess.BLACK and game.board.piece_at(sq).color == chess.BLACK)):
                                selected_square = sq
                        else:
                            if game.human_move(selected_square, sq):
                                selected_square = None
                            else:
                                selected_square = None

        # Lượt AI thực hiện nếu có
        if not game.is_game_over():
            if (game.board.turn == chess.WHITE and engine_w) or (game.board.turn == chess.BLACK and engine_b):
                ai_move = game.engine_move()
                if ai_move:
                    game.apply_move(ai_move)

        # Vẽ bàn cờ và quân
        renderer.draw_board()
        renderer.draw_pieces(game.board)

        if selected_square is not None:
            x, y = square_to_pixel(selected_square)
            pygame.draw.rect(screen, (0, 255, 0), (x, y, SQUARE_SIZE, SQUARE_SIZE), 3)

        pygame.display.flip()
        clock.tick(30)

        if game.is_game_over():
            result = game.get_result()
            print("Kết thúc ván cờ. Kết quả:", result)
            pygame.time.wait(3000)
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
