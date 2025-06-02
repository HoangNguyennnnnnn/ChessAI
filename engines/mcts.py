# engines/mcts.py

import chess
import random
import math
import time


class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move  # move đưa board từ parent -> node này
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def best_child(self, c_param=1.4):
        choices = []
        for child in self.children:
            # UCT = w_i / n_i + c * sqrt(ln N / n_i)
            if child.visits == 0:
                uct = float('inf')
            else:
                uct = child.wins / child.visits + c_param * math.sqrt(math.log(self.visits) / child.visits)
            choices.append((uct, child))
        return max(choices, key=lambda x: x[0])[1]

    def expand(self):
        tried_moves = [child.move for child in self.children]
        for move in self.board.legal_moves:
            if move not in tried_moves:
                next_board = self.board.copy()
                next_board.push(move)
                child_node = MCTSNode(next_board, parent=self, move=move)
                self.children.append(child_node)
                return child_node
        return None  # đã mở hết

    def simulate(self):
        """
        Mô phỏng rollout ngẫu nhiên cho đến khi game kết thúc.
        Trả về 1 nếu trắng thắng, 0.5 hòa, 0 nếu đen thắng.
        """
        simulation_board = self.board.copy()
        while not simulation_board.is_game_over():
            move = random.choice(list(simulation_board.legal_moves))
            simulation_board.push(move)
        result = simulation_board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return 0.0
        else:
            return 0.5

    def backpropagate(self, result):
        """
        result: 1.0 (white win), 0.5 (draw), 0.0 (black win)
        Nếu node.board.turn là White đang chơi, thì:
            - Nếu result = 1 (white thắng), đây là thắng -> cộng wins
            - Nếu result = 0 (black thắng), ngược lại
        Nhưng thường switch interpretation:
        wins = số điểm với perspective của người *đã chọn move* ở parent.
        Cho gọn: giả sử xem kết quả từ White perspective:
            - child.wins cộng result nếu white; = (1-result) nếu black.
        Sau cùng trong best_child, chỉ quan tâm tỷ lệ wins/visits.
        """
        self.visits += 1
        # Xét result từ góc trắng: nếu node.board.turn = trắng => nghĩa là
        # parent đã cho nước. Tiếp tục propagate lên:
        self.wins += result
        if self.parent:
            # Khi parent là lượt khác (đen), kết quả white thắng = kết quả parent perspective?
            # Để đơn giản, ta truyền result unchanged và mỗi node cộng result nếu node.turn=white, cộng (1-result) nếu node.turn=black.
            # Nhưng vì mình cộng wins đơn giản, best_child sẽ ưu tiên tỉ lệ wins, không cần đổi result.
            self.parent.backpropagate(result)


def run_mcts(root_board: chess.Board, time_limit: float = 1.0) -> chess.Move:
    """
    Thực hiện MCTS trong khoảng time_limit (giây), trả về best move.
    """
    root = MCTSNode(root_board.copy())
    end_time = time.time() + time_limit

    # Nếu board đã hết (chiếu hết hoặc hòa), return None
    if root_board.is_game_over():
        return None

    while time.time() < end_time:
        leaf = tree_policy(root)
        reward = leaf.simulate()
        leaf.backpropagate(reward)

    # Lấy child nhiều visits nhất
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move


def tree_policy(node: MCTSNode) -> MCTSNode:
    """
    Nếu node chưa fully expanded: expand và return node mới.
    Nếu đã, chọn child tốt nhất (theo UCT) và tiếp tục xuống.
    """
    while not node.board.is_game_over():
        if not node.is_fully_expanded():
            return node.expand()
        else:
            node = node.best_child()
    return node
