# engines/mcts.py

import chess
import math
from engines.greedy import greedy_move  # hàm greedy_move
from engines.helpers import evaluate_board  # heuristic evaluation


class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(list(self.board.legal_moves))

    def expand(self):
        # Mở rộng một nước chưa thử
        tried = {c.move for c in self.children}
        for mv in self.board.legal_moves:
            if mv not in tried:
                nb = self.board.copy()
                nb.push(mv)
                child = MCTSNode(nb, parent=self, move=mv)
                self.children.append(child)
                return child
        return None

    def best_child(self, c_param: float = 1.4) -> 'MCTSNode':
        best, best_score = None, -float('inf')
        for c in self.children:
            if c.visits == 0:
                score = float('inf')
            else:
                exploitation = c.value_sum / c.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / c.visits)
                score = exploitation + exploration
            if score > best_score:
                best_score, best = score, c
        return best

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value_sum += reward
        if self.parent:
            self.parent.backpropagate(-reward)


def select_node(root: MCTSNode, c_param: float) -> MCTSNode:
    node = root
    # selection until leaf
    while not node.board.is_game_over():
        if not node.is_fully_expanded():
            return node
        node = node.best_child(c_param)
    return node


def simulate(node_board: chess.Board, max_depth: int = 20) -> float:
    # Kết hợp greedy rollout và heuristic khi đạt max_depth
    b = node_board.copy()
    depth = 0
    while not b.is_game_over() and depth < max_depth:
        mv = greedy_move(b)
        if mv is None:
            break
        b.push(mv)
        depth += 1
    # Nếu game_over thì kết quả +-1 hoặc 0, ngược lại dùng đánh giá nhanh
    if b.is_game_over():
        res = b.result()
        if res == '1-0':
            return +1
        if res == '0-1':
            return -1
        return 0
    # heuristic centipawn → scale về [-1,1]
    score_cp = evaluate_board(b)
    return max(-1.0, min(1.0, score_cp / 1000.0))


def run_mcts(root_board: chess.Board,
             n_simulations: int = 100,
             c_param: float = 1.4) -> chess.Move:
    root = MCTSNode(root_board.copy())
    for _ in range(n_simulations):
        leaf = select_node(root, c_param)
        child = leaf.expand() if not leaf.board.is_game_over() else leaf
        reward = simulate(child.board)
        child.backpropagate(reward)
    if not root.children:
        return None
    # chọn con có visits cao nhất
    best = max(root.children, key=lambda c: c.visits)
    return best.move
