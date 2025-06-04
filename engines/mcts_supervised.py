import chess
import torch
import math, numpy as np
import random
from engines.helpers import board_to_tensor,evaluate_board
from engines.network_model import PolicyNet   # cần chứa đúng định nghĩa của PolicyNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load đúng cấu hình đã train:
model = PolicyNet(num_res_blocks=20, num_channels=256, action_size=4096).to(DEVICE)
model.load_state_dict(torch.load("models/policy_supervised_final.pt", map_location=DEVICE))
model.eval()

def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move_from_parent=None):
        self.board = board
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.P = {}
        self.N = {}
        self.W = {}
        self.Q = {}
        self.children = {}
        self._is_expanded = False

    def expand(self):
        board_np = board_to_tensor(self.board)  # numpy.ndarray shape (12,8,8)
        board_tensor = torch.from_numpy(board_np).unsqueeze(0).to(DEVICE)  # (1,12,8,8) trên device
        with torch.no_grad():
            logits = model(board_tensor)    # (1,4096)
        logits = logits.cpu().numpy().reshape(-1)
        legal_moves = list(self.board.legal_moves)
        random.shuffle(legal_moves)
        indices = [move_to_index(m) for m in legal_moves]
        logits_legal = np.array([logits[i] for i in indices], dtype=np.float32)
        exp = np.exp(logits_legal - np.max(logits_legal))
        probs = exp / np.sum(exp)
        for j, move in enumerate(legal_moves):
            self.P[move] = probs[j]
            self.N[move] = 0
            self.W[move] = 0.0
            self.Q[move] = 0.0
        self._is_expanded = True

def select_child(node: MCTSNode, c_puct=1.0):
    total_N = sum(node.N[a] for a in node.N)
    best_score, best_move = -float('inf'), None
    for a in node.N:
        Q = node.Q[a]
        P = node.P[a]
        N = node.N[a]
        U = Q + c_puct * P * math.sqrt(total_N) / (1 + N)
        if U > best_score:
            best_score, best_move = U, a
    return best_move

def backpropagate(node: MCTSNode, reward: float):
    current, v = node, reward
    while current.parent is not None:
        parent = current.parent
        move = current.move_from_parent
        parent.N[move] += 1
        parent.W[move] += v
        parent.Q[move] = parent.W[move] / parent.N[move]
        current = parent
        v = -v  # Đảo reward cho lượt khác

def run_mcts_supervised(root_board: chess.Board, n_simulations=400, c_puct=1.4):
    root = MCTSNode(root_board.copy())
    root.expand()
    for _ in range(n_simulations):
        node = root
        # 1. Selection
        while node._is_expanded and not node.board.is_game_over():
            a = select_child(node, c_puct)
            next_board = node.board.copy()
            next_board.push(a)
            if a in node.children:
                node = node.children[a]
            else:
                child = MCTSNode(next_board, parent=node, move_from_parent=a)
                node.children[a] = child
                node = child

        # 2. Expansion (nếu chưa game over)
        if not node.board.is_game_over():
            node.expand()
            # Dùng heuristic nhỏ cho reward giai đoạn leaf: 0 (hòa)
            # Tính giá trị centipawn rồi chuẩn hóa thành reward ∈ [−1, 1]
            score = evaluate_board(node.board)  # ví dụ ± vài trăm
            # Chọn K ≈ 1000 (có thể điều chỉnh tùy tập dữ liệu centipawn thực tế)
            reward = math.tanh(score / 1000.0)
        else:
            res = node.board.result()
            if res == "1-0": reward = 1.0
            elif res == "0-1": reward = -1.0
            else: reward = 0.0

        # 3. Backpropagation
        backpropagate(node, reward)

    # Lấy best move = argmax visits
    pi_vec = {move: root.N[move] for move in root.N}
    best_move = max(pi_vec, key=lambda m: pi_vec[m])
    return best_move,pi_vec

# Khi tích hợp vào main.py, bạn có thể gọi:
# run_mcts_supervised(board, n_simulations=200, c_puct=1.0)[0] để lấy move.
