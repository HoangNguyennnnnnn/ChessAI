# engines/mcts.py

import chess
import math
from engines.greedy import greedy_move  # import hàm greedy_move


class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board              # Trạng thái cờ tại node này
        self.parent = parent            # Node cha
        self.move = move                # Nước đi đưa từ parent → node này
        self.children = []              # Danh sách các node con
        self.visits = 0                 # Số lần node này được thăm
        self.value_sum = 0.0            # Tổng reward tích lũy (từ góc White perspective)

    def is_fully_expanded(self) -> bool:
        """
        Trả về True nếu node đã mở (expand) hết mọi nước đi hợp lệ.
        """
        return len(self.children) == len(list(self.board.legal_moves))

    def expand(self):
        """
        PHASE 2 – EXPANSION:
        - Chọn một nước move hợp lệ mà node hiện tại chưa có con tương ứng.
        - Tạo node con mới, đẩy move đó lên board, thêm vào children và trả về node con.
        - Nếu đã thử hết mọi nước hợp lệ thì trả về None.
        """
        tried_moves = {child.move for child in self.children}
        moves = list(self.board.legal_moves)

        for move in moves:
            if move not in tried_moves:
                next_board = self.board.copy()
                next_board.push(move)
                child_node = MCTSNode(next_board, parent=self, move=move)
                self.children.append(child_node)
                return child_node

        return None  # Đã expand hết mọi nước hợp lệ

    def best_child(self, c_param: float = 1.4):
        """
        PHASE 1 (trong vòng lặp Selection) – Compute UCT và chọn child có UCT cao nhất.
        UCT(child) = (Q_child / N_child) + c_param * sqrt( ln(N_parent) / N_child )
          - Q_child / N_child  là exploitation (trung bình reward của child)
          - c_param * sqrt( ln(N_parent) / N_child )  là exploration
        Nếu child.visits == 0 → cho UCT = +inf để ưu tiên mở rộng trước (exploration).
        """
        best_score = -float('inf')
        best_node = None

        for child in self.children:
            if child.visits == 0:
                # Nếu chưa thăm, luôn xếp ưu tiên cao nhất
                score = float('inf')
            else:
                avg_value = child.value_sum / child.visits
                score = avg_value + c_param * math.sqrt(
                    math.log(self.visits) / child.visits
                )

            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def backpropagate(self, reward: float):
        """
        PHASE 4 – BACKPROPAGATION:
        - Cập nhật visits và value_sum tại node này.
        - Đổi dấu reward khi truyền lên node cha (vì góc nhìn luân phiên White ↔ Black).
        reward ∈ {+1, 0, -1} luôn tính từ White perspective:
          +1 → White thắng, -1 → Black thắng, 0 → Hòa.
        """
        self.visits += 1
        self.value_sum += reward

        if self.parent is not None:
            # Đổi dấu reward khi lan ngược lên node cha
            self.parent.backpropagate(-reward)


def select_node(root: MCTSNode, c_param: float) -> MCTSNode:
    """
    PHASE 1 – SELECTION:
    - Bắt đầu từ root, lặp chọn child có UCT cao nhất (best_child) cho đến khi:
      • Node hiện tại chưa fully expanded ➔ dừng lại trả về node đó (để Expansion).
      • Hoặc Node đã terminal (board.is_game_over()) ➔ trả về node đó (để Simulation).
    """
    node = root
    while not node.board.is_game_over():
        if not node.is_fully_expanded():
            # Node còn nước chưa thử → dừng selection, sẽ expand từ đây
            return node
        else:
            # Node đã fully expanded → tiếp tục descent xuống best_child
            node = node.best_child(c_param)
    return node  # Nếu là node terminal thì trả về luôn


def simulate_to_end(board: chess.Board) -> int:
    """
    PHASE 3 – SIMULATION (Rollout greedy cho đến khi game kết thúc):
    - Sao chép board, liên tục gọi greedy_move để chọn nước “tốt nhất” tại mỗi trạng thái
    - Kết quả trả về ∈ {+1, 0, -1} theo White perspective:
      +1  → White thắng (“1-0”)
      -1  → Black thắng (“0-1”)
       0  → Hoà (“1/2-1/2”)
    """
    sim_board = board.copy()

    while not sim_board.is_game_over():
        move = greedy_move(sim_board)
        # Nếu greedy_move trả về None (ví dụ không có nước đi), break để tránh loop vô hạn
        if move is None:
            break
        sim_board.push(move)

    result = sim_board.result()
    if result == "1-0":
        return +1
    elif result == "0-1":
        return -1
    else:
        return 0


def run_mcts(root_board: chess.Board,
             n_simulations: int = 100,
             c_param: float = 1.4) -> chess.Move:
    """
    Hàm chính gọi MCTS cho cờ vua:
    1) Khởi tạo root node từ trạng thái root_board.
    2) Chạy n_simulations vòng lặp MCTS:
       a) SELECTION:    node = select_node(root, c_param)
       b) EXPANSION:    nếu node chưa game_over, thì child = node.expand(),
                        nếu node terminal thì child = node
       c) SIMULATION:   reward = simulate_to_end(child.board)
       d) BACKPROPAGATION: child.backpropagate(reward)
    3) Cuối cùng, từ root.children chọn node con có visits lớn nhất → trả về nước move đó.
    """
    root = MCTSNode(root_board.copy())

    for _ in range(n_simulations):
        # ----- 1) SELECTION -----
        node_to_expand = select_node(root, c_param)

        # ----- 2) EXPANSION -----
        if not node_to_expand.board.is_game_over():
            child = node_to_expand.expand()
        else:
            # Nếu node selection đã là terminal, không expand, simulation trực tiếp
            child = node_to_expand

        # ----- 3) SIMULATION -----
        reward = simulate_to_end(child.board)

        # ----- 4) BACKPROPAGATION -----
        child.backpropagate(reward)

    # Nếu root không có con (ví dụ không có nước đi hợp lệ), trả None
    if not root.children:
        return None

    # Chọn child có visits nhiều nhất để đưa ra move cuối cùng
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move
