import numpy as np
from enum import IntEnum

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Game2048Env:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((4,4), dtype=int)
        self.steps = 0
        self.total_score = 0
        self.reset()

    # --- Public API ---
    # Reset the environment to start a new game
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board[:] = 0
        self.total_score = 0
        self.steps = 0
        self._add_tile()
        self._add_tile()
        return self.board.copy()

    def get_state(self):
        return self.board.copy()

    # Update the environment with the given action
    def step(self, action):
        action = Action(action) 

        self.steps += 1

        # We changed the reward to be the time it survived (increment by 1 each valid move)
        # moved, reward = self._move(action)
        moved, _ = self._move(action)
        reward = 1 if moved else 0
        
        if not moved:
            self.steps -= 1  
            return self.board.copy(), 0, False  # reward = 0, step--, done = False

        self.total_score += reward
        self._add_tile()
        done = self._is_game_over()
        return self.board.copy(), reward, done

    # --- Movement / Merge Logic ---
    # _move handles all four directions by rotating the board
    def _move(self, action):
        # Apply move-left logic after rotating
        rot_map = {
            Action.LEFT: 0,
            Action.UP: 1,
            Action.RIGHT: 2,
            Action.DOWN: 3,
        }

        rot = rot_map[action]

        # Bacically rotate the board to use left-move logic, then rotate back
        board = np.rot90(self.board, k=rot)
        moved, reward = self._move_left(board)
        board = np.rot90(board, k=(4 - rot) % 4) 
        self.board = board
        return moved, reward

    # Helper function for _move 
    def _move_left(self, board):
        moved = False
        reward = 0
        new_board = np.zeros_like(board)

        for i in range(4):
            row = board[i]
            row = row[row != 0]  

            merged_row = []
            skip = False
            local_reward = 0

            for j in range(len(row)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(row) and row[j] == row[j+1]:
                    merged = row[j] * 2 # Score increment = sum of two merged tiles
                    merged_row.append(merged)
                    local_reward += merged
                    skip = True
                else:
                    merged_row.append(row[j])

            merged_row += [0] * (4 - len(merged_row))  # pad to length 4
            new_board[i] = merged_row

            if not np.array_equal(new_board[i], board[i]):
                moved = True
            reward += local_reward

        board[:] = new_board
        return moved, reward

    # --- Tile Spawn ---
    def _add_tile(self):
        empty = np.argwhere(self.board == 0)
        if empty.size == 0:
            return
        r, c = empty[self.rng.integers(len(empty))]
        self.board[r, c] = 4 if self.rng.random() < 0.1 else 2

    # --- Terminal Check ---
    def _is_game_over(self):
        # Either 2048 tile exists or no moves possible
        if (self.board >= 2048).any():
            return True
        if (self.board == 0).any():
            return False
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j+1] or self.board[j, i] == self.board[j+1, i]:
                    return False

        return True


# --- Self-Test when run standalone ---
if __name__ == "__main__":
    env = Game2048Env(seed=0)
    s = env.reset()
    print("Initial Board:\n", s)
    print(f"Total Score: {env.total_score}")

    for a in Action:
        ns, r, d = env.step(a)
        print(f"\nAction {a.name}: reward={r}, done={d}\n{ns}, Total Score: {env.total_score}, Steps: {env.steps}")