import numpy as np
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Game2048Env:
    def __init__(self, size=4, seed=None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.steps = 0

        self._add_tile()
        self._add_tile()

        # IMPORTANT: initialize board_value on the actual board state
        self.board_value = np.sum(self.board ** 2)
        return self.board.copy()

    def step(self, action):
        self.steps += 1

        moved, _ = self._move(action)

        if not moved:
            self.steps -= 1
            return self.board.copy(), 0.0, True

        # Spawn tile first
        self._add_tile()

        # Compute potential AFTER spawn, aligned with returned state
        new_board_value = np.sum(self.board ** 2)

        reward = 1.0
        if new_board_value > self.board_value:
            reward += np.log2(new_board_value - self.board_value)

        self.board_value = new_board_value

        done = self._is_game_over() or np.any(self.board >= 2048)
        return self.board.copy(), reward, done

    def _add_tile(self):
        empty = np.argwhere(self.board == 0)
        if len(empty) == 0:
            return

        i, j = empty[self.rng.integers(len(empty))]
        self.board[i, j] = 2 if self.rng.random() < 0.9 else 4

    def _move(self, action):
        rot_map = {
            Action.LEFT: 0,
            Action.UP: 1,
            Action.RIGHT: 2,
            Action.DOWN: 3,
        }

        k = rot_map[action]
        rotated = np.rot90(self.board, k)
        new_board, moved = self._move_left(rotated)
        self.board = np.rot90(new_board, -k)

        return moved, None

    def _move_left(self, board):
        new_board = np.zeros_like(board)
        moved = False

        for i in range(self.size):
            row = board[i]
            nonzero = row[row != 0]
            merged = []
            skip = False

            for j in range(len(nonzero)):
                if skip:
                    skip = False
                    continue

                if j + 1 < len(nonzero) and nonzero[j] == nonzero[j + 1]:
                    merged.append(nonzero[j] * 2)
                    skip = True
                else:
                    merged.append(nonzero[j])

            merged = np.array(merged, dtype=np.int64)
            new_board[i, :len(merged)] = merged

            if not np.array_equal(row, new_board[i]):
                moved = True

        return new_board, moved

    def _is_game_over(self):
        if np.any(self.board == 0):
            return False

        for i in range(self.size):
            for j in range(self.size):
                if i + 1 < self.size and self.board[i, j] == self.board[i + 1, j]:
                    return False
                if j + 1 < self.size and self.board[i, j] == self.board[i, j + 1]:
                    return False

        return True
