import numpy as np
from Game2048Env import Game2048Env, Action

# ==========================================================
# Reset / Initialization
# ==========================================================

def test_reset_board_shape_and_two_tiles():
    env = Game2048Env(seed=0)
    s = env.reset()
    assert s.shape == (4, 4)
    assert np.count_nonzero(s) == 2   # exactly two tiles initially


# ==========================================================
# Core Move Left Logic (no rotation)
# ==========================================================

def test_move_left_simple_merge():
    env = Game2048Env()
    env.board = np.array([
        [2,2,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
    ])
    moved, reward = env._move(Action.LEFT)
    assert moved is True
    assert reward == 4
    assert np.array_equal(env.board[0], np.array([4,0,0,0]))


def test_move_left_no_double_merge():
    env = Game2048Env()
    env.board = np.array([
        [2,2,2,2],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
    ])
    moved, reward = env._move(Action.LEFT)
    assert moved is True
    assert reward == 8
    assert np.array_equal(env.board[0], np.array([4,4,0,0]))


# ==========================================================
# Directional Move Tests (via rotation logic)
# ==========================================================

def test_move_up_rotation():
    env = Game2048Env()
    env.board = np.array([
        [2,0,0,0],
        [2,0,0,0],
        [4,0,0,0],
        [4,0,0,0],
    ])
    env._move(Action.UP)
    assert np.array_equal(env.board[:,0], np.array([4,8,0,0]))


def test_move_right():
    env = Game2048Env()
    env.board = np.array([
        [2,2,4,4],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
    ])
    env._move(Action.RIGHT)
    assert np.array_equal(env.board[0], np.array([0,0,4,8]))


def test_move_down():
    env = Game2048Env()
    env.board = np.array([
        [2,0,0,0],
        [2,0,0,0],
        [2,0,0,0],
        [2,0,0,0],
    ])
    env._move(Action.DOWN)
    assert np.array_equal(env.board[:,0], np.array([0,0,4,4]))


# ==========================================================
# Illegal Move Behavior
# ==========================================================

def test_illegal_move_no_change_no_reward():
    env = Game2048Env()
    env.board = np.array([
        [2,4,8,16],
        [32,64,128,256],
        [512,1024,2048,4096],
        [8,16,32,64],
    ])
    before = env.board.copy()
    moved, reward = env._move(Action.LEFT)
    assert moved is False
    assert reward == 0
    assert np.array_equal(env.board, before)


# ==========================================================
# Game Over Detection
# ==========================================================

def test_game_over_correct_detection():
    env = Game2048Env()
    env.board = np.array([
        [2,4,2,4],
        [4,2,4,2],
        [2,4,2,4],
        [4,2,4,2],
    ])
    assert env._is_game_over() is True


def test_game_over_idempotent():
    env = Game2048Env()
    env.board = np.array([
        [2,4,2,4],
        [4,2,4,2],
        [2,4,2,4],
        [4,2,4,2],
    ])
    assert env._is_game_over() == env._is_game_over() == True


# ==========================================================
# Tile Spawning Behavior
# ==========================================================

def test_tile_spawn_distribution():
    env = Game2048Env(seed=0)
    env.board[:] = 0
    counts = {2: 0, 4: 0}

    for _ in range(200):
        env.board[:] = 0
        env._add_tile()
        tile = env.board[env.board != 0][0]
        counts[tile] += 1

    assert counts[2] > counts[4]   # most are 2s
    assert counts[4] > 0           # some 4s must exist


# ==========================================================
# Score / Reward Consistency
# ==========================================================

def test_score_accumulation_matches_rewards():
    env = Game2048Env()
    env.board = np.array([
        [2,2,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
    ])
    _, r, _ = env.step(Action.LEFT)
    assert env.total_score == r == 4