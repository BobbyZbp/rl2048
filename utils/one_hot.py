import numpy as np

def one_hot_encode(board):
    # board: np.array shape (4,4) with values 0,2,4,...,2048 changed to log2: 0,1,2,...,11. (4 x 4 x 12)
    levels = np.zeros((4,4,12), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            v = board[i,j]
            if v > 0:
                k = int(np.log2(v))
                levels[i,j,k] = 1.0
    return levels.flatten()  # shape: (192,)