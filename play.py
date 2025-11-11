import time
import torch
import numpy as np

from env.game2048_env import Game2048Env
from agents.dueling_dqn_per import DuelingDDQNAgent

CKPT = "dqn_dueling_per.pt"

env = Game2048Env()
agent = DuelingDDQNAgent()
agent.load(CKPT)

def pretty(board):
    return "\n".join(" ".join(f"{x:4d}" for x in row) for row in board)

state, _ = env.reset()
total = 0.0

while True:
    s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        action = int(agent.model(s).argmax(dim=1).item())

    state, reward, done, _, _ = env.step(action)
    total += reward
    print("\nBoard:\n", pretty(state),
          f"\nStep reward: {reward:.1f}  Total: {total:.1f}  MaxTile: {int(np.max(state))}")
    time.sleep(0.15)
    if done:
        print("\nGame over. Final Total reward:", total, " MaxTile:", int(np.max(state)))
        break
