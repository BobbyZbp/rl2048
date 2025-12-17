from env.Game2048Env import Game2048Env
from dqn.dqn_agent import DQNAgent
from utils.one_hot import one_hot_encode
import torch
import numpy as np
from tqdm import tqdm

def train(num_episodes: int = 10000, log_every: int = 50):
    env = Game2048Env()
    agent = DQNAgent()
    agent.env = env  # Provide env to agent for can_move checks

    episode_scores = []
    episode_steps = []
    episode_max_tiles = []

    pbar = tqdm(range(num_episodes), desc="Training Episodes")
    for ep in pbar:
        board = env.reset()
        state = one_hot_encode(board)

        total_reward = 0.0
        done = False
        max_tile = int(board.max())

        step_idx = 0   #  Q-VALUE PRINT helper

        while not done:
            # -----------------------------------------------------------
            # <<< Q-VALUE PRINT: show current board and Q(s,*)
            # -----------------------------------------------------------
            with torch.no_grad():
                s_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                q_vals = agent.policy_net(s_tensor).squeeze(0).cpu().numpy()
            #print(f"\n[Episode {ep+1} | Step {step_idx}]")
            #print("Current Board:\n", board)
            #print("Q-values (UP, DOWN, LEFT, RIGHT):", q_vals)
            # -----------------------------------------------------------

            action = agent.choose_action(state)
            next_board, reward, done = env.step(action)
            next_state = one_hot_encode(next_board)

            agent.store(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += float(reward)
            max_tile = max(max_tile, int(next_board.max()))

            board = next_board
            step_idx += 1

        episode_scores.append(total_reward)
        episode_max_tiles.append(max_tile)

        if (ep + 1) % 1 == 0:
            k = min(log_every, len(episode_scores))

            avg_score = sum(episode_scores[-log_every:]) / k
            best_tile = max(episode_max_tiles[-k:])
            eps = agent._eps_threshold()  # current epsilon
            pbar.set_description(f"Episode {ep+1}")
            pbar.set_postfix({
                "AvgScore": f"{avg_score:7.1f}",
                "BestTile": f"{best_tile:4d}",
                "Epsilon": f"{eps:.4f}",
            })
            # print(
            #     f"[Episode {ep+1:5d}] "
            #     f"AvgScore={avg_score:7.1f}, "
            #     f"BestTile={best_tile:4d}, "
            #     f"Epsilon={eps:.4f},"
            #     f"Board:\n{next_board},"
            #     )

if __name__ == "__main__":
    agent = DQNAgent()
    print("Running on device:", agent.device)
    train()
