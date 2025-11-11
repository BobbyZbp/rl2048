from env.Game2048Env import Game2048Env
from dqn.dqn_agent import DQNAgent
from utils.one_hot import one_hot_encode

def train(num_episodes: int = 5000, log_every: int = 50):
    env = Game2048Env()
    agent = DQNAgent()

    episode_scores = []
    episode_steps = []
    episode_max_tiles = []

    for ep in range(num_episodes):
        board = env.reset()
        state = one_hot_encode(board)

        total_reward = 0.0
        done = False
        max_tile = int(board.max())

        while not done:
            action = agent.choose_action(state)
            next_board, reward, done = env.step(action)
            next_state = one_hot_encode(next_board)

            agent.store(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += float(reward)
            max_tile = max(max_tile, int(next_board.max()))
            # print the trajectory of the last episode
            # print("Final Episode Board:\n", next_board)
            # print(f"Final Episode Total Score: {total_reward}, Max Tile: {max_tile}")

        episode_scores.append(total_reward)
        episode_max_tiles.append(max_tile)


        if (ep + 1) % 1 == 0:
            avg_score = sum(episode_scores[-log_every:]) / log_every
            best_tile = max(episode_max_tiles[-log_every:])
            eps = agent._eps_threshold()  # current epsilon

            print(
                f"[Episode {ep+1:5d}] "
                f"AvgScore={avg_score:7.1f}, "
                f"BestTile={best_tile:4d}, "
                f"Epsilon={eps:.4f},"
                f"Board:\n{next_board},"
                )
            #print the weights of the policy network
            for name, param in agent.policy_net.named_parameters():
                print(f"{name}: {param.data.norm():.4e}")
       

if __name__ == "__main__":
    agent = DQNAgent()
    print("Running on device:", agent.device)
    train()
