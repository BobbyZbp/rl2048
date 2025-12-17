# rl2048

**Reinforcement Learning agent for the game 2048**

A personal project exploring **Deep Reinforcement Learning** for learning strong policies in the 2048 game.  
The agent is trained using **Deep Q-Learning (DQN)** with experience replay and target networks, aiming to learn stable mid-game strategies and push toward higher tiles (512 / 1024 / beyond).

---

## Overview

2048 is a deterministic sliding-tile game with stochastic tile spawning.  
This project formulates the game as a **Markov Decision Process (MDP)** and trains a neural network to approximate the action-value function:

\[
Q(s, a) \approx \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]
\]

Key challenges:
- Sparse and delayed rewards  
- Large state space with strong spatial structure  
- Long-horizon credit assignment  

---

## Key Features

-  Custom 2048 environment (deterministic transitions + stochastic tile spawn)
-  DQN agent with:
  - Target network
  - Experience replay
  - ε-greedy exploration
-  Training statistics tracking
-  Separate evaluation / play script

---

