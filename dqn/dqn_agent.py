import math
import random
from collections import deque
from typing import Deque, Tuple
import numpy as np
import torch
import torch.nn as nn
from dqn.dqn_model import DQN


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class DQNAgent:
    def __init__(
        self,
        num_actions: int = 4,
        gamma: float = 1.0,
        lr: float = 1e-4,
        memory_size: int = 50_000,
        batch_size: int = 64,
        tau: float = 5e-3,          # soft target update rate (θ' ← τθ + (1-τ)θ')
        eps_start: float = 0.5,     # Maybe change to 0.5, but turns out 1.0 has better performance. Actually not much difference.
        eps_end: float = 0.1,
        eps_decay_steps: int = 20_000
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss: combines mean squared error (MSE) and mean absolute error (MAE)

        # Replay memory
        self.memory: Deque[Transition] = deque(maxlen=memory_size)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = max(1, eps_decay_steps)
        self.steps_done = 0  # counts action-selection steps

    # Compute epsilon threshold use the idea of "first quickly then slowly"
    def _eps_threshold(self) -> float:
        # exponential decay over steps
        t = self.steps_done
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-t / self.eps_decay_steps)

    def choose_action(self, state_vec: np.ndarray) -> int:
        self.steps_done += 1
        eps_thresh = self._eps_threshold()
        if random.random() < eps_thresh:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    def store(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.memory.append((s, a, r, s2, done))

    def _soft_update_target(self) -> None:
        # θ' ← τθ + (1-τ)θ'
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def train_step(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size) 
        s, a, r, s2, done = zip(*batch)

        # Fast stacking (avoid slow list-of-arrays tensor creation)
        s   = torch.tensor(np.array(s),  dtype=torch.float32, device=self.device)
        s2  = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        a   = torch.tensor(a,           dtype=torch.int64,   device=self.device).unsqueeze(1)
        r   = torch.tensor(r,       
                               dtype=torch.float32, device=self.device)
        done = torch.tensor(done,       dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_sa = self.policy_net(s).gather(1, a).squeeze(1) #scalar

        # Target: r + γ * max_a' Q_target(s', a') for non-terminal transitions
        with torch.no_grad():
            next_q_max = self.target_net(s2).max(dim=1)[0]
            target = r + self.gamma * next_q_max * (1.0 - done) # sca;ar

        loss = self.loss_fn(q_sa, target)
        print("q_sa:", q_sa.mean().item(), " target:", target.mean().item(), " loss:", loss.item())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)  # mild safety
        self.optimizer.step()

        # Soft update each training step (instead of every N steps)
        self._soft_update_target()




# reward has a bug in the original code
# 1. We should use the Expected future reward alone instead of cummulative reward.
# 2. Check why the variable target (steps or survival time) is not aligned with the reward definition in env/Game2048Env.py (example, the print statement in the train.py shows Fraction number for big tile number/(mathmatically wrong))
# 3. For now, the batch should be ignored first. 
# 4. weights updated, but the t is not as big as expected.(maybe caused by problem 1?)

# Small notes:
# Track the shape of Q_sa and target, and others variables if applicable.
